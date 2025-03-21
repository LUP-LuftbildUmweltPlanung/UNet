import glob
import os
import shutil
import sys
import urllib.parse
import warnings
import numpy as np
import torch
import socket
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import time
from tqdm import tqdm
from pathlib import Path
from osgeo import gdal
from fastai.learner import load_learner
from sklearn.metrics import confusion_matrix, classification_report
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# save the predicted tiles
def store_tif(output_folder, output_array, dtype, geo_transform, geo_proj, nodata_value, class_zero=False):
    """Stores a tif file in a specified folder."""
    driver = gdal.GetDriverByName('GTiff')

    if len(output_array.shape) == 3:
        out_ds = driver.Create(str(output_folder), output_array.shape[2], output_array.shape[1], output_array.shape[0],
                               dtype)
    else:
        out_ds = driver.Create(str(output_folder), output_array.shape[1], output_array.shape[0], 1, dtype)
    out_ds.SetGeoTransform(geo_transform)

    out_ds.SetProjection(geo_proj)

    if class_zero:
        # Process the output array to handle class definitions
        processed_array = np.where(output_array == 0, nodata_value, output_array - 1)  # Class 0 as NaN and decrement other classes by 1
    else:
        processed_array = output_array


    if len(processed_array.shape) == 3:
        for b in range(processed_array.shape[0]):
            out_ds.GetRasterBand(b + 1).WriteArray(processed_array[b])
    else:
        out_ds.GetRasterBand(1).WriteArray(processed_array)

    # Loop through the image bands to set nodata
    if nodata_value is not None:
        for i in range(1, out_ds.RasterCount + 1):
            # Set the nodata value of the band
            out_ds.GetRasterBand(i).SetNoDataValue(nodata_value)

    out_ds.FlushCache()
    out_ds = None


# create valid figures
def plot_valid_predict(output_folder, predict_path, regression=False, merge=False, class_zero=False):
    if merge:
        raise ValueError("It's not possible to calculate the confusion matrix with merged tiles")
    elif regression:
        raise ValueError("This function is just for classification problems")

    # Create a new folder to save the figures
    valid_path = os.path.join(output_folder, "Valid_figures")
    os.makedirs(valid_path, exist_ok=True)

    # Replace the last part of the truth_label path
    truth_label = predict_path.replace('img_tiles', 'mask_tiles')

    y_true = []
    y_pred = []

    for file_name in os.listdir(output_folder):
        if file_name.endswith('.tif'):
            pred_path = os.path.join(output_folder, file_name)
            true_path = os.path.join(truth_label, file_name)

            with rasterio.open(pred_path) as src_pred:
                pred_data = src_pred.read(1).astype(np.int64)  # Assuming single band for class labels

            with rasterio.open(true_path) as src_true:
                true_data = src_true.read(1).astype(np.int64)  # Assuming single band for class labels

            # Determine the most frequent class in the tile
            pred_class = np.argmax(np.bincount(pred_data.flatten()))
            true_class = np.argmax(np.bincount(true_data.flatten()))

            if class_zero:
                true_class = true_class[true_class != 0] - 1


            y_true.append(true_class)
            y_pred.append(pred_class)

    if not y_true or not y_pred:
        raise ValueError("No valid tiles found for evaluation")

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=1)
    # ✅ Extract only class names (exclude "accuracy", "macro avg", etc.)
    class_labels = [str(label) for label in class_report.keys() if
                    label not in ["accuracy", "macro avg", "weighted avg"]]

    # Convert the classification report dictionary into a DataFrame for visualization
    dataframe = pd.DataFrame(class_report).transpose()

    # ✅ Save classification report as an image
    cm_path = os.path.join(valid_path, "Confusion_Matrix.png")
    classification_report_path = os.path.join(valid_path, "classification_report.png")

    # ✅ Plot and save the classification report heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(dataframe.iloc[:-1, :-1], annot=True, fmt='.2f', cmap='crest')  # Remove support row
    plt.title('Classification Report')
    plt.savefig(classification_report_path)
    plt.close()

    # ✅ Plot and save the confusion matrix heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='crest', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(cm_path)
    plt.close()

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report (as dictionary):")
    print(class_report)

    return cm, class_report, cm_path, classification_report_path


def save_predictions(predict_model, predict_path, regression, merge=False, all_classes=False, specific_class=None,
                     large_file=False, AOI=None, year=None, validation_vision=True, class_zero=False):
    """
    Runs a prediction on all tiles within a folder and stores predictions in the predict_tiles folder

    Parameters:
    -----------
        learn :             Unet learner containing a Unet prediction model
        path :              Path containing tiles for prediction
        regression :        If the prediction should output continuous values (else: classification)
        merge :             If predicted tiles should be merged to a single .tif file (default=False)
        all_classes :       If the prediction should contain all prediction values for all classes (default=False)
        specific_class :    Only prediction values for this specific class will be stored (default=None)
    """
    # ✅ Get PC name dynamically
    pc_name = socket.gethostname()
    with mlflow.start_run(run_name=f"Prediction_{os.path.basename(predict_model).split('.')[0]}"):
        # ✅ Log parameters
        mlflow.log_param("predict_model", predict_model)
        mlflow.log_param("predict_path", predict_path)
        mlflow.log_param("regression", regression)
        mlflow.log_param("merge", merge)
        mlflow.log_param("all_classes", all_classes)
        mlflow.log_param("specific_class", specific_class)
        mlflow.log_param("large_file", large_file)
        mlflow.log_param("AOI", AOI)
        mlflow.log_param("year", year)
        mlflow.log_param("validation_vision", validation_vision)
        mlflow.log_param("class_zero", class_zero)

        # Set the MLflow Source Name with PC name
        mlflow.set_tag("mlflow.source.name", f"{pc_name}_params_and_main.py")
        # Log PC name as a parameter in MLflow
        mlflow.log_param("pc_name", pc_name)

        learn = load_learner(Path(predict_model))

        path = Path(predict_path)

        # Get MLflow artifact directory
        artifact_uri = mlflow.get_artifact_uri()

        # Detect if the OS is Windows
        is_windows = sys.platform.startswith("win")

        # ✅ Apply different path handling for Windows vs Linux
        if is_windows:
            parsed_uri = urllib.parse.urlparse(artifact_uri)
            if parsed_uri.scheme == "file":
                output_folder_mlflow = Path(parsed_uri.path.lstrip("/")) / "predictions"
            else:
                output_folder_mlflow = Path(artifact_uri) / "predictions"

            # Convert to absolute Windows path
            output_folder_mlflow = output_folder_mlflow.resolve()
        else:
            # On Linux/Mac, just use the artifact path normally
            output_folder_mlflow = Path(artifact_uri) / "predictions"

        # ✅ Ensure the directory exists
        os.makedirs(output_folder_mlflow, exist_ok=True)
        mlflow.log_param("output_folder_mlflow", str(output_folder_mlflow))

        print(f"📁 Final MLflow artifact path: {output_folder_mlflow}")

        if not merge:
            output_folder = path.parent / ('predicted_tiles_' + Path(predict_model).stem)
        else:
            output_folder = path.parent

        model_name = os.path.basename(predict_model).split('.')[0]

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        tiles = glob.glob(str(path) + "\\*.tif")

        # create necessary variables to track merge
        if merge:
            geoproj_for_merge = None
            predictions_for_merge = []
            predictions_for_merge_size = 0
            geotrans_for_merge = []

        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(f'Started at: {current_time}')

        for i in tqdm(range(len(tiles)), desc='Processing tiles'):
            # print(f'Current progress: {i}/{len(tiles)}')
            tile_preds = learn.predict(Path(tiles[i]), with_input=False)


            class_lst = []

            if regression:
                for cl in range(len(tile_preds[1])):
                    class_lst.append(tile_preds[1][cl])
            else:
                for cl in range(len(tile_preds[2])):
                    class_lst.append(tile_preds[2][cl])

            class_lst = torch.stack(class_lst)

            # go through all predictions and store their physical coordinates and check, that all have the same projection
            if merge:
                img_ds_proj = gdal.Open(str(tiles[i]))

                if geoproj_for_merge is None:
                    geoproj_for_merge = img_ds_proj.GetProjection()
                elif geoproj_for_merge is not None and geoproj_for_merge != img_ds_proj.GetProjection():
                    warnings.warn("Geoprojection is not the same for all prediction tiles.")

                ulx, xres, xskew, uly, yskew, yres = img_ds_proj.GetGeoTransform()
                class_lst = class_lst.numpy()

                if large_file and np.max(class_lst) <= 1:
                    class_lst *= ((128 / 4) - 1)
                    class_lst = np.around(class_lst).astype(np.int8)
                predictions_for_merge.append(class_lst)
                predictions_for_merge_size += class_lst.nbytes
                geotrans_for_merge.append([ulx, img_ds_proj.RasterXSize, xres, uly, img_ds_proj.RasterYSize, yres])


            else:
                if regression:
                    pass
                elif all_classes:
                    pass
                elif specific_class is None:
                    # for decoded argmax value
                    class_lst = class_lst.argmax(axis=0)
                else:
                    # for probabilities of specific class [1] -> klasse 1
                    class_lst = class_lst[specific_class]
                img_ds_proj = gdal.Open(str(tiles[i]))
                geotrans = img_ds_proj.GetGeoTransform()
                geoproj = img_ds_proj.GetProjection()

                if "float" in str(class_lst.dtype):
                    dtype = gdal.GDT_Float32
                else:
                    dtype = gdal.GDT_Byte

                # ✅ Save prediction in both locations
                output_file_mlflow = str(output_folder_mlflow / os.path.basename(tiles[i]))

                if large_file and np.max(class_lst.numpy()) <= 1 and (all_classes or specific_class):
                    class_lst = class_lst.numpy()
                    class_lst *= ((128 / 4) - 1) # values range from 0 to 1. to values range from 0 to 31, stored efficiently as GDT_Byte, saving disk space
                    class_lst = np.around(class_lst).astype(np.int8)
                    dtype = gdal.GDT_Byte
                    store_tif(str(output_folder) + "\\" + os.path.basename(tiles[i]), class_lst, dtype, geotrans, geoproj,
                              None, class_zero)
                    store_tif(str(output_folder_mlflow / os.path.basename(tiles[i])), class_lst, dtype, geotrans,
                              geoproj, None, class_zero)

                else:
                    store_tif(str(output_folder) + "\\" + os.path.basename(tiles[i]), class_lst.numpy(), dtype, geotrans,
                              geoproj, None, class_zero)
                    store_tif(str(output_folder_mlflow / os.path.basename(tiles[i])), class_lst.numpy(), dtype,
                              geotrans, geoproj, None, class_zero)

        if validation_vision:
            cm, class_report, cm_path, class_report_path = plot_valid_predict(output_folder, predict_path, regression, merge, class_zero)
            # ✅ Log metrics correctly
            mlflow.log_metric("accuracy", class_report["accuracy"] if "accuracy" in class_report else class_report.get(
                "weighted avg", {}).get("precision", 0))
            mlflow.log_metric("precision", class_report["weighted avg"]["precision"])
            mlflow.log_metric("recall", class_report["weighted avg"]["recall"])
            mlflow.log_metric("f1_score", class_report["weighted avg"]["f1-score"])

            # ✅ Create a structured DataFrame for detailed evaluation results
            eval_results_df = pd.DataFrame({
                "metric": ["accuracy", "precision", "recall", "f1_score"],
                "value": [
                    class_report["accuracy"],
                    class_report["weighted avg"]["precision"],
                    class_report["weighted avg"]["recall"],
                    class_report["weighted avg"]["f1-score"]
                ]
            })

            # ✅ Save evaluation results as a CSV for consistency
            def get_local_path_from_artifact_uri(artifact_uri: str, experiment_id: str) -> Path:
                """
                Convert MLflow Linux-style artifact URI to a proper Windows path.
                Inserts experiment_id between mlruns and run_id.
                """
                from urllib.parse import urlparse
                from pathlib import Path

                parsed = urlparse(artifact_uri)
                linux_str = parsed.path.replace("\\", "/")

                root_prefix = "/home/embedding/Data_Center/qnap3b"
                if not linux_str.startswith(root_prefix):
                    raise ValueError(f"❌ Unexpected path format. Got: {linux_str}")

                # Extract relative part after root
                relative = linux_str[len(root_prefix):].lstrip("/")
                parts = Path(relative).parts

                try:
                    # Locate 'mlruns' and isolate the run_id
                    mlruns_idx = parts.index("mlruns")
                    run_id = parts[mlruns_idx + 1]
                    rest = parts[mlruns_idx + 2:]  # e.g., ['artifacts', 'models']

                    corrected = Path("mlruns") / run_id / Path(*rest)
                    return Path("N:/MnD/hub/mlflow") / corrected
                except Exception as e:
                    raise ValueError(f"❌ Could not parse run_id and artifact path from: {parts}\n{e}")

            artifact_dir = get_local_path_from_artifact_uri(
                mlflow.get_artifact_uri(),
                experiment_id=mlflow.active_run().info.experiment_id
            )
            os.makedirs(artifact_dir, exist_ok=True)

            eval_results_csv = os.path.join(output_folder, "unet_evaluation_results.csv")
            eval_results_df.to_csv(eval_results_csv, index=False)
            # ✅ Manually copy prediction output files to artifact folder
            try:
                for f in [eval_results_csv, cm_path, class_report_path]:
                    if os.path.exists(f):
                        shutil.copy2(f, artifact_dir)
                        print(f"📁 Copied to artifact folder: {Path(f).name}")
                    else:
                        print(f"⚠️ File not found: {f}")
            except Exception as e:
                print(f"❌ Error copying prediction outputs to artifact folder: {e}")
        if merge:
            # go through the information for all tiles, find upper left most corner and lower right most corner
            # --> these define the extend of the final output
            # remember: lower left to upper right
            geotrans_for_merge = np.array(geotrans_for_merge)
            upleft_x_full = np.min(geotrans_for_merge[:, 0])
            upleft_y_full = np.max(geotrans_for_merge[:, 3])
            xmax_raster = np.argmax(geotrans_for_merge[:, 0])
            ymin_raster = np.argmin(geotrans_for_merge[:, 3])
            # calculate coordinate from array index
            lowright_x_full = np.max(geotrans_for_merge[:, 0]) + geotrans_for_merge[
                xmax_raster, 1] * geotrans_for_merge[xmax_raster, 2]
            lowright_y_full = np.min(geotrans_for_merge[:, 3]) + geotrans_for_merge[
                ymin_raster, 4] * geotrans_for_merge[ymin_raster, 5]

            if len(set(geotrans_for_merge[:, 1])) != 1 or len(set(geotrans_for_merge[:, 4])) != 1:
                warnings.warn("Not all tiles have the same resolution.")

            x_length = (round((lowright_x_full - upleft_x_full) / geotrans_for_merge[0, 2]))
            y_length = (round((lowright_y_full - upleft_y_full) / geotrans_for_merge[0, 5]))

            if large_file:
                dty = np.int8
            else:
                dty = np.float32

            # create array to contain raster data
            merged_raster = np.zeros((predictions_for_merge[0].shape[0], y_length, x_length), dtype=dty)
            print(f'True merged raster size: {merged_raster.nbytes / (1024 ** 2): .1f}MB.')
            # create array to contain if pixel is the sum of 1, 2, or 4 tiles
            # (1 - single tile, 2 - two overlapping edges, 4 - overlapping corners)
            merge_counter = np.zeros((predictions_for_merge[0].shape[0], y_length, x_length),
                                     dtype=np.int8)

            # for each image
            for i, (pred, geotrans) in enumerate(zip(predictions_for_merge, geotrans_for_merge)):
                # find location
                upleft_x = round((geotrans[0] - upleft_x_full) / geotrans[2])
                upleft_y = round((geotrans[3] - upleft_y_full) / geotrans[5])
                lowright_x = round((geotrans[0] + geotrans[1] * geotrans[2] - upleft_x_full) / geotrans[2])
                lowright_y = round((geotrans[3] + geotrans[4] * geotrans[5] - upleft_y_full) / geotrans[5])

                # place raster
                merged_raster[:, upleft_y:lowright_y, upleft_x:lowright_x] += pred
                # increase counter for placed rasters
                merge_counter[:, upleft_y:lowright_y, upleft_x:lowright_x] += np.ones_like(pred, dtype=np.int8)

                # delete tile to use less space
                predictions_for_merge[i] = []

            if regression:
                merged_raster = merged_raster[0]
                merge_counter = merge_counter[0]

                # divide raster by counter to turn overlaps into realistic values
                merged_raster[merge_counter > 0] /= merge_counter[merge_counter > 0]

                # set raster to -9999 where no predictions were placed
                nodata = -9999
                merged_raster[merge_counter == 0] = nodata  # maybe change to merged_raster[merged_raster == 0] = nodata
            else:
                if large_file:

                    # Create a mask that is True where merge_counter is positive - uses less space than [condition] indexing (?)
                    mask = merge_counter > 0
                    # Apply the mask to both arrays and perform the division
                    merged_raster[mask] //= merge_counter[mask]

                else:
                    merged_raster[merge_counter > 0] /= merge_counter[merge_counter > 0]

                # divide raster by counter to turn overlaps into realistic values
                if all_classes:
                    pass
                elif specific_class is None:
                    # for decoded argmax value
                    # merged_raster = np.where(merged_raster.max(axis=0) < 0.1, 14, merged_raster.argmax(axis=0)) #check if this is still relevant?
                    merged_raster = merged_raster.argmax(axis=0)
                else:
                    # for probabilities of specific class [1] -> klasse 1
                    merged_raster = merged_raster[specific_class]

                # define nodata for classification (either background class or where no Tiles were placed)
                nodata = None

            if "float" in str(merged_raster.dtype):
                dtype = gdal.GDT_Float32
            else:
                dtype = gdal.GDT_Byte

            # Define the parameters for the name of output:
            output_file_name_parts = [AOI, year, model_name, "prediction"]
            output_file_name = "_".join(filter(None, output_file_name_parts)) + ".tif"
            output_file = os.path.join(output_folder, output_file_name)
            print(output_file)



            store_tif(output_file, merged_raster, dtype,
                      [upleft_x_full, geotrans_for_merge[0, 2], 0.0, upleft_y_full, 0.0, geotrans_for_merge[0, 5]],
                      geoproj_for_merge, nodata, class_zero)

            output_file_mlflow = os.path.join(output_folder_mlflow, output_file_name)
            store_tif(output_file_mlflow, merged_raster, dtype,
                      [upleft_x_full, geotrans_for_merge[0, 2], 0.0, upleft_y_full, 0.0, geotrans_for_merge[0, 5]],
                      geoproj_for_merge, nodata, class_zero)



            print(f"Prediction stored in {output_folder}.")
