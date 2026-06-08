import glob
import os
import warnings
import numpy as np
import torch
import socket
import mlflow
import mlflow.pytorch
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
import pathlib
import albumentations as A
from mosaic_2 import *

def load_fastai_model_flexible(model_uri):
    """
    Load a FastAI model from a local path, MLflow run URI, or artifact URI.

    Parameters:
        model_uri (str or Path):
            - Local path: "/path/to/model.pkl"
            - MLflow run artifact: "mlflow-artifacts:/<exp_id>/<run_id>/artifacts/<model.pkl>"
            - Run ID style: "runs:/<run_id>/model.pkl"

    Returns:
        Learner object loaded via fastai.
    """

    # Save the original PosixPath to restore it later
    temp = pathlib.PosixPath

    # Redirect PosixPath to WindowsPath to avoid issues on Windows
    pathlib.PosixPath = pathlib.WindowsPath

    try:
        model_uri_str = str(model_uri)  # Convert Path object to string

        # Ensure the path is formatted correctly
        model_uri_str = model_uri_str.replace("\\", "/")  # Replace backslashes with forward slashes

        if model_uri_str.startswith("mlflow-artifacts:/") or model_uri_str.startswith("runs:/"):
            print(f"Downloading model artifact from MLflow: {model_uri_str}")
            local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri_str)
            return load_learner(local_path, cpu=False)
        elif pathlib.Path(model_uri_str).exists():
            print(f"Loading model from local path: {model_uri_str}")
            return load_learner(model_uri_str, cpu=False)
        else:
            raise ValueError(f"Unsupported or non-existent model path: {model_uri_str}")
    finally:
        # Restore the original PosixPath to avoid side effects
        pathlib.PosixPath = temp


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
    valid_path = output_folder / "Valid_figures"
    os.makedirs(valid_path, exist_ok=True)

    # Replace the last part of the truth_label path
    truth_label = Path(str(predict_path).replace('img_tiles', 'mask_tiles'))

    y_true = []
    y_pred = []

    for file_name in os.listdir(output_folder):
        if file_name.endswith('.tif'):
            pred_path = output_folder / file_name
            true_path = truth_label / file_name

            with rasterio.open(pred_path) as src_pred:
                pred_data = src_pred.read(1).astype(np.int64)  # Assuming single band for class labels

            with rasterio.open(true_path) as src_true:
                true_data = src_true.read(1).astype(np.int64)  # Assuming single band for class labels

            # If class_zero is true, shift class values accordingly
            if class_zero:
                # true_class = true_class[true_class != 0] - 1
                true_data[true_data != 0] -= 1

            y_true.extend(true_data.flatten())
            y_pred.extend(pred_data.flatten())

    if not y_true or not y_pred:
        raise ValueError("No valid tiles found for evaluation")

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=1)
    #  Extract only class names (exclude "accuracy", "macro avg", etc.)
    class_labels = [str(label) for label in class_report.keys() if
                    label not in ["accuracy", "macro avg", "weighted avg"]]

    # Convert the classification report dictionary into a DataFrame for visualization
    dataframe = pd.DataFrame(class_report).transpose()

    #  Save classification report as an image
    classification_report_path = os.path.join(valid_path, "classification_report.png")

    # Keep only class label rows (like 0, 1, 2...) and drop "support" column
    filtered_df = dataframe.loc[dataframe.index.str.isdigit(), ['precision', 'recall', 'f1-score']]

    #  Plot and save the classification report heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(filtered_df.astype(float), annot=True, fmt='.2f', cmap='crest')
    plt.title('Classification Report')
    plt.savefig(classification_report_path)
    plt.close()

    #  Plot and save the confusion matrix heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='crest', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    confusion_matrix_path = valid_path / "Confusion_Matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close()

    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report (as dictionary):")
    print(class_report)

    return cm, class_report, confusion_matrix_path, classification_report_path


def predict_with_tta(learn, tile_path):
    """
    Perform Test-Time Augmentation (TTA) on a single (RGB,RGBI,RGBI+nDOM) tile and return the best per-pixel probabilities.

    This function applies four transformations to the input image:
        - Original
        - Horizontal flip
        - Vertical flip
        - Both horizontal and vertical flip

    For each pixel, it selects the prediction from the transformation with the highest confidence.

    Args:
        learn: A fastai Learner object containing the trained segmentation model.
        tile_path (str): Path to the raster tile (RGBI with 5 channels) to predict on.

    Returns:
        tuple: A tuple of (None, None, final_probs)
            - final_probs (torch.Tensor): Tensor of shape [C, H, W] with the best per-pixel class probabilities.

    Notes:
        - The final probabilities are selected per pixel based on maximum confidence across the TTA transformations.
    """
    # Get the device (CPU or GPU) used by the model
    device = next(learn.model.parameters()).device

    # Read the image (RGBI with 5 channels) using rasterio
    with rasterio.open(tile_path) as src:
        image = src.read()  # Reads all bands, shape: [bands, H, W]
        image = image.transpose(1, 2, 0)  # Shape: [H, W, bands]

    # Automatically determine the scale factor
    if image.dtype == np.uint16:
        scale = 65535.0
    elif image.dtype == np.uint8:
        scale = 255.0
    else:
        raise ValueError(f"Unsupported dtype {image.dtype}")

    # Define the transformations to be applied during Test-Time Augmentation (TTA)
    transforms = [
        ("orig", A.NoOp(p=1.0)),  # No operation, original image
        ("h", A.HorizontalFlip(p=1.0)),  # Horizontal flip
        ("v", A.VerticalFlip(p=1.0)),  # Vertical flip
        ("hv", A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]))  # Both flips
    ]

    preds_list = []  # List to store predictions from each transformation

    # Loop through each transformation and predict
    for name, t in transforms:
        aug = t(image=image)
        img_aug = aug["image"]  # Get the augmented image
        # Convert the augmented image to tensor and move it to the correct device (GPU/CPU)
        tensor = torch.from_numpy(img_aug.astype(np.float32) / scale) \
                      .permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = learn.model(tensor)   # Predict: shape [1, 8, H, W]

        # Apply softmax to convert logits to probabilities
        prob = torch.softmax(pred, dim=1).squeeze(0)  # shape [8, H, W]

        # Reverse the transformation (flip back the image to the original orientation)
        if name == "h":
            prob = torch.flip(prob, dims=[2])  # Reverse horizontal flip
        elif name == "v":
            prob = torch.flip(prob, dims=[1])  # Reverse vertical flip
        elif name == "hv":
            prob = torch.flip(prob, dims=[1, 2])  # Reverse both horizontal and vertical flips

        # Add the reversed predictions to the list
        preds_list.append(prob)

    # Stack all predictions → [4, 8, H, W] (4 transformations)
    stacked = torch.stack(preds_list, dim=0)

    # STEP 1: Calculate confidence per prediction by taking the max probability across classes
    confidence = stacked.max(dim=1).values  # shape [4, H, W]

    # STEP 2: Find the best prediction index per pixel (which transformation gave the best result)
    best_idx = confidence.argmax(dim=0)   # shape [H, W]

    # STEP 3: Gather the best probabilities based on the best prediction index
    C, H, W = stacked.shape[1:]  # C = 8 (classes), H = height, W = width
    final_probs = torch.zeros((C, H, W), device="cpu")

    for i in range(stacked.shape[0]):  # Loop through each transformation
        mask = (best_idx == i)  # Find where the best index equals the current transformation
        final_probs[:, mask] = stacked[i].cpu()[:, mask]  # Gather the best probabilities

    # fastai-compatible return
    return (None, None, final_probs)

def save_predictions(predict_model, predict_path, regression, merge=False, all_classes=False, specific_class=None,
                     large_file=False, AOI=None, year=None, validation_vision=True, class_zero=False, TTA=False):
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
    #  Get PC name dynamically
    pc_name = socket.gethostname()
    with mlflow.start_run(run_name=f"Prediction_{os.path.basename(predict_model).split('.')[0]}"):
        #  Log parameters
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

        # Mlflow the path of the valid data
        # Create a minimal DataFrame with just the path
        df = pd.DataFrame([], columns=[])  # empty dataset

        # Log it using from_pandas with the folder path as the source
        dataset = mlflow.data.from_pandas(df, source=predict_path, name="prediction_tiles")
        mlflow.log_input(dataset, context="inference")

        print(f" Logged dataset folder path as input: {predict_path}")

        print(f" Logged dataset source path: {predict_path}")

       # learn = load_learner(Path(predict_model))
        learn = load_fastai_model_flexible(predict_model)

        path = Path(predict_path)

        if not merge:
            output_folder = path.parent / ('predicted_tiles_' + Path(predict_model).stem)
        else:
            output_folder = path.parent

        model_name = os.path.basename(predict_model).split('.')[0]

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        tiles = glob.glob(str(path) + "\\*.tif")

        # ---------------- Large file detection ----------------
        total_pixels = sum([gdal.Open(t).RasterXSize * gdal.Open(t).RasterYSize for t in tiles])
        threshold = 1e9  # example threshold in pixels, adjust as needed

        if total_pixels > threshold:
            print("⚠️ Large file detected. Using tile-wise output. Merge set to False.")
            merge = False

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
            if TTA:
                tile_preds = predict_with_tta(learn, Path(tiles[i]))
            else:
                tile_preds = learn.predict(Path(tiles[i]), with_input=False)
            # tile_preds = learn.predict(Path(tiles[i]), with_input=False)


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

                if large_file and np.max(class_lst.numpy()) <= 1 and (all_classes or specific_class):
                    class_lst = class_lst.numpy()
                    class_lst *= ((128 / 4) - 1) # values range from 0 to 1. to values range from 0 to 31, stored efficiently as GDT_Byte, saving disk space
                    class_lst = np.around(class_lst).astype(np.int8)
                    dtype = gdal.GDT_Byte
                    store_tif(str(output_folder) + "\\" + os.path.basename(tiles[i]), class_lst, dtype, geotrans, geoproj,
                              None, class_zero)

                else:
                    store_tif(str(output_folder) + "\\" + os.path.basename(tiles[i]), class_lst.numpy(), dtype, geotrans,
                              geoproj, None, class_zero)

        if validation_vision:
            cm, class_report, cm_path, class_report_path = plot_valid_predict(output_folder, predict_path, regression,
                                                                              merge, class_zero)

            #  Log metrics
            mlflow.log_metric("accuracy",
                              class_report.get("accuracy", class_report.get("weighted avg", {}).get("precision", 0)))
            mlflow.log_metric("precision", class_report["weighted avg"]["precision"])
            mlflow.log_metric("recall", class_report["weighted avg"]["recall"])
            mlflow.log_metric("f1_score", class_report["weighted avg"]["f1-score"])

            #  Save evaluation results as CSV
            eval_results_df = pd.DataFrame({
                "metric": ["accuracy", "precision", "recall", "f1_score"],
                "value": [
                    class_report.get("accuracy", 0),
                    class_report["weighted avg"]["precision"],
                    class_report["weighted avg"]["recall"],
                    class_report["weighted avg"]["f1-score"]
                ]
            })
            eval_results_csv = os.path.join(output_folder, "unet_evaluation_results.csv")
            eval_results_df.to_csv(eval_results_csv, index=False)

            #  Upload all to MLflow (MinIO)
            for f in [eval_results_csv, cm_path, class_report_path]:
                if os.path.exists(f):
                    mlflow.log_artifact(f, artifact_path="predictions")  # optional: "predictions" subfolder
                    print(f" Logged artifact to MLflow: {Path(f).name}")
                else:
                    print(f" File not found: {f}")

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

            print(f"Prediction stored in {output_folder}.")
            # ---------------- Merge large file tiles safely ----------------
            if not merge and total_pixels > threshold:
                merged_output_file = output_folder.parent / f"{Path(predict_model).stem}_merged_large.tif"
                print(f"🔹 Large file: merging tiles windowed into {merged_output_file} ...")
                merge_label_tiles_windowed(
                    tiles_folder=output_folder,
                    output_file=merged_output_file,
                    window_px=4096,
                    recursive=False,
                    show_progress=True
                )
                print(f"✅ Windowed merge finished: {merged_output_file}")
