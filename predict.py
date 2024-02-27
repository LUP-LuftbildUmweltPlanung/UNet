import glob
import os
import warnings
import numpy as np
import torch
import time
from tqdm import tqdm
from pathlib import Path
from osgeo import gdal
from fastai.learner import load_learner

from utils import store_tif


def save_predictions(predict_model, predict_path, regression, merge=False, all_classes=False, specific_class=None, large_file=False):
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
    learn = load_learner(Path(predict_model))
    path = Path(predict_path)

    output_folder = path / ('predicted_tiles_' + Path(predict_model).stem)

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
    #for i in range(len(tiles)):
    for i in tqdm(range(len(tiles)), desc='Processing tiles'):
        #print(f'Current progress: {i}/{len(tiles)}')
        tile_preds = learn.predict(Path(tiles[i]), with_input=False)
        # if single class creates issues try: class_lst = tile_preds[1]
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
                class_lst *= ((128 / 4) - 1)
                class_lst = np.around(class_lst).astype(np.int8)
                dtype = gdal.GDT_Byte
                store_tif(str(output_folder) + "\\" + os.path.basename(tiles[i]), class_lst, dtype, geotrans,geoproj, None)
            else:
                store_tif(str(output_folder) + "\\" + os.path.basename(tiles[i]), class_lst.numpy(), dtype, geotrans, geoproj, None)

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

            #set raster to -9999 where no predictions were placed
            nodata = -9999
            merged_raster[merge_counter == 0] = nodata #maybe change to merged_raster[merged_raster == 0] = nodata
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
                #merged_raster = np.where(merged_raster.max(axis=0) < 0.1, 14, merged_raster.argmax(axis=0)) #check if this is still relevant?
                merged_raster = merged_raster.argmax(axis=0)
            else:
                # for probabilities of specific class [1] -> klasse 1
                merged_raster = merged_raster[specific_class]

            # define nodata for classification (either background class or where no Tiles were placed)
            nodata = 0

        if "float" in str(merged_raster.dtype):
            dtype = gdal.GDT_Float32
        else:
            dtype = gdal.GDT_Byte

        store_tif(str(output_folder) + "\\class.tif", merged_raster, dtype,
                  [upleft_x_full, geotrans_for_merge[0, 2], 0.0, upleft_y_full, 0.0, geotrans_for_merge[0, 5]],
                  geoproj_for_merge, nodata)
        print(f"Prediction stored in {output_folder}.")
