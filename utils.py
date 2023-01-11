import glob
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from pathlib import Path

from fastai.losses import BaseLoss
from fastai.data.transforms import get_image_files
from fastai.vision.core import PILMask
from fastai.callback.schedule import minimum, steep, valley, slide
from fastcore.foundation import L

from osgeo import gdal, gdal_array


def get_image_tiles(path: Path, ) -> L:
    """Returns a list of the image tile filenames in path"""
    files = L()
    for folder in path.ls():
        folder = folder
        files.extend(get_image_files(path=folder, folders='img_tiles'))
    return files


def get_y_fn(fn: Path) -> str:
    """Returns filename of the associated mask tile for a given image tile"""
    return str(fn).replace('img_tiles', 'mask_tiles')


def load_gdal(path):
    """Loads an image file from path using gdal."""
    img_ds = gdal.Open(path, gdal.GA_ReadOnly)
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    return img


def get_y(fn: Path):
    """Returns a PILMask object of 0s and 1s for a given tile"""
    fn = get_y_fn(fn)
    msk2 = load_gdal(fn)[:, :, 0]
    return PILMask.create(msk2)


def delete_folder(folder_path):
    """Deletes an empty folder by given path"""
    # checking whether folder exists or not
    if os.path.exists(folder_path):

        # checking whether the folder is empty or not
        if len(os.listdir(folder_path)) == 0:
            # removing the file using the os.remove() method
            os.rmdir(folder_path)
        else:
            # messaging saying folder not empty
            print("Folder is not empty")
    else:
        # file not found message
        print("Folder not found in the directory")


def annot_min(y, ax=None):
    """Adds an arrow for the lowest loss within a plot to a plot"""
    xmin = np.argmin(y)
    ymin = np.min(y)
    text = f"Lowest Loss={ymin:.2f}, Ep. {xmin}"
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=120")
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="left", va="top")
    ax.annotate(text, xy=(xmin, ymin), xytext=(0.06, 0.96), **kw)


def store_tif(output_folder, output_array, dtype, geo_transform, geo_proj):
    """Stores a tif file in a specified folder."""
    driver = gdal.GetDriverByName('GTiff')

    if len(output_array.shape) == 3:
        out_ds = driver.Create(str(output_folder), output_array.shape[2], output_array.shape[1], output_array.shape[0],
                               dtype)
    else:
        out_ds = driver.Create(str(output_folder), output_array.shape[1], output_array.shape[0], 1, dtype)
    out_ds.SetGeoTransform(geo_transform)

    out_ds.SetProjection(geo_proj)
    if len(output_array.shape) == 3:
        for b in range(output_array.shape[0]):
            out_ds.GetRasterBand(b + 1).WriteArray(output_array[b])
    else:
        out_ds.GetRasterBand(1).WriteArray(output_array)
    out_ds.FlushCache()
    out_ds = None


def get_datatype(path):
    """Gets the largest datatype of all images within a directory"""
    file = glob.glob(str(path / r'trai\img_tiles\*tif'))[0]
    img_ds = gdal.Open(file, gdal.GA_ReadOnly)
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))

    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()

    no_data = img_ds.GetRasterBand(1).GetNoDataValue()
    max_val = np.max(img[img[:, :, 0] != no_data])
    if max_val < 257:
        print('Data in int8')
        return 'int8'
    else:
        print('Data in int16')
        return 'int16'


def is_outlier(points, thresh=3.5):
    """Returns a boolean array with True if points are outliers and False otherwise."""
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def get_class_weights(path, tiles):
    """Creates class weights anti-proportional to the amount of class-counts in the dataset."""
    msk_files = path / r"trai/mask_tiles"
    dls = tiles.dataloaders(path, bs=np.min([len(list(msk_files.glob('*.tif'))), 700]), num_workers=0)
    count_tensor = dls.one_batch()[1].unique(return_counts=True)[1]

    class_w = []
    for classes in range(len(count_tensor)):
        count_classes = count_tensor[classes].item()
        class_w.append(1 / count_classes)
    class_w /= np.sum(class_w)
    class_w_nobackg = (0.99/np.sum(class_w[1:])*class_w[1:]) #get weights without background class
    class_w = np.concatenate((np.array([1-sum(class_w_nobackg)]),class_w_nobackg )) #create weights array with fixed background wheight

    return class_w


def visualize_data(inputs):
    """Plots a detailed histogram of the data bands"""
    if len(inputs.shape) != 3:
        inputs_bands = inputs.shape[1]
    else:
        inputs_bands = 1
    fig, axes = plt.subplots(nrows=2, ncols=inputs_bands, sharey='row', figsize=(10, 10))
    if inputs_bands > 1:
        for band in range(inputs_bands):
            band_data = inputs[:, band].flatten()
            axes[0, band].hist(band_data[band_data > 0], bins=255)
            axes[0, band].set_title(f'Band {band + 1}')
            axes[1, band].hist(band_data[band_data > 0], bins=255, range=(0, 1))
        plt.suptitle('Image batch example histogram')
        plt.savefig(Path(str(model_path).rsplit('.', 1)[0] + "_image_plot.png"))


    else:
        inputs = inputs.flatten()
        axes[0].hist(inputs, bins=255)
        axes[1].hist(inputs, bins=255, range=(0, 1))
        plt.suptitle('Mask batch example histogram')
        plt.savefig(Path(str(model_path).rsplit('.', 1)[0] + "_mask_plot.png"))



def Smoothl1(*args, axis=1, floatify=True, **kwargs):
    """Same as 'nn.L1Loss', but flattens input and target."""
    return BaseLoss(nn.SmoothL1Loss, *args, axis=axis, floatify=floatify, is_2d=False, beta=0.5, **kwargs)


def find_lr(learn, finder):
    """Finds the suggested maximum learning rate using a fastai learning rate finder"""
    lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide),show_plot=True)
    #plt.show()
    if finder == 'valley':
        lr_max = lrs.valley
    elif finder == 'slide':
        lr_max = lrs.slide
    elif finder == 'steep':
        lr_max = lrs.steep
    elif finder == 'minimum':
        lr_max = lrs.minimum
    else:
        lr_max = lrs.valley
        warnings.warn("Learning rate finder parameter not recognised (minimum, steep, valley, slide, None)."
                      " Using valley.")

    return lr_max



