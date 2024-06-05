import glob
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


import torch
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


def store_tif(output_folder, output_array, dtype, geo_transform, geo_proj, nodata_value):
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

    # loop through the image bands to set nodata
    if nodata_value is not None:
        for i in range(1, out_ds.RasterCount + 1):
            # set the nodata value of the band
            out_ds.GetRasterBand(i).SetNoDataValue(nodata_value)

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
    """Creates class weights inversely proportional to the amount of class-counts in the dataset."""
    msk_files = path / r"trai/mask_tiles"
    dls = tiles.dataloaders(path, bs=np.min([len(list(msk_files.glob('*.tif'))), 1200]), num_workers=0)
    count_tensor = dls.one_batch()[1].unique(return_counts=True)[1]
    total_samples = sum(count_tensor)  # Total number of samples in the dataset
    class_w = []
    for count in count_tensor:
        class_weight = total_samples.item() / count.item()
        class_w.append(class_weight)

    return class_w


def visualize_data(inputs, model_path):
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


def check_and_fill(args, target_len):
    """
    Ensure that all argument lists match the target length by repeating their single element if necessary.

    This function iterates through a list of argument lists (args) and checks each one against the target length (target_len).
    If an argument list has exactly one element, it is repeated to match the target length. If an argument list does not match
    the target length and has more than one element, a ValueError is raised to indicate a configuration error.

    Parameters:
    - args: A list of lists. Each inner list corresponds to an argument that might need adjustment.
    - target_len: The target length that all argument lists should match.

    Returns:
    - A list of lists, where each inner list has been adjusted to match the target length or is left as is if it already matches.

    Raises:
    - ValueError: If an argument list has more than one element but does not match the target length.
    """
    for i, arg in enumerate(args):
        if len(arg) == 1:
            args[i] = arg * target_len
        elif len(arg) != target_len:
            raise ValueError(f"Argument list at index {i} has {len(arg)} elements; expected {target_len}.")
    return args



## Augmentation
class SegmentationAlbumentationsTransform(ItemTransform):
    """Applies Albumentations augmentations to images and optionally masks.

    Args:
        aug (callable): Albumentations augmentation function.
        prop (float): Proportion of the batch to apply augmentation to (default is 0.5).
    
    Note:
        This transform expects input data in the form of tuples (image, mask).
        If only images are provided, it assumes no masks are present.
    """
    def __init__(self, aug, Num=2, **kwargs):
        """
        Initializes the SegmentationAlbumentationsTransform.

        Args:
            aug (callable): Albumentations augmentation function.
            Num (float): Number of the augmented images minus from the batch size (default is 2).
        """
        super().__init__(**kwargs)
        self.aug = aug
        self.Num = Num

    def encodes(self, x):
        """
        Applies Albumentations augmentations to input images and masks.

        Args:
            x (tuple or Tensor): Input data containing images and masks.

        Returns:
            Tensor or tuple of Tensors: Transformed images and masks (if provided).
        """
        try:
            batch_img, batch_mask = x  # Expecting tuple (img, mask)
        except ValueError:
            batch_img = x  # Only one value provided, assuming it's just the image
            batch_mask = None  # No mask provided
            # Check if Num is greater than or equal to the batch size
        if len(batch_img) <= self.Num:
            raise ValueError(f"The Num parameter ({self.Num}) must be less than the batch size ({len(batch_img)}).")
  
        transformed_images = []
        transformed_masks = []
        
        if batch_mask is None:
            for img in batch_img:  # Ensure this iterates correctly over a batch
                # Ensure the image has the correct dimensions [B, C, H, W] -> [B, H, W, C] for Albumentations
                if img.dim() == 4:
                    img_np = img.permute(0, 2, 3, 1).cpu().numpy()  # Change to [B, H, W, C]
            
                    # Apply augmentation to each image individually in the batch
                    try:
                        transformed = self.aug(image=img_np[0])  # Apply to the first (or only) image in the batch
                        img_aug = np.transpose(transformed['image'], (2, 0, 1))
                        transformed_images.append(TensorImage(torch.from_numpy(img_aug).unsqueeze(0)))  # Re-add batch dimension
                    except Exception as e:
                        print("Error during augmentation:", e)
            return torch.stack(transformed_images)  # Stack to get [B, C, H, W]
        
        # Process each image and mask in the first proportion of the batch
        for img, mask in zip(batch_img[:int(self.Num - len(batch_img))], batch_mask[:int(self.Num - len(batch_img))]):
            # Normalize the image
            img = img / img.max()
        
            # Permute the image dimensions from (C, H, W) to (H, W, C) for albumentations
            img = img.permute(1, 2, 0)  # Now shape is [W, H, C]
        
            print('Applying augmentation')
            # Ensure tensor is on CPU before converting to numpy array
            img_np = img.cpu().numpy()
            mask_np = mask.cpu().numpy() if mask.is_cuda else mask.numpy()
        
            # Apply augmentation
            aug = self.aug(image=img_np, mask=mask_np)
        
            # After augmentation, transpose image back to [C, H, W]
            img_aug = np.transpose(aug['image'], (2, 0, 1))
            mask_aug = aug['mask']  # Assume mask needs no transposition if it's 2D
        
            # Convert augmented images and masks back to tensors and append to the transformed lists
            transformed_images.append(TensorImage(torch.from_numpy(img_aug).to(img.device)))
            transformed_masks.append(TensorMask(torch.from_numpy(mask_aug).to(mask.device)))
            
        # Leave the second proportion of the batch unchanged
        for img, mask in zip(batch_img[int(self.Num - len(batch_img)):], batch_mask[int(self.Num - len(batch_img)):]):
            # Append the unchanged images and masks to the transformed lists
            transformed_images.append(img)
            transformed_masks.append(mask)
            print('Not applying augmentation')
        
        # Stack all processed items in the batch back into tensors
        return torch.stack(transformed_images), torch.stack(transformed_masks)
