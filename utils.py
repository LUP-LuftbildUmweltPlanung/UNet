import glob
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
import math
import re
import tifffile

from torch import nn
import torch

from fastai.losses import BaseLoss
from fastai.data.transforms import get_image_files
from fastai.vision.core import PILMask
from fastai.callback.schedule import minimum, steep, valley, slide
from fastcore.foundation import L
from fastai.vision.all import ItemTransform, TensorImage, TensorMask

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
    lrs = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide), show_plot=True)
    # plt.show()
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


class SegmentationAlbumentationsTransform(ItemTransform):
    """Applies Albumentations augmentations to images and optionally masks.

    Args:
        aug (callable): Albumentations augmentation function.
    Note:
        This transform expects input data in the form of tuples (image, mask).
        If only images are provided, it assumes no masks are present.
    """
    split_idx = 0  # Apply Augmentations for 0 = Train, 1 = Validation, None = Both

    def __init__(self, dtype, aug, n_transform_imgs=2, **kwargs):
        """
        Initializes the SegmentationAlbumentationsTransform.

        Args:
            aug (callable): Albumentations augmentation function.
            n_transform_imgs (int): Number of the augmented images minus from the batch size (default is 2).
        """
        super().__init__(**kwargs)
        self.aug = aug
        self.n_transform_imgs = n_transform_imgs
        self.dtype = dtype

    def encodes(self, x):
        """
        Applies albumentations augmentations to input images and masks.

        Args:
            x (tuple or Tensor): Input data containing images and masks.

        Returns:
            Tensor or tuple of Tensors: Transformed images and masks (if provided).
        """
        try:
            batch_img, batch_mask = x  # Expecting tuple (img, mask)
        except ValueError:
            batch_img = x  # Only one value is provided, assuming it's just the image
            batch_mask = None  # No mask provided
            # Check if n_transform_imgs is greater than or equal to the batch size
        if not (0 <= self.n_transform_imgs <= 1):
            raise ValueError(
                f"The n_transform_imgs parameter ({self.n_transform_imgs}) must be between 1 and 0.")

        Batch = len(batch_img)
        n_transform = math.ceil(Batch * self.n_transform_imgs)

        transformed_images = []
        transformed_masks = []

        if batch_mask is None:
            batch_img = batch_img[0]

            if self.dtype == 'int16':
                batch_img /= 255

            return [batch_img]

        # Process each image and mask in the last proportion of the batch
        else:
            for img, mask in zip(batch_img[:int(n_transform - len(batch_img))],
                                 batch_mask[:int(n_transform - len(batch_img))]):

                # Permute the image dimensions from (C, H, W) to (H, W, C) for albumentations
                img = img.permute(1, 2, 0)  # Now shape is [W, H, C]

                # Ensure tensor is on CPU before converting to numpy array
                img_np = img.cpu().numpy()
                mask_np = mask.cpu().numpy() if mask.is_cuda else mask.numpy()
                if self.dtype == 'int16':
                    img_np /= 65535
                elif self.dtype == 'int8':
                    img_np /= 255
                else:
                    ValueError("The data_type should be int8 or int16, your data not valid")

                # Apply augmentation
                aug = self.aug(image=img_np, mask=mask_np)

                # After augmentation, return to Uint8 Image for the Dataloader
                aug['image'] *= 255

                # After augmentation, transpose image back to [C, H, W]
                img_aug = np.transpose(aug['image'], (2, 0, 1))
                mask_aug = aug['mask']  # Assume mask needs no transposition if it's 2D

                # Convert augmented images and masks back to tensors and append to the transformed lists
                transformed_images.append(TensorImage(torch.from_numpy(img_aug).to(img.device)))
                transformed_masks.append(TensorMask(torch.from_numpy(mask_aug).to(mask.device)))

        # Leave the first proportion of the batch unchanged

        for img, mask in zip(batch_img[int(n_transform - len(batch_img)):],
                             batch_mask[int(n_transform- len(batch_img)):]):
            if self.dtype == 'int16':
                img /= 255
            
            # Append the unchanged images and masks to the transformed lists
            transformed_images.append(img)
            transformed_masks.append(mask)
        # Stack all processed items in the batch back into tensors
        return torch.stack(transformed_images), torch.stack(transformed_masks)



def save_params(params, model_Path, description):
    """
    Save parameters to a JSON file.

    Parameters:
    - params (dict): Dictionary of parameters to save.
    - description (str): Description to be used as the folder and file name.
    """

    def default_converter(o):
        if isinstance(o, (int, float, str, bool, type(None))):
            return o
        return str(o)

    # Path to save the JSON file
    json_path = Path(model_Path) / f"{description}.json"

    with open(json_path, 'w') as json_file:
        json.dump(params, json_file, indent=4, default=default_converter)
    print(f'Parameters saved to {json_path}')


def get_patch_size(base_dir):
    base_dir = Path(base_dir)
    base_dir = base_dir / "trai" / "img_tiles"

    # List all files in the directory
    files = [f for f in os.listdir(base_dir) if f.endswith('.tif')]

    if not files:
        raise ValueError("No .tif files found in the directory")

    # Open the first file to get the size, resolution, and data type
    file_path = base_dir / files[0]
    with tifffile.TiffFile(file_path) as tif:
        # Get image size
        width, height = tif.pages[0].shape[:2]

        # Attempt to get resolution from ModelPixelScaleTag if available
        resolution = None
        try:
            model_pixel_scale_tag = tif.pages[0].tags['ModelPixelScaleTag'].value
            resolution = (model_pixel_scale_tag[0], model_pixel_scale_tag[1])
        except KeyError:
            pass

        # If ModelPixelScaleTag is not found, use the default or any other available tags
        if resolution is None:
            for tag_name in ['XResolution', 'YResolution', 'Pixel Size']:
                try:
                    res_value = tif.pages[0].tags[tag_name].value
                    if isinstance(res_value, tuple) or isinstance(res_value, list):
                        resolution = (res_value[0], res_value[1])
                    else:
                        resolution = res_value
                    break
                except KeyError:
                    continue

        # Get data type
        data_type = tif.pages[0].dtype

        # Get number of bands
        number_of_bands = tif.pages[0].samplesperpixel

    return width, resolution, data_type, number_of_bands


def process_and_save_params(data_path, aug_pipe, model_path, description, transforms=False, **kwargs):
    """
    Process and save parameters to a JSON file.

    Parameters:
    - data_path (str): Path to the data.
    - aug_pipe (object): Augmentation pipeline.
    - model_path (str): Path to save the model parameters.
    - description (str): Description to be used as the folder and file name.
    - kwargs: Additional parameters to be processed.
    """

    # Extract patch size from the img path
    patch_size, resolution, data_type, number_of_bands = get_patch_size(data_path)

    # Extract the augmentation parameters
    aug_params_ = {transform.__class__.__name__: transform.p for transform in aug_pipe.transforms}

    # Capture parameters using locals()
    params = locals()
    params['patch_size'] = patch_size
    params['resolution'] = resolution
    params['data_type'] = data_type
    params['number_of_bands'] = number_of_bands
    params['aug_params_'] = aug_params_
    # Remove specific keys from params
    for key in ['data_path', 'aug_pipe', 'model_path', 'description']:
        params.pop(key, None)

    # Conditionally delete specific keys
    if not transforms:
        params.pop('aug_params_', None)
        # Remove specific keys from kwargs if necessary
        kwargs.pop('n_transform_imgs', None)

    # Ensure 'transforms' is set to True if it is supposed to be
    if transforms:
        params['transforms'] = True

    # Keys to delete
    keys_to_delete = ['BATCH_SIZE', 'EPOCHS', 'regression', 'LEARNING_RATE', 'LR_FINDER', 'ENCODER_FACTOR', 'loss_func',
                      'self_attention', 'monitor', 'ARCHITECTURE', 'CODES']

    # Delete the keys
    for key in keys_to_delete:
        if key in params:
            del params[key]

    def default_converter(o):
        if isinstance(o, (int, float, str, bool, type(None))):
            return o
        return str(o)

    # Convert the parameters dictionary to a JSON string
    json_string = json.dumps(params, indent=4, default=default_converter)

    formatted_json_string = re.sub(
        r'("CODES":\s*\[)([^\]]*)(\])|("VALID_SCENES":\s*\[)([^\]]*)(\])|("resolution":\s*\[)([^\]]*)(\])',
        lambda
            m: f'{m.group(1) or m.group(4) or m.group(7)}{" ".join((m.group(2) or m.group(5) or m.group(8)).split())}{m.group(3) or m.group(6) or m.group(9)}',
        json_string
    )
    # Path to save the JSON file
    json_path = Path(model_path) / f"{description}.json"

    # Save the formatted JSON string to a file
    with open(json_path, 'w') as json_file:
        json_file.write(formatted_json_string)

    print(f'Parameters saved to {json_path}')
