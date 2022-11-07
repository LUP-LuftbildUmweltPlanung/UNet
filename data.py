from functools import partial
from pathlib import Path

import numpy as np
import rasterio
import torch
from fastai.data.block import TransformBlock, RegressionBlock, DataBlock
from fastai.data.transforms import IntToFloatTensor, FuncSplitter
from fastai.torch_core import TensorImage, TensorMask
from fastai.vision.data import MaskBlock
from fastcore.basics import store_attr
from fastcore.foundation import L
from fastcore.transform import DisplayedTransform

from utils import get_y, get_image_tiles


def open_npy(fn, chnls=None, cls=torch.Tensor):
    """Opens an image file using rasterio. Returns a torch tensor"""
    numpy_image = rasterio.open(fn).read()
    #try:
    #im = torch.from_numpy(numpy_image).type(torch.float32)
    #except:
    im = torch.from_numpy(numpy_image.astype(np.int32)).type(torch.float32)

    if chnls is not None:
        im = im[chnls]
    return cls(im)


class MSTensorImage(TensorImage):
    """Class handling the image files. The create function was added to allow loading of images.\n
    Taken from: https://github.com/cordmaur/Fastai2-Medium/blob/master/01_Create_Datablock.ipynb"""
    def __init__(self,x, chnls_first=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chnls_first = chnls_first

    @classmethod
    def create(cls, data: (Path, str, np.ndarray), chnls=None, chnls_first=True):
        if isinstance(data, Path) or isinstance(data, str):
            im = open_npy(fn=data, chnls=chnls, cls=torch.Tensor)

        elif isinstance(data, np.ndarray):
            im = torch.from_numpy(data)
        else:
            im = data

        return cls(im, chnls_first=chnls_first)

    def __repr__(self):

        return f'MSTensorImage: {self.shape}'


def get_lbl_fn(img_fn: Path):
    """Gets the mas path for a image."""
    lbl_path = img_fn.parent.parent / 'mask_tiles'
    lbl_name = img_fn.name
    return lbl_path / lbl_name


class Int16ToFloatTensor(DisplayedTransform):
    """Transform image to float tensor, optionally dividing by 255 (e.g. for images)."""
    def __init__(self, div=65535., div_mask=1): store_attr()

    def encodes(self, o: TensorImage): return o.float().div_(self.div)

    def encodes(self, o: TensorMask): return o.long() // self.div_mask

    def decodes(self, o: TensorImage): return ((o.clamp(0., 1.) * self.div).long()) if self.div else o


def create_data_block(valid_scenes, codes, dtype, regression=False, transforms=None):
    """
    Creates a general fastai image datablock. This datablock will handle creation of dataloaders.

    Parameters:
    -----------
        valid_scenes : Folder name of the validation files (usually 'vali')
        codes : Class labels
        dtype : Datatype of the provided data
        regression : If output data should be handled as continuous values (default=False)
        enable_transforms : If transformations on the data should be enabled (default=False)

    Returns:
    ---------
        db : Multispectral capable dataloader for int16, int8, and discrete and continuous values.

    References:
    ----------
        Only necessary if code is based on anything
    """
    if dtype == 'int16':
        ImgBlock = TransformBlock(type_tfms=partial(MSTensorImage.create, chnls_first=True),
                                  batch_tfms=Int16ToFloatTensor)
    else:
        ImgBlock = TransformBlock(type_tfms=partial(MSTensorImage.create, chnls_first=True),
                                  batch_tfms=IntToFloatTensor)

    if regression:
        blocks = (ImgBlock, RegressionBlock())
    else:
        blocks = (ImgBlock, MaskBlock(codes))  # Independent variable is Image, dependent variable is Mask
    def valid_split(item, valid_scenes=valid_scenes):
        """XXXXXXXXX"""
        scene = item.parent.parent.name
        return scene in valid_scenes

    def get_undersampled_tiles(path: Path) -> L:
        """Returns a list of image tile filenames in `path`.
        For tiles in the training set, empty tiles are ignored.
        All tiles in the validation set are included."""

        files = get_image_tiles(path)
        train_idxs, valid_idxs = FuncSplitter(valid_split)(files)
        train_files = files[train_idxs]
        valid_files = files[valid_idxs]

        return train_files + valid_files

    if transforms is not None:
        db = DataBlock(
            blocks=blocks,
            get_items=get_undersampled_tiles,  # Collect undersampled tiles
            get_y=get_y,  # Get dependent variable: mask
            splitter=FuncSplitter(valid_split),  # Split into training and validation set
            batch_tfms=transforms  # Transforms on GPU: augmentation, normalization
        )
    else:
        db = DataBlock(
            blocks=blocks,
            get_items=get_undersampled_tiles,  # Collect undersampled tiles
            get_y=get_y,  # Get dependent variable: mask
            splitter=FuncSplitter(valid_split),  # Split into training and validation set
        )

    return db
