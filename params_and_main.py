from create_tiles_unet import split_raster
from predict import save_predictions
from train import train_func
import time

from fastai.vision.models.xresnet import xresnet34, xresnet101, xresnet50, xresnet34_deep, xresnet18
from fastai.vision.augment import Dihedral, Rotate, Brightness, Contrast, Saturation
from fastai.vision.core import imagenet_stats
from fastai.data.transforms import Normalize
from fastai.losses import MSELossFlat, CrossEntropyLossFlat, L1LossFlat, FocalLossFlat, DiceLoss

import torch

import pathlib
import os
import warnings
import albumentations as A


# PARAMETERS
Create_tiles = False
Train = False
Predict = False

######################################################
#################### CREATE TILES ####################
######################################################

# if using without mask, set mask_path = None
image_path = "PATH"
mask_path = "PATH"
base_dir = "PATH"

#for prediction patch_overlap = 0.2 to prevent edge artifacts and split = [1] to predict full image
patch_size = 1024
patch_overlap = 0.2
split = [1]



############################################################
#################### TRAINING ##############################
############################################################
# if using on created tiles, set data_path = base_dir
data_path = r"PATH"
model_path = r"PATH"
existing_model = None #or existing model path for transfer_learning
BATCH_SIZE = 4  # 3 for xresnet50, 12 for xresnet34 with Tesla P100 (16GB)
EPOCHS = 3
LEARNING_RATE = 0.005
enable_regression = False
visualize_data_example = True
export_model_summary = True
# only relevant for classification
CODES = ['background', 'unversiegelt', 'teilversiegelt', 'versiegelt', 'versiegelt_haus']
CLASS_WEIGHTS = "even" #[0.0001, 1, 1, 10, 10] #"weighted"  # list (e.g. [3, 2, 5]) or string ("even" or "weighted")
#CLASS_WEIGHTS = "weighted" #"weighted" #[0.01, 0.3, 0.69]
#CLASS_WEIGHTS =[19, 4, 6, 35, 3, 38, 85, 5, 52, 123, 54]

########################################################
#################### PREDICTION ########################
########################################################
predict_path = "PATH"
predict_model = "PATH"
merge = False
regression = False
# CONFIG END


############################################################
#################### EXTRA PARAMTERS #######################
############################################################

enable_extra_parameters = True # only for experienced users

self_attention = False
ENCODER_FACTOR = 10  # minimal lr_rate factor
LR_FINDER = None  # None, "minimum", "steep", "valley", "slide"
VALID_SCENES = ['vali']
loss_func = CrossEntropyLossFlat(axis=1) #FocalLossFlat(gamma=2, axis=1)
# Regression: MSELossFlat(axis=1), L1LossFlat(axis=-1)
# Classification: CrossEntropyLossFlat(axis=1), FocalLossFlat(gamma=0.5, axis=1)
monitor = 'dice_multi' #'dice_multi'  'r2_score'
# Regression: 'train_loss', 'valid_loss', 'r2_score' !if existing model is used, monitor of original model is applied!
# Classification: 'dice_multi', 'train_loss'    !if existing model is used, monitor of original model is applied!
all_classes = False # If all class predictions should be stored
specific_class = None  # None or integer of class -> Only this class will be stored
large_file = False # If predicted probabilities should be stretched to int8 to increase storage capacity
max_empty = 0.2  # Maximum no data area in created image crops
ARCHITECTURE = xresnet34 #xresnet34
aug_pipe = A.Compose([
            A.HorizontalFlip(p=0.5), # Applies a horizontal flip to the image with a probability of 0.5.
            A.VerticalFlip(p=0.5), # Applies a vertical flip to the image with a probability of 0.5.
            A.ShiftScaleRotate(p=0.5), # Randomly applies affine transforms: translation, scaling and rotation in one call with a probability of 0.5.
            A.RandomBrightnessContrast( # Randomly changes brightness and contrast of the image with a probability of 0.5.
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
            A.CoarseDropout(p=0.5), # Randomly masks out rectangular regions in the image with a probability of 0.5.

]) # For more Augmentation options: https://github.com/albumentations-team/albumentations/tree/main#i-am-new-to-image-augmentation

n_transform_imgs = 2 # Number of augmented images (default is 2).


# Create an instance of the transforms 
transforms = False
# EXTRA END


def main():
    """Main function."""

    global large_file, specific_class, all_classes, transforms, VALID_SCENES, self_attention, monitor, loss_func, LR_FINDER, ENCODER_FACTOR, ARCHITECTURE, enable_regression, max_empty

    start_time = time.time()
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    if enable_extra_parameters:
        warnings.warn("Extra parameters are enabled. Code may behave in unexpected ways. "
                      "Please disable unless experienced with the code.")
    else:
        ENCODER_FACTOR = 10
        LR_FINDER = None
        VALID_SCENES = ['vali']
        loss_func = None
        monitor = None
        all_classes = False
        specific_class = None
        enable_regression = False
        large_file = False
        max_empty = 0.9
        ARCHITECTURE = xresnet34
        transforms = None
        self_attention = False

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(torch.cuda.get_device_properties(0))
    else:
        print("No CUDA device available.")

    if Create_tiles:
        split_raster(
            path_to_raster=image_path,
            path_to_mask=mask_path,
            patch_size=patch_size,
            patch_overlap=patch_overlap,
            base_dir=base_dir,
            split=split,
            max_empty=max_empty
        )

    if Train:
        #run train function
        train_func(data_path, existing_model, model_path, BATCH_SIZE, visualize_data_example, enable_regression, CLASS_WEIGHTS,
                ARCHITECTURE, EPOCHS, LEARNING_RATE, ENCODER_FACTOR, LR_FINDER, loss_func, monitor, self_attention, VALID_SCENES, 
                CODES, transforms, export_model_summary,  aug_pipe, n_transform_imgs)

    if Predict:
        save_predictions(predict_model, predict_path, regression, merge, all_classes, specific_class, large_file)

    end_time = time.time()
    print(f"The operation took {(end_time - start_time):.2f} seconds or {((end_time - start_time) / 60):.2f} minutes")

if __name__ == '__main__':
    main()
