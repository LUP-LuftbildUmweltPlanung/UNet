from create_tiles_unet import split_raster
from predict import save_predictions
from train import train_func
from utils import backslash_to_forwardslash

import os 
import time
import torch
from pathlib import Path
import warnings
import albumentations as A
import random
import numpy as np
import imgaug


from fastai.vision.models.xresnet import xresnet34, xresnet101, xresnet50, xresnet34_deep, xresnet18
from fastai.vision.augment import Dihedral, Rotate, Brightness, Contrast, Saturation
from fastai.vision.core import imagenet_stats
from fastai.data.transforms import Normalize
from fastai.losses import MSELossFlat, CrossEntropyLossFlat, L1LossFlat, FocalLossFlat, DiceLoss

import fastai.learner as fastai_learner


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# PARAMETERS
Create_tiles = True
Train = True
Predict = True

######################################################
#################### CREATE TILES ####################
######################################################

# if using without mask, set mask_path = None
image_path = r"/home/embedding/Data_Center/qnap3b/2024_BfN_Naturerbe/Prozessierung/UNet_Modell/Model_fixxed/data/RH_mosaic1.tif"
mask_path = r"/home/embedding/Data_Center/qnap3b/2024_BfN_Naturerbe/Prozessierung/UNet_Modell/Model_fixxed/data/rüthnicker_heide_LUP_2009_mask.tif"
base_dir = r"/home/embedding/Data_Center/qnap3b/2024_BfN_Naturerbe/Prozessierung/UNet_Modell/Model_fixxed/create_tiles_sample"

#for prediction patch_overlap = 0.2 to prevent edge artifacts and split = [1] to predict full image
patch_size = 400
patch_overlap = 0
split = [0.7, 0.2,0.1]



############################################################
#################### TRAINING ##############################
############################################################
# If using created tiles, set data_path to base_dir.
data_path = base_dir
model_path = r"/home/embedding/Data_Center/qnap3b/2024_BfN_Naturerbe/Prozessierung/UNet_Modell/Model_fixxed/models" # The path where the model directories will be created.
description = "test_seed7_noaug" # A description of the model folder, typically formatted as "response_specific_use_case". # Example: "canopycover_augmentationtest".
info = "CIR 50cm images" # Additional information about the model, such as necessary input features (e.g., RGBI) and other relevant details.
existing_model = None #or existing model path for transfer_learning
BATCH_SIZE = 4  # 3 for xresnet50, 12 for xresnet34 with Tesla P100 (16GB)
EPOCHS = 1
LEARNING_RATE = 0.01
enable_regression = False
visualize_data_example = True
export_model_summary = True
save_confusion_matrix = False # A boolean to enable or disable saving the confusion matrix table.
# only relevant for classification
CODES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
# CODES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']
CLASS_WEIGHTS= [120.40, 15.25, 1.36, 69.11, 17.58, 35.90, 90.34, 26.58, 108.04, 1106.80, 190.26,
                93.27, 484.23, 1247.66, 1547.07, 364.96, 5077.46, 188.60, 0, 3118.93, 4633.75,
                2770.34, 3020.36, 303.24, 8116.60, 3810.15, 13784.80, 0, 1673.77, 6061.20, 817.59,
                31456.04, 2741.77, 5259.98] #[0.0001, 1, 1, 10, 10] #"weighted"  # list (e.g. [3, 2, 5]) or string ("even" or "weighted")
#CLASS_WEIGHTS = "even" #[0.0001, 1, 1, 10, 10] #"weighted"  # list (e.g. [3, 2, 5]) or string ("even" or "weighted")
#CLASS_WEIGHTS = "weighted" #"weighted" #[0.01, 0.3, 0.69]
#CLASS_WEIGHTS =[19, 4, 6, 35, 3, 38, 85, 5, 52, 123, 54]

########################################################
#################### PREDICTION ########################
########################################################

# Prediction parameters
predict_path = r"/home/embedding/Data_Center/qnap3b/2024_BfN_Naturerbe/Prozessierung/UNet_Modell/Model_fixxed/create_tiles_sample/test/img_tiles" # define the images path
predict_model = r"/home/embedding/Data_Center/qnap3b/2024_BfN_Naturerbe/Prozessierung/UNet_Modell/Model_fixxed/models/test_seed7_noaug/test_seed7_noaug.pkl" # the path where the model saved and the name of the model "name.torch"
AOI = "RH" # Area of Interest (AOI). This parameter is used to append the output TIFF file to define the city of the prediction data.
year = "2009" # The year of the prediction data. To append the output TIFF file to define the year.
merge = False # A boolean to decide whether to merge the output prediction tiles into a single file or keep them as separate tiles.
regression = False
validation_vision = True # Confusion matrix and classification report figures, Keep merge and regression False to work!
# CONFIG END


############################################################
#################### EXTRA PARAMTERS #######################
############################################################

enable_extra_parameters = True  # only for experienced users

self_attention = True
ENCODER_FACTOR = 10  # minimal lr_rate factor
LR_FINDER = None  # None, "minimum", "steep", "valley", "slide"
VALID_SCENES = ['vali']
loss_func = CrossEntropyLossFlat(axis=1)  # FocalLossFlat(gamma=2, axis=1)
# Regression: MSELossFlat(axis=1), L1LossFlat(axis=-1)
# Classification: CrossEntropyLossFlat(axis=1), FocalLossFlat(gamma=0.5, axis=1)
monitor = 'valid_loss'  # 'dice_multi'  'r2_score'
# Regression: 'train_loss', 'valid_loss', 'r2_score' !if existing model is used, monitor of original model is applied!
# Classification: 'dice_multi', 'train_loss'    !if existing model is used, monitor of original model is applied!
all_classes = False  # If all class predictions should be stored
specific_class = None  # None or integer of class -> Only this class will be stored
large_file = False  # If predicted probabilities should be stretched to int8 to increase storage capacity
max_empty = 1  # Maximum no data area in created image crops
class_zero = False  # Enable for seperating 0 prediction class from nodata

ARCHITECTURE = xresnet34  # xresnet34

# Create an instance of the transforms
transforms = False
n_transform_imgs = 1 # Percentage of augmented images [0-1]. Decimals always be rounded up.
aug_pipe = A.Compose([
    A.HorizontalFlip(p=0.5),  # Applies a horizontal flip to the image with a probability of 0.5.
    A.VerticalFlip(p=0.5),  # Applies a vertical flip to the image with a probability of 0.5.
#    A.RandomBrightnessContrast(  # Randomly changes brightness and contrast of the image with a probability of 0.5.
#        brightness_limit=(-0.1, 0.1),
#        contrast_limit=(-0.1, 0.1),
#        p=0.5
 #   ),
#    A.CoarseDropout(p=0.5),  # Randomly masks out rectangular regions in the image with a probability of 0.5.

])  # For more Augmentation options: https://github.com/albumentations-team/albumentations/tree/main#i-am-new-to-image-augmentation


# EXTRA END

#change paths to work on Windows and Linux
image_path = backslash_to_forwardslash(image_path)
mask_path = backslash_to_forwardslash(mask_path)
base_dir = backslash_to_forwardslash(base_dir)
model_path = backslash_to_forwardslash(model_path)
predict_path = backslash_to_forwardslash(predict_path)
predict_model = backslash_to_forwardslash(predict_model)


def main():
    """Main function."""

    global large_file, specific_class, all_classes, transforms, VALID_SCENES, self_attention, monitor, loss_func, \
        LR_FINDER, ENCODER_FACTOR, ARCHITECTURE, enable_regression, max_empty

    start_time = time.time()
    #temp = pathlib.PosixPath
    #pathlib.PosixPath = pathlib.WindowsPath




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
        transforms = transforms
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
            max_empty=max_empty,
            class_zero=class_zero
        )

    if Train:
        # run train function
        train_func(data_path, existing_model, model_path, description, BATCH_SIZE, visualize_data_example,
                   enable_regression, CLASS_WEIGHTS,
                   ARCHITECTURE, EPOCHS, LEARNING_RATE, ENCODER_FACTOR, LR_FINDER, loss_func, monitor, self_attention,
                   VALID_SCENES,
                   CODES, transforms, export_model_summary, aug_pipe, n_transform_imgs, save_confusion_matrix, info,
                   class_zero)

    if Predict:
        save_predictions(predict_model, predict_path, regression, merge, all_classes, specific_class, large_file, AOI,
                         year, validation_vision, class_zero=class_zero)

    end_time = time.time()
    print(f"The operation took {(end_time - start_time):.2f} seconds or {((end_time - start_time) / 60):.2f} minutes")


if __name__ == '__main__':
    main()





