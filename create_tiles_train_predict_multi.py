from create_tiles_unet import split_raster
from predict import save_predictions
from train import train_func
from utils import check_and_fill
import time
from pathlib import Path

from fastai.vision.models.xresnet import xresnet34, xresnet101, xresnet50, xresnet34_deep, xresnet18
from fastai.vision.augment import Dihedral, Rotate, Brightness, Contrast, Saturation
from fastai.vision.core import imagenet_stats
from fastai.data.transforms import Normalize
from fastai.losses import MSELossFlat, CrossEntropyLossFlat, L1LossFlat, FocalLossFlat, DiceLoss


############# Switches
Create_tiles = False
Train = False
Predict = False

############## PARAMETERS - if input Parameters are [Lists], they may contain multiple elements for subsequent processing,
############# in case of single element parameters, the same is used for each iteration

######################################################
#################### CREATE TILES ####################
######################################################

# if using without mask, set mask_path = None
image_path = [
            Path('PATH'),
              ]

mask_path = [
            Path('PATH'),
             ]

base_dir = [
           Path('PATH'),
            ]


patch_size = 400
patch_overlap = 0.2 #0
split = [1] #[0.7,0.3]
#for prediction patch_overlap = 0.2 to prevent edge artifacts and split = [1] to predict full image
max_empty = 0.9  # Maximum no data area in created image crops

######################################################
#################### TRAIN ##########################
######################################################
# if using on created tiles, set data_path = base_dir

model_path = [
                Path("PATH"),
              ]

data_path = [
                Path("PATH"),
             ]

existing_model = [
                None,
                  ]#Path("PATH") or None
BATCH_SIZE = [4]# 3 for xresnet50, 12 for xresnet34 with Tesla P100 (16GB)
EPOCHS = [2]
LEARNING_RATE = [0.00001]
enable_regression = [False]
visualize_data_example = [True]

export_model_summary = [True]
# only relevant for classification
CODES = [['background', 'unversiegelt','teilversiegelt','versiegelt','versiegelt_haus']]
         #['background', 'unversiegelt', 'teilversiegelt', 'versiegelt', 'versiegelt_haus'],
         #['background', 'unversiegelt', 'teilversiegelt', 'versiegelt', 'versiegelt_haus'],
         #['background', 'unversiegelt', 'teilversiegelt', 'versiegelt', 'versiegelt_haus'],
         #['background', 'unversiegelt', 'versiegelt', 'versiegelt_haus']]
CLASS_WEIGHTS = ["even"]#, "even",[0.0001, 1, 1, 20, 20], "even", "even"]#[0.0001, 1, 1, 10, 10] #"weighted"  # list (e.g. [3, 2, 5]) or string ("even" or "weighted")
#CLASS_WEIGHTS = "weighted" #"weighted" #[0.01, 0.3, 0.69]

##EXTRA PARAMS
self_attention = [False]
ENCODER_FACTOR = [10]  # minimal lr_rate factor
LR_FINDER = [None]  # None, "minimum", "steep", "valley", "slide"
VALID_SCENES = [['vali']]
loss_func = [CrossEntropyLossFlat(axis=1)]#FocalLossFlat(gamma=2, axis=1)
# Regression: MSELossFlat(axis=1), L1LossFlat(axis=-1)
# Classification: CrossEntropyLossFlat(axis=1), FocalLossFlat(gamma=0.5, axis=1)
monitor = ['dice_multi'] #'dice_multi'  'r2_score'
# Regression: 'train_loss', 'valid_loss', 'r2_score' !if existing model is used, monitor of original model is applied!
# Classification: 'dice_multi', 'train_loss'    !if existing model is used, monitor of original model is applied!
ARCHITECTURE = [xresnet34] #xresnet34
transforms = [None]


########################################################
#################### PREDICTION ########################
########################################################
predict_path = [
    Path("PATH"),
            ]

predict_model = [
    Path("PATH")
            ]
merge = [True]
regression = False

all_classes = [False] # If all class predictions should be stored
specific_class = None  # None or integer of class -> Only this class will be stored
large_file = True # If predicted probabilities should be stretched to int8 to increase storage capacity

# CONFIG END


def main():
    global image_path, mask_path, base_dir
    global data_path, existing_model, BATCH_SIZE, visualize_data_example, enable_regression, CLASS_WEIGHTS
    global ARCHITECTURE, EPOCHS, LEARNING_RATE, ENCODER_FACTOR, LR_FINDER, loss_func, monitor
    global self_attention, VALID_SCENES, CODES, transforms, export_model_summary
    global predict_path, predict_model, merge, all_classes
    """Main function."""
    start_time = time.time()

    # Iterate over each dataset using zip()
    #########################################CREATE TILES###############################################################
    if Create_tiles:
        # Determine the target length based on the number of images
        target_len = len(image_path)

        # Prepare the arguments for filling
        args_to_fill = [image_path, mask_path, base_dir]

        # Call check_and_fill to adjust each argument list
        filled_values = check_and_fill(args_to_fill, target_len)

        # Unpack the filled values back into their respective variables
        image_path, mask_path, base_dir = filled_values

        for image_i, mask_i, base_dir_i in zip(image_path, mask_path, base_dir):
            split_raster(
                path_to_raster=image_i,
                path_to_mask=mask_i,
                patch_size=patch_size,
                patch_overlap=patch_overlap,
                base_dir=base_dir_i,
                split=split,
                max_empty=max_empty
            )

    #########################################TRAIN###############################################################

    if Train:
        # Determine the target length based on the number of models to train
        target_len = len(model_path)

        # Prepare the arguments for filling
        args_to_fill = [data_path, existing_model, BATCH_SIZE, visualize_data_example, enable_regression, CLASS_WEIGHTS,
                        ARCHITECTURE, EPOCHS, LEARNING_RATE, ENCODER_FACTOR, LR_FINDER, loss_func, monitor,
                        self_attention, VALID_SCENES, CODES, transforms, export_model_summary]

        # Call check_and_fill to adjust each argument list
        filled_values = check_and_fill(args_to_fill, target_len)

        # Unpack the filled values back into their respective variables
        data_path, existing_model, BATCH_SIZE, visualize_data_example, enable_regression, CLASS_WEIGHTS, ARCHITECTURE, EPOCHS, \
        LEARNING_RATE, ENCODER_FACTOR, LR_FINDER, loss_func, monitor, self_attention, VALID_SCENES, CODES, transforms, \
        export_model_summary = filled_values

        for data_path_i, existing_model_i, model_path_i, BATCH_SIZE_i, visualize_data_example_i, enable_regression_i, CLASS_WEIGHTS_i ,\
            ARCHITECTURE_i, EPOCHS_i, LEARNING_RATE_i, ENCODER_FACTOR_i, LR_FINDER_i, loss_func_i, monitor_i, self_attention_i, VALID_SCENES_i, CODES_i, transforms_i, export_model_summary_i  \
                in zip(data_path, existing_model, model_path, BATCH_SIZE, visualize_data_example, enable_regression, CLASS_WEIGHTS,
                ARCHITECTURE, EPOCHS, LEARNING_RATE, ENCODER_FACTOR, LR_FINDER, loss_func, monitor, self_attention, VALID_SCENES, CODES, transforms, export_model_summary):

                #run train function for each batch of input variables
                train_func(data_path_i, existing_model_i, model_path_i, BATCH_SIZE_i, visualize_data_example_i, enable_regression_i, CLASS_WEIGHTS_i,
                ARCHITECTURE_i, EPOCHS_i, LEARNING_RATE_i, ENCODER_FACTOR_i, LR_FINDER_i, loss_func_i, monitor_i, self_attention_i, VALID_SCENES_i, CODES_i, transforms_i, export_model_summary_i)

    #########################################Predict###############################################################
    if Predict:
        # Determine the target length based on the number of prediction models
        target_len = len(predict_model)

        # Prepare the arguments for filling
        args_to_fill = [predict_path, predict_model, merge, all_classes]

        # Call check_and_fill to adjust each argument list
        filled_values = check_and_fill(args_to_fill, target_len)

        # Unpack the filled values back into their respective variables
        predict_path, predict_model, merge, all_classes = filled_values

        for predict_path_i, predict_model_i, merge_i, all_classes_i in zip(predict_path, predict_model, merge, all_classes):
                #learn = load_learner(Path(predict_model))
                #predict_path_i = Path(predict_path_i)

                save_predictions(
                    predict_model_i,
                    predict_path_i,
                    regression,
                    merge_i,
                    all_classes_i,
                    specific_class,
                    large_file
                )
    end_time = time.time()
    print(f"The operation took {(end_time - start_time):.2f} seconds or {((end_time - start_time) / 60):.2f} minutes")

if __name__ == '__main__':
    main()



