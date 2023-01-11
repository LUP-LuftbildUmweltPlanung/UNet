from fastai.vision.models.xresnet import xresnet34
from fastai.vision.augment import Dihedral, Rotate, Brightness, Contrast, Saturation
from fastai.vision.core import imagenet_stats
from fastai.data.transforms import Normalize
from fastai.losses import MSELossFlat, CrossEntropyLossFlat, L1LossFlat, FocalLossFlat


# PARAMETERS
Create_tiles = False
Train = True
Predict = False

######################################################
#################### CREATE TILES ####################
######################################################

# if using without mask, set mask_path = None
image_path = r"E:\+DeepLearning_Extern\stacks_and_masks\stacks\Leipzig\leipzig_rgbi_ndom_stack_2017.tif"
mask_path = r"E:\+DeepLearning_Extern\versiegelung_klassifikation\Daten\daten_leipzig\versiegelung_reclass_haus\leipzig_versiegelung_2017_50cm_nachklassif.tif"
base_dir = r"E:\+DeepLearning_Extern\versiegelung_klassifikation\Daten\tiles_nachklass_nDOM_leipzig"
#mask_path = None

patch_size = 400
patch_overlap = 0
split = [0.7,0.3]
#for prediction patch_overlap = 0.2 to prevent edge artifacts and split = [1] to predict full image


############################################################
#################### TRAINING ##############################
############################################################
# if using on created tiles, set data_path = base_dir
data_path = r"E:\+DeepLearning_Extern\versiegelung_klassifikation\Daten\tiles_leipzig"
model_path = r"E:\+DeepLearning_Extern\versiegelung_klassifikation\Models\temp_test\versiegelung_test.pkl"
existing_model = None
BATCH_SIZE = 4  # 3 for xresnet50, 12 for xresnet34 with Tesla P100 (16GB)
EPOCHS = 30
LEARNING_RATE = 0.005
enable_regression = False
visualize_data_example = True
export_model_summary = True
# only relevant for classification
CODES = ['background', 'nicht_versiegelt','teilversiegelt','versiegelt', 'versiegelt_gebaeude']
#CLASS_WEIGHTS = "even"  # list (e.g. [3, 2, 5]) or string ("even" or "weighted")
CLASS_WEIGHTS = "even"
weights = None

########################################################
#################### PREDICTION ########################
########################################################
predict_path = r"E:\+DeepLearning_Extern\versiegelung_klassifikation\Daten\tiles_nachklass_nDOM_leipzig\vali\img_tiles"
predict_model = r"E:\+DeepLearning_Extern\versiegelung_klassifikation\Models\versiegelung_leipzig_30ep_focalloss_weighted_lr005_04_01_22.pkl"
merge = False
regression = False
# CONFIG END


############################################################
#################### EXTRA PARAMTERS #######################
############################################################

enable_extra_parameters = True  # only for experienced users

ENCODER_FACTOR = 10  # minimal lr_rate factor
LR_FINDER = None  # None, "minimum", "steep", "valley", "slide"
VALID_SCENES = ['vali']
loss_func = FocalLossFlat(axis=1, weight=weights)
# Regression: MSELossFlat(axis=1), L1LossFlat(axis=-1)
# Classification: CrossEntropyLossFlat(axis=1, weight=weights), FocalLossFlat(axis=1, weight=weights)
monitor = 'dice_multi'
# Regression: 'train_loss', 'valid_loss', 'r2_score'
# Classification: 'dice_multi', 'train_loss'
all_classes = False  # If all class predictions should be stored
specific_class = None  # None or integer of class -> Only this class will be stored
large_file = False  # If predicted files should be converted to int8 to increase storage capacity
max_empty = 0.9  # Maximum no data area in created image crops
ARCHITECTURE = xresnet34
transforms = [Dihedral(0.5),  # Horizontal and vertical flip
              Rotate(max_deg=180, p=0.5),  # Rotation in any direction possible
              Brightness(0.2, p=0.5),
              Contrast(0.2, p=0.5),
              Saturation(0.2),
              Normalize.from_stats(*imagenet_stats)]
transforms = None
# EXTRA END