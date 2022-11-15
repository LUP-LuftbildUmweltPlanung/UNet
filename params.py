from fastai.vision.models.xresnet import xresnet34
from fastai.vision.augment import Dihedral, Rotate, Brightness, Contrast, Saturation
from fastai.vision.core import imagenet_stats
from fastai.data.transforms import Normalize
from fastai.losses import MSELossFlat, CrossEntropyLossFlat, L1LossFlat

# PARAMETERS
Create_tiles = False
Train = True
Predict = False

######################################################
#################### TRAINING ########################
######################################################

# if using without mask, set mask_path = None
image_path = r"C:\DeepLearning_Local\temp\Test_Deeplearning\09_11_2022\Unet_s2\berlin_composite_fullbands_16bit.tif"
mask_path = r"C:\DeepLearning_Local\temp\Test_Deeplearning\09_11_2022\Unet_s2\berlin_gv_groundtruth_2020_10m_class.tif"
base_dir = r"C:\DeepLearning_Local\temp\Test_Deeplearning\09_11_2022\Unet_s2\tiles_test"
#mask_path = None

patch_size = 400
patch_overlap = 0
split = [0.7,0.3]
#split = [1]

############################################################
#################### TRAINING ##############################
############################################################
# if using on created tiles, set data_path = base_dir
data_path = r"C:\DeepLearning_Local\temp\Test_Deeplearning\09_11_2022\Unet_s2\tiles_test"
model_path = r"C:\DeepLearning_Local\temp\Test_Deeplearning\09_11_2022\Unet_s2\tiles_test\test.pkl"
existing_model = None
BATCH_SIZE = 4  # 3 for xresnet50, 12 for xresnet34 with Tesla P100 (16GB)
EPOCHS = 12
LEARNING_RATE = 0.005
enable_regression = True
visualize_data_example = False
# only relevant for classification
CODES = ['background', '1','2','3']
#CODES = ['background', 'keingruendach', 'gruendach']
#CODES = ['background', 'unversiegelt', 'teilversiegelt', 'versiegelt','wasser']
#CLASS_WEIGHTS = "even"  # list (e.g. [3, 2, 5]) or string ("even" or "weighted")
CLASS_WEIGHTS = "even"

########################################################
#################### PREDICTION ########################
########################################################
predict_path = r"D:\GV_Berlin_Deepl_Predict\Magdeburg\tiles\img_tiles"
predict_model = r"Z:\B_CNN_DeepLearning\+Projekte\LuBi_GrünvolumenLandnutzungsklassifikation\Results\Modelle\gv_luc_9ep_new.pkl"
merge = True
regression = False
# CONFIG END


############################################################
#################### EXTRA PARAMTERS #######################
############################################################

enable_extra_parameters = False  # only for experienced users


ENCODER_FACTOR = 10  # minimal lr_rate factor
LR_FINDER = None  # None, "minimum", "steep", "valley", "slide"
VALID_SCENES = ['vali']
loss_func = None
# Regression: MSELossFlat(axis=1), L1LossFlat(axis=-1)
# Classification: CrossEntropyLossFlat(axis=1, weight=weights), FocalLossFlat(axis=1)
monitor = None
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
