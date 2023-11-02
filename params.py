from fastai.vision.models.xresnet import xresnet34, xresnet101, xresnet50, xresnet34_deep, xresnet18
from fastai.vision.augment import Dihedral, Rotate, Brightness, Contrast, Saturation
from fastai.vision.core import imagenet_stats
from fastai.data.transforms import Normalize
from fastai.losses import MSELossFlat, CrossEntropyLossFlat, L1LossFlat, FocalLossFlat, DiceLoss


# PARAMETERS
Create_tiles = False
Train = False
Predict = True

######################################################
#################### CREATE TILES ####################
######################################################

# if using without mask, set mask_path = None
image_path = r'Q:\LuBi\potsdam\top_cir_2022.tif'
mask_path = r"Q:\LuBi\potsdam\top_cir_2022_reinb_mitm_mitnotree_mitshadow85gr95nir_ref.tif"
base_dir = r'H:\+DeepLearning_Extern\baumarten\reference\tiles\treespecies_potsdam_mitnotree_mitshadow85gr95nir_400ps'
#mask_path = r"E:\+DeepLearning_Extern\stacks_and_masks\masks\Potsdam\veg_height\potsdam_vh_2022_50cm.tif"

patch_size = 400
patch_overlap = 0 #0.2
split = [0.8, 0.2]
#for prediction patch_overlap = 0.2 to prevent edge artifacts and split = [1] to predict full image


############################################################
#################### TRAINING ##############################
############################################################
# if using on created tiles, set data_path = base_dir
data_path = r"H:\+DeepLearning_Extern\baumarten\reference\tiles\combination_treespecies_bielefeld_potsdammitmisch_mitshadow85gr95nir_ohnenotree_400ps"
model_path = r"H:\+DeepLearning_Extern\baumarten\models\tree_species_400ps_10classes_bielefeld_potsdammitmisch_100ep_focal.pkl"
existing_model = None #r"H:\+DeepLearning_Extern\baumarten\models\tree_species_400ps_9classes_rhoenfrankfurt.pkl"
BATCH_SIZE = 4  # 3 for xresnet50, 12 for xresnet34 with Tesla P100 (16GB)
EPOCHS = 100
LEARNING_RATE = 0.001
enable_regression = False
visualize_data_example = False
export_model_summary = False
# only relevant for classification
#CODES = ['background', 'no_road','road']
#CODES = ['background','bu','ei','fi','ki','pap'] #frankfurt
#CODES = ['background','bu','ei','fi','ki','pap','erle','douglasie','ela'] #frankfurt und bielefeld
CODES = ['background','bu','ei','fidougl','ki','pap','erle','shadow','ela','birke','robinie'] # potsdam oder x mit potsdam
#CLASS_WEIGHTS = "even"  # list (e.g. [3, 2, 5]) or string ("even" or "weighted")
CLASS_WEIGHTS = "weighted" #[0.01, 0.3, 0.69]


########################################################
#################### PREDICTION ########################
########################################################
predict_path = r"H:\+DeepLearning_Extern\baumarten\reference\prediction\prediction_zeitzer_forst_tiles_400ps\img_tiles"
predict_model = r"H:\+DeepLearning_Extern\baumarten\models\tree_species_400ps_10classes_bielefeld_potsdammitmisch_100ep_focal.pkl"
merge = True
regression = False
# CONFIG END


############################################################
#################### EXTRA PARAMTERS #######################
############################################################

enable_extra_parameters = True # only for experienced users

ENCODER_FACTOR = 10  # minimal lr_rate factor
LR_FINDER = None  # None, "minimum", "steep", "valley", "slide"
VALID_SCENES = ['vali']
loss_func = FocalLossFlat(gamma=2, axis=1)
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
transforms = [Dihedral(0.5),  # Horizontal and vertical flip
              Rotate(max_deg=180, p=0.5),  # Rotation in any direction possible
              Brightness(0.2, p=0.5),
              Contrast(0.2, p=0.5),
              Saturation(0.2),
              Normalize.from_stats(*imagenet_stats)]
transforms = None
# EXTRA END