from fastai.vision.models.xresnet import xresnet34
from fastai.vision.augment import Dihedral, Rotate, Brightness, Contrast, Saturation
from fastai.vision.core import imagenet_stats
from fastai.data.transforms import Normalize
from fastai.losses import MSELossFlat, CrossEntropyLossFlat, L1LossFlat

# PARAMETERS
Create_tiles = False
Train = False
Predict = True

# CREATE TILES
# if using without mask, set mask_path = None
image_path = r"Z:\B_CNN_DeepLearning\+Projekte\LuBi_GrünvolumenLandnutzungsklassifikation\Daten\berlin_rgbi_ndom_mask_stack2.tif"
mask_path = r"Z:\B_CNN_DeepLearning\+Projekte\LuBi_GrünvolumenLandnutzungsklassifikation\Daten\berlin_klassifikation_50cm_snap_rgbi_output_mask2.tif"
base_dir = r"Z:\B_CNN_DeepLearning\+Projekte\LuBi_GrünvolumenLandnutzungsklassifikation\Daten\tiles2"
#mask_path = None

patch_size = 400
patch_overlap = 0
split = [0.7, 0.3]
#split = [1]

# TRAINING
# if using on created tiles, set data_path = base_dir
data_path = r"Z:\B_CNN_DeepLearning\+Projekte\LuBi_GrünvolumenLandnutzungsklassifikation\Daten\tiles_combined"
model_path = r"C:\DeepLearning_Local\temp\Test_Deeplearning\Unet\gv_luc_35ep.pkl"
existing_model = None
BATCH_SIZE = 4  # 3 for xresnet50, 12 for xresnet34 with Tesla P100 (16GB)
EPOCHS = 35
LEARNING_RATE = 0.005
enable_regression = False
visualize_data_example = False
# only relevant for classification
CODES = ['background', 'class1','class2','class3','class4','class5','class6','class7']
#CODES = ['background', 'keingruendach', 'gruendach']
#CODES = ['background', 'unversiegelt', 'teilversiegelt', 'versiegelt','wasser']
#CLASS_WEIGHTS = "even"  # list (e.g. [3, 2, 5]) or string ("even" or "weighted")
CLASS_WEIGHTS = [0,0.33,0,0,0.33,0,0,0.34]
# PREDICTION
predict_path = r"\\QNAP2\projekte2\B_CNN_DeepLearning\+Projekte\LuBi_GrünvolumenLandnutzungsklassifikation\Daten\tiles_combined\vali\img_tiles"
predict_model = r"C:\DeepLearning_Local\temp\Test_Deeplearning\Unet\gv_luc_35ep.pkl"
merge = False
regression = False
# CONFIG END

enable_extra_parameters = True  # only for experienced users



# EXTRA PARAMETERS
ENCODER_FACTOR = 10  # minimal lr_rate factor
LR_FINDER = None  # None, "minimum", "steep", "valley", "slide"
VALID_SCENES = ['vali']
loss_func = None
# Regression: MSELossFlat(axis=1), L1LossFlat(axis=-1)
# Classification: CrossEntropyLossFlat(axis=1, weight=weights)
monitor = 'dice_multi'
# Regression: 'train_loss', 'valid_loss', 'r2_score'
# Classification: 'dice_multi', 'train_loss'
all_classes = False  # If all class predictions should be stored
specific_class = None  # None or integer of class -> Only this class will be stored
quantile_stretch = False  # If data should be stretched to a 99% quantile
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
