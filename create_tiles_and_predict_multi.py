from create_tiles_unet import split_raster
from predict import save_predictions
from fastai.learner import load_learner
from pathlib import Path
import time

# PARAMETERS
Create_tiles = True
Predict = False

######################################################
#################### CREATE TILES ####################
######################################################

# if using without mask, set mask_path = None
image_path = [r"H:\+DeepLearning_Extern\stacks_and_masks\stacks\Leipzig\2017\leipzig_RGBI_TOP_2017_20cm.tif",
              r"H:\+DeepLearning_Extern\stacks_and_masks\stacks\Leipzig\2022\leipzig_TOP_maerz_2022_rgbi_stack_20cm.tif"
              ]

mask_path = [r"H:\+DeepLearning_Extern\stacks_and_masks\masks\Leipzig\2017\versiegelung\leipzig_versiegelung_2017_20cm_nachklass_final.tif",
             None
             ]

base_dir = [r"H:\+DeepLearning_Extern\versiegelung_klassifikation\Daten\tiles_leipzig_neu_20cm\img_mask_4class_tiles_2017_20cm",
            r"H:\+DeepLearning_Extern\versiegelung_klassifikation\Daten\tiles_leipzig_neu_20cm\img_tiles_2022_20cm"
            ]


patch_size = 400
patch_overlap = 0.2 #0
split = [1] #[0.7,0.3]
#for prediction patch_overlap = 0.2 to prevent edge artifacts and split = [1] to predict full image
max_empty = 0.9  # Maximum no data area in created image crops

########################################################
#################### PREDICTION ########################
########################################################
predict_path = [
            r"T:\2023_VersiegelungLeipzig\Prozessierung\Vergleich_20cm_50cm_Modell\vali_tiles_aoi_20cm",
            ]

predict_model = r"H:\+DeepLearning_Extern\versiegelung_klassifikation\Models\legacy_models\unet_rgbi_20cm_3class\versiegelung_leipzig_3class_20ep_focalloss_newmanualw_lr005_20cm.pkl"
merge = True
regression = False

all_classes = False # If all class predictions should be stored
specific_class = None  # None or integer of class -> Only this class will be stored
large_file = False # If predicted probabilities s¶hould be stretched to int8 to increase storage capacity

# CONFIG END

def main():
    """Main function."""
    start_time = time.time()
    # Iterate over each dataset using zip()
    if Create_tiles:
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

    if Predict:
        for predict_path_i in predict_path:
                learn = load_learner(Path(predict_model))
                predict_path_i = Path(predict_path_i)

                save_predictions(
                    learn,
                    predict_path_i,
                    regression,
                    merge,
                    all_classes,
                    specific_class,
                    large_file
                )
    end_time = time.time()
    print(f"The operation took {(end_time - start_time):.2f} seconds or {((end_time - start_time) / 60):.2f} minutes")

if __name__ == '__main__':
    main()
