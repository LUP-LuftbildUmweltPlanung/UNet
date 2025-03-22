import os
import warnings
import numpy as np
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import math
import shutil
import mlflow.pytorch
import socket
import sys
from torch import nn, Tensor
import json
from pathlib import Path
from typing import Optional
import albumentations as A

from data import create_data_block
from utils import annot_min, find_lr, get_datatype, get_class_weights, visualize_data, \
    SegmentationAlbumentationsTransform, process_and_save_params, get_image_metadata

import fastai.vision.models as models
from fastai.vision.core import imagenet_stats
from fastai.vision.learner import model_meta, create_body

from fastai.layers import NormType
from fastai.learner import Learner
from fastai.learner import load_learner
from fastai.losses import MSELossFlat, CrossEntropyLossFlat, L1LossFlat, FocalLossFlat
from fastai.metrics import rmse, R2Score, DiceMulti, foreground_acc
from fastai.optimizer import Adam

from fastai.callback.progress import CSVLogger
from fastai.callback.tracker import SaveModelCallback
from fastai.data.transforms import Normalize
from fastai.torch_core import params, to_device, apply_init

from fastcore.basics import risinstance, defaults, ifnone
from fastcore.foundation import L

def log_metrics_mlflow(hist_path, monitor):
    """
    Logs training metrics (train_loss, valid_loss, dice_multi) from history CSV to MLflow.

    Parameters:
    - hist_path (Path): Path to the training history CSV.
    - monitor (str): The primary metric to monitor (e.g., "valid_loss", "dice_multi").
    """
    if not hist_path.exists():
        print(f"⚠️ Warning: Metrics file not found at {hist_path}")
        return

    # ✅ Read the training history
    hist = pd.read_csv(hist_path)

    # ✅ Log metrics for each epoch
    for epoch, row in hist.iterrows():
        mlflow.log_metric("train_loss", row["train_loss"], step=epoch)
        mlflow.log_metric("valid_loss", row["valid_loss"], step=epoch)
        mlflow.log_metric("dice_multi", row["dice_multi"], step=epoch)

        # ✅ Log primary monitoring metric separately (for MLflow visualization)
        if monitor in row:
            mlflow.log_metric(monitor, row[monitor], step=epoch)

    print(f"✅ Metrics logged to MLflow from {hist_path}")




def load_split_raster_params(json_path):
    """
    Load parameters from a JSON file and extract the values.

    Parameters:
    -----------
    json_path: Path to the JSON file containing the parameters.

    Returns:
    --------
    params: A dictionary containing the parameters.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, 'r') as json_file:
        params = json.load(json_file)

    return params


def _add_norm(dls, meta, pretrained):
    """Adds a normalization to a pretrained model."""
    if not pretrained:
        return
    stats = meta.get('stats')
    if stats is None:
        return
    if not dls.after_batch.fs.filter(risinstance(Normalize)):
        dls.add_tfms([Normalize.from_stats(*stats)], 'after_batch')


def default_split(m):
    """Default split of a model between body and head"""
    return L(m[0], m[1:]).map(params)


def _xresnet_split(m):
    """Splits XResnet between body and head."""
    return L(m[0][:3], m[0][3:], m[1:]).map(params)


_default_meta = {'cut': None, 'split': default_split}
_xresnet_meta = {'cut': -4, 'split': _xresnet_split, 'stats': imagenet_stats}


class Learner_adjust(Learner):
    """Edits the fastai Learner predict function to work with regression output."""

    def predict(self, item, rm_type_tfms=None, with_input=False):
        """Only contains the data-handling necessary to return regression outputs."""
        dl = self.dls.test_dl([Path(item)], rm_type_tfms=rm_type_tfms, num_workers=0)
        _, preds, _, dec_preds = self.get_preds(dl=dl, with_input=True, with_decoded=True)
        res = dec_preds[0], preds[0]
        return res


def unet_learner_MS(dls, arch, pretrained=True,
                    # learner args
                    loss_func=None, norm_type: Optional[NormType] = NormType, opt_func=Adam, lr=defaults.lr,
                    splitter=None, cbs=None, metrics=None, path=None,
                    model_dir='models', wd=None, wd_bn_bias=False, train_bn=True, moms=(0.95, 0.85, 0.95),
                    regression=False, self_attention=False):
    """
    Creates a fastai Unet Learner based on a classification architecture using Dynamic Unet.
    To allow for more input-bands, the first layer of the classification architecture is removed
    and replaced with a new convolutional layer.

    Parameters:
    -----------
        dls :       Dataloaders containing the paths to training and validation data
        arch :      Architecture to use as body for the Unet (e.g. xResNet34)
        loss_func : Loss function to use during training
        ...

    Returns:
    ---------
        learn :     A fastai Learner class

    References:
    ----------
        Based on the unet_learner function in fastai.vision.learner
    """
    size = next(iter(dls.train_ds))[0].shape[-2:]
    n_input_channels = next(iter(dls.train_ds))[0].size(0)

    meta = model_meta.get(arch, _default_meta)
    body = create_body(arch, pretrained, cut=None)

    prev_layer = body[0][0]
    body[0][0] = nn.Conv2d(n_input_channels, prev_layer.out_channels,
                           kernel_size=prev_layer.kernel_size,
                           stride=prev_layer.stride,
                           padding=prev_layer.padding,
                           bias=prev_layer.bias)

    if regression:
        n_out = 1
    else:
        n_out = len(dls.vocab)
    model = to_device(models.unet.DynamicUnet(body, n_out=n_out, img_size=size, blur=True, blur_final=True,
                                              self_attention=self_attention, y_range=None, norm_type=norm_type,
                                              last_cross=True,
                                              bottle=False), dls.device)

    splitter = ifnone(splitter, meta['split'])
    if regression:
        learn = Learner_adjust(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter,
                               cbs=cbs, metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias,
                               train_bn=train_bn, moms=moms)
    else:
        learn = Learner(dls=dls, model=model, loss_func=loss_func, opt_func=opt_func, lr=lr, splitter=splitter, cbs=cbs,
                        metrics=metrics, path=path, model_dir=model_dir, wd=wd, wd_bn_bias=wd_bn_bias,
                        train_bn=train_bn, moms=moms)
    # if pretrained and n_input_channels == 3:
    #     learn.freeze()
    #     apply_init(model[2], nn.init.kaiming_normal_)
    # else:
    #     apply_init(model, nn.init.kaiming_normal_)
    return learn


def train_unet(class_weights, dls, architecture, epochs, path, lr, encoder_factor, lr_finder=None, regression=False,
               loss_func=None, monitor=None, existing_model=None, self_attention=False, export_model_summary=False):
    """
    Takes a created unet_learner and trains the model on data provided within the dataloaders.

    Parameters:
    -----------
        class_weights :     Training weights for the different classes
        dls :               Fastai dataloader containing training and validation data
        architecture :      Classification body within the Unet
        epochs :            Training epochs
        path :              Path for storing training history and plot
        lr :                Learning rate
        encoder_factor :    lr / encoder_factor = lower bound of learning rate testing
        lr_finder :         Which method to use to find an optimal learning rate (default=None)
        regression :        If training a regression method (default=False -> classification)
        loss_func :         Which loss function to use (default=None -> MSELossFlat or CrossEntropyLossFlat)
        monitor :           Which training monitor to use (default=None -> 'valid_loss')

    Returns:
    ---------
        learn :             Unet learner now containing a trained model
    """

    weights = Tensor(class_weights).cuda()

    if regression:
        if loss_func is None:
            loss_func = MSELossFlat(axis=1)
        metrics = [rmse, R2Score()]
    else:
        if loss_func is None:
            loss_func = CrossEntropyLossFlat(axis=1, weight=weights)
        metrics = [DiceMulti()]

    if regression and monitor is None:
        monitor = 'r2_score'
    elif monitor is None:
        monitor = 'dice_multi'

    if monitor in ['train_loss', 'valid_loss']:
        comp = np.less
    else:
        comp = np.greater
        if monitor not in ['train_loss', 'valid_loss', 'r2_score', 'dice_multi']:
            warnings.warn("Monitor not recognised. Assuming maximization.")
    cbs = [SaveModelCallback(monitor=monitor, comp=comp, fname='best-model'), CSVLogger()]

    loss_func.func.weight = weights
    # print('weights_tensor: ',loss_func.func.weight)

    if existing_model is None:
        learn = unet_learner_MS(dls,  # DataLoaders
                                architecture,  # xResNet34
                                loss_func=loss_func,  # Weighted cross entropy loss
                                opt_func=Adam,  # Adam optimizer
                                metrics=metrics,
                                cbs=cbs,
                                regression=regression,
                                self_attention=self_attention
                                )
    else:
        learn = load_learner(existing_model)
        learn.dls = dls
        learn.add_cb(CSVLogger())
        learn.loss_func = loss_func
        learn.opt_func = Adam

    # save model summary
    if export_model_summary:
        default_stdout = sys.stdout
        summary_path = str(path).rsplit('.', 1)[0] + "_model_summary.txt"
        sys.stdout = open(summary_path, 'w')
        print('Class_weights:', class_weights)
        print(learn.summary())
        print(learn.model)
        sys.stdout.close()
        sys.stdout = default_stdout

    if lr_finder is not None:
        lr = find_lr(learn, lr_finder)
        print(f'Optimized learning rate: {lr}')

    learn.unfreeze()
    learn.fit_one_cycle(
        epochs,
        lr_max=slice(lr / encoder_factor, lr)
    )

    # plot loss
    learn.recorder.plot_loss()
    # move history
    hist_path = Path(str(path).rsplit('.', 1)[0] + "_history.csv")
    # os.rename(learn.path / learn.csv_logger.fname, hist_path)
    shutil.move(learn.path / learn.csv_logger.fname, hist_path)
    learn.remove_cb(CSVLogger)

    hist = pd.read_csv(hist_path, header=0, index_col=None)
    train_loss = hist['train_loss'].tolist()
    valid_loss = hist['valid_loss'].tolist()

    plt.figure(figsize=(7, 7))
    # plt.plot(train_loss, label='Training')
    plt.plot(valid_loss, label='Validation')

    if monitor not in ['train_loss', 'valid_loss']:
        monitor = hist['train_loss'].tolist()
        plt.plot(monitor, label='Training')
        annot_min(monitor)
        plt.ylim(0, np.max(monitor) * 1.3)
    else:
        annot_min(valid_loss)
        plt.ylim(0, 1.1)

    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Model Training Overview')
    plt.legend()
    loss_plot_path = str(hist_path).rsplit('.', 1)[0] + '_loss_plot.png'
    plt.savefig(loss_plot_path, dpi=200)
    plt.close()  # ✅ Free memory

    return learn

#### define train function to be able to use for train_multi and new params approach
def train_func(data_path, existing_model, model_Path, description, BATCH_SIZE, visualize_data_example,enable_regression, CLASS_WEIGHTS,
                ARCHITECTURE, EPOCHS, LEARNING_RATE, ENCODER_FACTOR, LR_FINDER, loss_func, monitor, self_attention,
               VALID_SCENES, CODES, transforms, split_idx, export_model_summary, aug_pipe, n_transform_imgs, info,
               class_zero, register_model):
    try:
        pc_name = socket.gethostname()
        #  Check if an MLflow run
        if mlflow.active_run():
            print(f"⚠️ Using existing MLflow run: {mlflow.active_run().info.run_id}")
        else:
            mlflow.start_run(run_name=description)
            print(f" Started MLflow run: {mlflow.active_run().info.run_id}")
            # Log system or run-level params/tags
            #mlflow.set_tag("mlflow.source.name", pc_name)
            mlflow.log_param("pc_name", pc_name)

            # Define Folder which contains "trai" and "vali" folder with "img_tiles" and "mask_tiles"
            data_path = Path(data_path)
            # Get datatype of training data
            print(data_path)
            dtype = get_datatype(data_path)
            patch_size, resolution, number_of_bands = get_image_metadata(data_path)
            if existing_model is not None:
                existing_model = Path(existing_model)
            if transforms:
                n_transform = math.ceil(BATCH_SIZE * n_transform_imgs)
                print(f"Applying Augmentation on ({n_transform}) images from ({BATCH_SIZE}) images")
                # Use the imported aug_pipe
                transforms = SegmentationAlbumentationsTransform(dtype, aug_pipe, n_transform_imgs=n_transform_imgs, split_idx= split_idx)
            else:
                # Define a default augmentation pipeline
                aug_pipe = A.Compose([
                    A.NoOp()  # No operation, pass-through transform
                ])
                transforms = SegmentationAlbumentationsTransform(dtype, aug_pipe, n_transform_imgs=n_transform_imgs)

            # Update new_path to include the 'models' directory and description
            new_path = Path(model_Path) / description

            # Create the directories if they don't exist
            new_path.mkdir(parents=True, exist_ok=True)

            # Path to save the model with .pkl extension
            model_path = new_path / f"{description}.pkl"

            # Save parameters to a JSON file
            process_and_save_params(data_path, aug_pipe, new_path, description, transforms=transforms, BATCH_SIZE=BATCH_SIZE,
                                    EPOCHS=EPOCHS, enable_regression=enable_regression,
                                    LEARNING_RATE=LEARNING_RATE, LR_FINDER=LR_FINDER, ENCODER_FACTOR=ENCODER_FACTOR,
                                    CLASS_WEIGHTS=CLASS_WEIGHTS,
                                    loss_func=loss_func, self_attention=self_attention, monitor=monitor,
                                    VALID_SCENES=VALID_SCENES,
                                    ARCHITECTURE=ARCHITECTURE, CODES=CODES, n_transform_imgs=n_transform_imgs, info=info,
                                    class_zero=class_zero)
            # Structure the parameters dictionary like the JSON file
            params_dict = {
                "transforms": bool(transforms),
                "BATCH_SIZE": BATCH_SIZE,
                "EPOCHS": EPOCHS,
                "enable_regression": enable_regression,
                "LEARNING_RATE": LEARNING_RATE,
                "LR_FINDER": LR_FINDER,
                "ENCODER_FACTOR": ENCODER_FACTOR,
                "CLASS_WEIGHTS": CLASS_WEIGHTS,
                "loss_func": str(loss_func),
                "self_attention": self_attention,
                "monitor": monitor,
                "VALID_SCENES": VALID_SCENES,
                "ARCHITECTURE": str(ARCHITECTURE),
                "CODES": CODES,
                "info": info,
                "class_zero": class_zero,
                "patch_size": str(patch_size),
                "resolution": str(resolution),
                "data_type": dtype,
                "number_of_bands": str(number_of_bands),
                "aug_params_": aug_pipe,
                "Percentage of augmented images": n_transform_imgs
            }
            mlflow.log_params(params_dict)

        # Data Block for Reference Storage
        db = create_data_block(valid_scenes=VALID_SCENES, codes=CODES, dtype=dtype, regression=enable_regression,
                               transforms=transforms)
        if enable_regression:
            CLASS_WEIGHTS = [1]
        elif isinstance(CLASS_WEIGHTS, str):
            if CLASS_WEIGHTS == "even":
                CLASS_WEIGHTS = np.ones(len(CODES)) / len(CODES)
            elif CLASS_WEIGHTS == "weighted":
                CLASS_WEIGHTS = get_class_weights(data_path, db)

        dls = db.dataloaders(data_path, bs=BATCH_SIZE, num_workers=0)
        dls.vocab = CODES

        inputs, targets = dls.one_batch()
        if visualize_data_example:
            inputs_np = inputs.cpu().detach().numpy()
            targets_np = targets.cpu().detach().numpy()
            visualize_data(inputs_np, model_path)
            os.system(str(model_path).rsplit('.', 1)[0] + "_image_plot.png")
            visualize_data(targets_np, model_path)
            os.system(str(model_path).rsplit('.', 1)[0] + "_mask_plot.png")

        print(f'Train files: {len(dls.train_ds)}, Test files: {len(dls.valid_ds)}')
        # print(f'Train files data: {dls.train_ds}, Test files data: {dls.valid_ds}')
        print(f'Input shape: {inputs.shape}, Output shape: {targets.shape}')
        print(f'Examplary value range INPUT: {inputs[0].min()} to {inputs[0].max()}')

        if enable_regression:
            print(f'Examplary value range TARGET: {targets[0].min()} to {targets[0].max()}')
        else:
            print(f"Class weights: {CLASS_WEIGHTS}")

        learn = train_unet(class_weights=CLASS_WEIGHTS, dls=dls, architecture=ARCHITECTURE, epochs=EPOCHS,
                           path=model_path, lr=LEARNING_RATE, encoder_factor=ENCODER_FACTOR, lr_finder=LR_FINDER,
                           regression=enable_regression, loss_func=loss_func, monitor=monitor,
                           existing_model=existing_model, self_attention=self_attention,
                           export_model_summary=export_model_summary)

        # Call `log_metrics_mlflow()` to log metrics to MLflow
        hist_path = Path(str(model_path).rsplit('.', 1)[0] + "_history.csv")
        log_metrics_mlflow(hist_path, monitor)

        def get_local_path_from_artifact_uri(artifact_uri: str, experiment_id: str) -> Path:
            """
            Convert MLflow Linux-style artifact URI to a proper Windows path.
            Handles artifact paths like: mlruns/<experiment_id>/<run_id>/artifacts
            """
            from urllib.parse import urlparse
            from pathlib import Path

            parsed = urlparse(artifact_uri)
            linux_str = parsed.path.replace("\\", "/")

            root_prefix = "/home/embedding/Data_Center/qnap3b"
            if not linux_str.startswith(root_prefix):
                raise ValueError(f"❌ Unexpected path format. Got: {linux_str}")

            # Extract relative part after root
            relative = linux_str[len(root_prefix):].lstrip("/")
            parts = Path(relative).parts

            try:
                # Expected: ['mlruns', experiment_id, run_id, ...]
                mlruns_idx = parts.index("mlruns")
                exp_id_from_uri = parts[mlruns_idx + 1]
                run_id = parts[mlruns_idx + 2]
                rest = parts[mlruns_idx + 3:]  # e.g., ['artifacts', 'models']

                # ✅ Reconstruct path as is
                corrected = Path("mlruns") / exp_id_from_uri / run_id / Path(*rest)
                return Path("N:/MnD/hub/mlflow") / corrected

            except Exception as e:
                raise ValueError(f"❌ Could not parse experiment_id and run_id from: {parts}\n{e}")

        artifact_dir = get_local_path_from_artifact_uri(
            mlflow.get_artifact_uri(),
            experiment_id=mlflow.active_run().info.experiment_id
        )


        #  Log Model to MLflow (Conditionally)
        try:
            artifact_path = "models"  # ✅ Use relative path instead of `mlflow.get_artifact_uri("models")`
            if register_model:
                mlflow.pytorch.log_model(learn.model, artifact_path=artifact_path, registered_model_name=description)
                print(f"✅ Model Registered in MLflow as: {description}")
            else:
                mlflow.pytorch.log_model(learn.model, artifact_path=artifact_path)
                print("✅ Model Logged to MLflow (but NOT registered).")

            # ✅ Export the model locally
            learn.export(model_path)
            print(f"✅ Training Completed! Model saved at: {model_path}")

        except Exception as e:
            print(f"❌ Error during training: {e}")

            # ✅ Export the model locally
            learn.export(model_path)
            print(f"✅ Training Completed! Model saved at: {model_path}")

        except Exception as e:
            print(f"❌ Error during training: {e}")
        # ✅ Copy all model files to the MLflow artifact directory manually
        import shutil
        try:
            print(f"📁 Copying model output files to: {artifact_dir}")
            os.makedirs(artifact_dir, exist_ok=True)

            for file in os.listdir(new_path):
                src_file = os.path.join(new_path, file)
                dst_file = os.path.join(artifact_dir, file)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, dst_file)
                    print(f"✅ Copied: {file}")
        except Exception as copy_error:
            print(f"❌ Error while copying model files to artifact dir: {copy_error}")
        # ✅ Log the artifacts to MLflow so they appear in the UI
        for file in os.listdir(new_path):
            src_file = os.path.join(new_path, file)
            if os.path.isfile(src_file):
                mlflow.log_artifact(src_file)
                print(f"📦 Logged artifact: {file}")


    finally:
        # ✅ Ensure the MLflow run is closed properly
        if mlflow.active_run():
            print(f"✅ Ending MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()
