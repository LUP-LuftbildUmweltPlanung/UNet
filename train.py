import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import shutil
import mlflow.pytorch
import socket
import sys
import shutil
import tempfile
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from torch import nn, Tensor
import torch.nn.functional as F
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
from fastai.losses import MSELossFlat, CrossEntropyLossFlat, L1LossFlat, FocalLossFlat, DiceLoss
from fastai.metrics import rmse, R2Score, DiceMulti, foreground_acc
from fastai.optimizer import Adam

from fastai.callback.progress import CSVLogger
from fastai.callback.tracker import SaveModelCallback
from fastai.data.transforms import Normalize
from fastai.torch_core import params, to_device, apply_init

from fastcore.basics import risinstance, defaults, ifnone, store_attr
from fastcore.foundation import L

def log_metrics_mlflow(hist_path, monitor):
    """
    Logs training metrics (train_loss, valid_loss, dice_multi) from history CSV to MLflow.

    Parameters:
    - hist_path (Path): Path to the training history CSV.
    - monitor (str): The primary metric to monitor (e.g., "valid_loss", "dice_multi").
    """
    if not hist_path.exists():
        print(f" Warning: Metrics file not found at {hist_path}")
        return

    #  Read the training history
    hist = pd.read_csv(hist_path)

    #  Log metrics for each epoch
    for epoch, row in hist.iterrows():
        mlflow.log_metric("train_loss", row["train_loss"], step=epoch)
        mlflow.log_metric("valid_loss", row["valid_loss"], step=epoch)
        mlflow.log_metric("dice_multi", row["dice_multi"], step=epoch)

        #  Log primary monitoring metric separately (for MLflow visualization)
        if monitor in row:
            mlflow.log_metric(monitor, row[monitor], step=epoch)

    print(f" Metrics logged to MLflow from {hist_path}")




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


# Add combined loss function (Dice Loss and Focal loss combined)
class CombinedLoss:

    def __init__(self, axis=1, smooth=1., alpha=1.):
        store_attr()
        self.focal_loss = FocalLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis, smooth)

    def __call__(self, pred, targ):
        return self.focal_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x):    return x.argmax(dim=self.axis)

    def activation(self, x): return F.softmax(x, dim=self.axis)


# Add Attention Gates to code
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi  # Verstärkt relevante Features


def add_attention_to_unet(unet):
    for name, layer in unet.named_children():
        if isinstance(layer, models.unet.DynamicUnet):
            for idx in range(len(layer.sfs)):
                in_channels = layer.sfs[idx].features.shape[1]
                gate_channels = layer.sfs[max(idx - 1, 0)].features.shape[1]
                attn_gate = AttentionGate(gate_channels, in_channels, in_channels // 2)
                setattr(layer, f'attn_{idx}', attn_gate)
    return unet

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
                    regression=False, self_attention=False, attention_gates=False):
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

    if attention_gates:
        model = add_attention_to_unet(model)  # Add Attention Gates to model

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
               loss_func=None, monitor=None, existing_model=None, self_attention=False, export_model_summary=False, attention_gates=False):
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
        monitor = 'valid_loss'

    if monitor in ['train_loss', 'valid_loss']:
        comp = np.less
    else:
        comp = np.greater
        if monitor not in ['train_loss', 'valid_loss', 'r2_score', 'dice_multi']:
            warnings.warn("Monitor not recognised. Assuming maximization.")
    cbs = [SaveModelCallback(monitor=monitor, comp=comp, fname='best-model'), CSVLogger()]

    if isinstance(loss_func, CrossEntropyLossFlat):
        loss_func.func.weight = weights
        # print('weights_tensor: ',loss_func.func.weight)
    elif isinstance(loss_func, CombinedLoss):
        loss_func.focal_loss.func.weight = weights
    else:
        pass

    if existing_model is None:
        learn = unet_learner_MS(dls,  # DataLoaders
                                architecture,  # xResNet34
                                loss_func=loss_func,  # Weighted cross entropy loss
                                opt_func=Adam,  # Adam optimizer
                                metrics=metrics,
                                cbs=cbs,
                                regression=regression,
                                self_attention=self_attention,
                                attention_gates=attention_gates
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
        summary_path = Path(path.with_stem(path.stem + "_model_summary").with_suffix(".txt"))
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
    hist_path = Path(path.with_stem(path.stem + "_history").with_suffix(".csv"))
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
    plt.close()  #  Free memory

    return learn

### define train function to be able to use for train_multi and new params approach
def train_func(data_path, existing_model, model_Path, description, BATCH_SIZE, visualize_data_example,enable_regression, CLASS_WEIGHTS,
                ARCHITECTURE, EPOCHS, LEARNING_RATE, ENCODER_FACTOR, LR_FINDER, loss_func, monitor, self_attention,
               VALID_SCENES, CODES, transforms, split_idx, export_model_summary, aug_pipe, n_transform_imgs, info,
               class_zero, register_model, attention_gates):
    try:
        pc_name = socket.gethostname()
        #  Check if an MLflow run
        if mlflow.active_run():
            print(f" Using existing MLflow run: {mlflow.active_run().info.run_id}")
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
                                    class_zero=class_zero, attention_gates=attention_gates)
            # Structure the parameters dictionary like the JSON file
            params_dict = {
                "data_path": str(data_path),
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
                "n_transform_imgs": n_transform_imgs,
                "info": info,
                "class_zero": class_zero,
                "patch_size": str(patch_size),
                "resolution": str(resolution),
                "data_type": dtype,
                "number_of_bands": str(number_of_bands),
                "aug_params_": aug_pipe,
                "Percentage of augmented images": n_transform_imgs,
                "class_zero": class_zero
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
        # Convert FastAI datasets to DataFrames and log as input
        try:
            train_items = dls.train_ds.items if hasattr(dls.train_ds, "items") else None
            valid_items = dls.valid_ds.items if hasattr(dls.valid_ds, "items") else None

            if isinstance(train_items, (list, tuple, np.ndarray)):
                train_df = pd.DataFrame(train_items, columns=["train_paths"])
                dataset_train = mlflow.data.from_pandas(train_df, name="training_dataset")
                mlflow.log_input(dataset_train, context="training")
                print(" Training dataset logged to MLflow.")

            if isinstance(valid_items, (list, tuple, np.ndarray)):
                valid_df = pd.DataFrame(valid_items, columns=["valid_paths"])
                dataset_valid = mlflow.data.from_pandas(valid_df, name="validation_dataset")
                mlflow.log_input(dataset_valid, context="validation")
                print(" Validation dataset logged to MLflow.")

        except Exception as e:
            print(f" Failed to log datasets to MLflow: {e}")

        inputs, targets = dls.one_batch()
        # Prepare inputs for logging (move to CPU and detach)
        sample_input = inputs.cpu().detach()
        sample_output = targets.cpu().detach()

        # Infer the input/output schema
        signature = infer_signature(sample_input.numpy(), sample_output.numpy())
        input_example = sample_input.numpy()

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
                           export_model_summary=export_model_summary, attention_gates=attention_gates)

        # Call `log_metrics_mlflow()` to log metrics to MLflow
        hist_path = Path(str(model_path).rsplit('.', 1)[0] + "_history.csv")
        log_metrics_mlflow(hist_path, monitor)

        #  Define relative artifact path inside MLflow run
        artifact_path = "models"

        try:
            #  Log the model to MLflow (register or just log)
            if register_model:
                mlflow.pytorch.log_model(
                    learn.model,
                    artifact_path=artifact_path,
                    registered_model_name=description,
                    signature=signature,
                    input_example=input_example
                )
                print(f" Model Registered in MLflow under name: {description}")
            else:
                mlflow.pytorch.log_model(learn.model, artifact_path=artifact_path, signature=signature, input_example=input_example)
                print(" Model Logged to MLflow (but NOT registered)")

            #  Export model locally
            learn.export(model_path)
            print(f" Training Completed! Model saved at: {model_path}")

            #  Log all files in new_path as artifacts
            for file in os.listdir(new_path):
                src_file = os.path.join(new_path, file)
                if os.path.isfile(src_file):
                    mlflow.log_artifact(src_file)
                    print(f" Logged artifact: {file}")

            # 🔍 List logged artifacts for confirmation
            client = MlflowClient()
            artifacts = client.list_artifacts(mlflow.active_run().info.run_id, artifact_path)
            print("🔍 Artifacts in 'models/':", [a.path for a in artifacts])

        except Exception as e:
            print(f" Error during MLflow model logging: {e}")
            try:
                learn.export(model_path)
                print(f" Model fallback exported to: {model_path}")
            except Exception as export_error:
                print(f" Failed to export model fallback: {export_error}")
    finally:
        if mlflow.active_run():
            print(f" Ending MLflow run: {mlflow.active_run().info.run_id}")
            mlflow.end_run()
