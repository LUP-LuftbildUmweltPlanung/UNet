import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import sys
from params import export_model_summary

from torch import nn, Tensor
from pathlib import Path
from typing import Optional

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

from utils import annot_min, find_lr


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
                    regression=False):
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

    model = to_device(models.unet.DynamicUnet(body, n_out=n_out, img_size=size, blur=False, blur_final=True,
                                              self_attention=False, y_range=None, norm_type=norm_type, last_cross=True,
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
               loss_func=None, monitor=None, existing_model=None):
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
    loss_func.func.weight = weights

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

    #print('weights_tensor: ',loss_func.func.weight)
    if existing_model is None:
        learn = unet_learner_MS(dls,  # DataLoaders
                                architecture,  # xResNet34
                                loss_func=loss_func,  # Weighted cross entropy loss
                                opt_func=Adam,  # Adam optimizer
                                metrics=metrics,
                                cbs=cbs,
                                regression=regression
                                )
    else:
        learn=load_learner(existing_model)
        learn.dls=dls
        learn.add_cb(CSVLogger())
        learn.loss_func=loss_func
        learn.opt_func=Adam

    #save model summary
    if export_model_summary:
        default_stdout =sys.stdout
        summary_path = str(path).rsplit('.', 1)[0] + "_model_summary.txt"
        sys.stdout = open(summary_path, 'w')
        print('Class_weights:',class_weights)
        print(learn.summary())
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
    #os.rename(learn.path / learn.csv_logger.fname, hist_path)
    shutil.move(learn.path / learn.csv_logger.fname, hist_path)
    learn.remove_cb(CSVLogger)

    hist = pd.read_csv(hist_path, header=0, index_col=None)
    train_loss = hist['train_loss'].tolist()
    valid_loss = hist['valid_loss'].tolist()

    plt.figure(figsize=(7, 7))
    plt.plot(train_loss, label='Training')
    plt.plot(valid_loss, label='Validation')

    if monitor not in ['train_loss', 'valid_loss']:
        monitor = hist['train_loss'].tolist()
        plt.plot(monitor, label='Monitor')
        annot_min(monitor)
        plt.ylim(0, np.max(monitor) * 1.3)
    else:
        annot_min(valid_loss)
        plt.ylim(0, 1.1)

    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Model Training Overview')
    plt.legend()
    plt.savefig(str(hist_path).rsplit('.', 1)[0] + '.png', dpi=200)

    return learn
