# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 09:12:59 2021

@author: Benny, Manel
"""

import os
import warnings

import numpy as np
import torch
import shutil

from pathlib import Path

from fastai.learner import load_learner
from fastai.callback.progress import CSVLogger

from data import create_data_block
from create_tiles_unet import split_raster
from predict import save_predictions
from train import train_unet
from utils import get_datatype, get_class_weights, visualize_data, find_lr  # Smoothl1

from params import *

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


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
    enable_transforms = False
    large_file = False
    max_empty = 0.9
    ARCHITECTURE = xresnet34
    transforms = None

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
        max_empty=max_empty
    )

if Train:
    # Train Model Start
    # Define Folder which contains "trai" and "vali" folder with "img_tiles" and "mask_tiles"
    data_path = Path(data_path)

    if existing_model is not None:
        existing_model = Path(existing_model)

    model_path = Path(model_path)

    # Get datatype of training data
    dtype = get_datatype(data_path)
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
        visualize_data(inputs_np)
        os.system(str(model_path).rsplit('.', 1)[0] + "_image_plot.png")
        visualize_data(targets_np)
        os.system(str(model_path).rsplit('.', 1)[0] + "_mask_plot.png")

    print(f'Train files: {len(dls.train_ds)}, Test files: {len(dls.valid_ds)}')
    #print(f'Train files data: {dls.train_ds}, Test files data: {dls.valid_ds}')
    print(f'Input shape: {inputs.shape}, Output shape: {targets.shape}')
    print(f'Examplary value range INPUT: {inputs[0].min()} to {inputs[0].max()}')


    if enable_regression:
        print(f'Examplary value range TARGET: {targets[0].min()} to {targets[0].max()}')
    else:
        print(f"Class weights: {CLASS_WEIGHTS}")
    if existing_model is None:
        learn = train_unet(class_weights=CLASS_WEIGHTS, dls=dls, architecture=ARCHITECTURE, epochs=EPOCHS,
                           path=model_path, lr=LEARNING_RATE, encoder_factor=ENCODER_FACTOR, lr_finder=LR_FINDER,
                           regression=enable_regression, loss_func=loss_func, monitor=monitor)

        learn.export(model_path)
    else:
        learn = load_learner(existing_model)
        learn.dls = dls
        learn.add_cb(CSVLogger())
        if enable_regression:
            learn.loss_func = MSELossFlat(axis=1)
        else:
            weights = torch.Tensor(CLASS_WEIGHTS).cuda()
            learn.loss_func = CrossEntropyLossFlat(axis=1, weight=weights)
        # alternative: learn.loss_func = Smoothl1

        if enable_regression:
            if loss_func is None:
                learn.loss_func = MSELossFlat(axis=1)
        else:
            if loss_func is None:
                learn.loss_func = CrossEntropyLossFlat(axis=1, weight=weights)

        if LR_FINDER is not None:
            LEARNING_RATE = find_lr(learn, LR_FINDER)
            print(f'Optimized learning rate: {LEARNING_RATE}')

        learn.unfreeze()
        learn.fit_one_cycle(EPOCHS, lr_max=slice(LEARNING_RATE / ENCODER_FACTOR, LEARNING_RATE))
        learn.export(model_path)
        hist_path = Path(str(model_path).rsplit('.', 1)[0] + "_history.csv")
        shutil.move(learn.path / learn.csv_logger.fname, hist_path)
        learn.remove_cb(CSVLogger)


if Predict:
    learn = load_learner(Path(predict_model))
    predict_path = Path(predict_path)
    save_predictions(learn, predict_path, regression, merge, all_classes, specific_class, large_file)

