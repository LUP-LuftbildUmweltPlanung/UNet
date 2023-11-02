# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 09:12:59 2021

@author: Benny, Manel, Sebastian
"""

import os
import warnings

import numpy as np
import torch
import shutil

from pathlib import Path
from sklearn.metrics import confusion_matrix
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

    learn = train_unet(class_weights=CLASS_WEIGHTS, dls=dls, architecture=ARCHITECTURE, epochs=EPOCHS,
                       path=model_path, lr=LEARNING_RATE, encoder_factor=ENCODER_FACTOR, lr_finder=LR_FINDER,
                       regression=enable_regression, loss_func=loss_func, monitor=monitor, existing_model=existing_model)

    learn.export(model_path)

    if not regression:
        valid_preds, valid_labels = learn.get_preds(dl=dls.valid)

        # Convert predictions to class labels (assuming it's a multi-class classification problem)
        valid_preds = np.argmax(valid_preds, axis=1)
        # Assuming valid_labels and valid_preds are tensors
        valid_labels = valid_labels.cpu().numpy()  # Convert to NumPy array
        valid_preds = valid_preds.cpu().numpy()  # Convert to NumPy array
        ##flatten x y dimension
        valid_labels_flat = valid_labels.ravel()
        valid_preds_flat = valid_preds.ravel()
        # Calculate the confusion matrix
        confusion = confusion_matrix(valid_labels_flat, valid_preds_flat)
        # Print or use the confusion matrix as needed
        print("Confusion Matrix:")
        print(confusion)


if Predict:
    learn = load_learner(Path(predict_model))
    predict_path = Path(predict_path)
    save_predictions(learn, predict_path, regression, merge, all_classes, specific_class, large_file)

