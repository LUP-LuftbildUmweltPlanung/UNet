# UNet

A DeepLearning Architecture for image segmentation.

## Description

This repository contains the code necessary to run a [UNet](https://arxiv.org/abs/1505.04597) based on the Dynamic Unet implementation of [fastai](https://www.fast.ai/). 
The implementation uses the PyTorch DeepLearning framework. UNet is used for image segmentation (pixel-wise classification).
The repository contains all code necessary to preprocess large tif-images, run training and validation, and perform predictions using the trained models.

## Getting Started

### Dependencies

* GDAL, Pytorch-fast.ai, Scipy ... (see installation)
* Cuda-capable GPU ([overview here](https://developer.nvidia.com/cuda-gpus))
* Anaconda ([download here](https://www.anaconda.com/products/distribution))
* developed on Windows 10

### Installation

#### For Windows
* clone the Stable UNet repository
* `conda create --name UNet python==3.9.6`
* `conda activate UNet`
* `cd ../UNet/environment`
* `pip install -r requirements.txt`

#### For Linux
* clone the Stable UNet repository
* `conda create --name UNet python==3.9.6`
* `conda activate UNet`
* `pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`
* `cd ../UNet/environment`
* `pip install -r requirements_Linux.txt`


### Executing program

* set parameters and run in params_and_main.py

## Help/Known Issues

* None yet

# Info

## Authors

* [Benjamin Stöckigt](https://github.com/benjaminstoeckigt)
* [Malik-Manel Hashim](https://github.com/irukandi) 
* [Sebastian Lehmler](https://github.com/SebastianLeh)
* [Shadi Ghantous](https://github.com/Shadiouss)


## Version History

* 0.1
    * Initial Release

## License

Not licensed

## Acknowledgments

Inspiration, code snippets, etc.

* [fastai](https://www.fast.ai/)
* [fastai documentation](https://docs.fast.ai/)
* [UNet tutorial by Deep Learning Berlin](https://deeplearning.berlin/satellite%20imagery/computer%20vision/fastai/2021/02/17/Building-Detection-SpaceNet7.html)
* [UNet adjustable input-channels tutorial by Navid Panchi](https://github.com/navidpanchi/N-Channeled-Input-UNet-Fastai/blob/master/N-Channeled-Input-UNet%20.ipynb)
* [UNet paper](https://arxiv.org/abs/1505.04597)
