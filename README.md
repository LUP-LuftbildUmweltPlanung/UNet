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
* The following setup was tested with Python 3.10.20, CUDA 12.8, PyTorch 2.8.0, and fastai 2.5.1.
### For Windows & Linux
#### clone the Stable UNet repository
* `conda create -n UNet5090 python=3.10.20 -y`
* `conda activate UNet5090`
#### Install geospatial and scientific dependencies
* `conda install -c conda-forge gdal=3.10.3 rasterio=1.4.3 fiona=1.10.1 geopandas=1.1.3 numpy=2.2.6 pandas=2.3.3 scipy=1.15.2 scikit-learn=1.7.2 matplotlib-base=3.10.9 -y`
#### Install PyTorch with CUDA 12.8
* `pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128`
#### Install project requirements
* `cd ../UNet/environment`
* `pip install -r requirements.txt`
#### Install fastai and helper packages
* `pip install fastai==2.5.1 --no-deps`
* `pip install fastcore==1.3.29 fastdownload==0.0.5 fastprogress==1.0.5 spacy==3.8.14`
#### Patch fastai for compatibility with newer PyTorch
fastai 2.5.1 requires a small compatibility patch when used with PyTorch 2.8.0.
* `python -c "import fastai, pathlib; p=pathlib.Path(fastai.__file__).parent/'callback'/'progress.py'; s=p.read_text(); old=\"self.pbar.comment = f'{self.smooth_loss:.4f}'\"; new=\"self.pbar.comment = f'{float(self.smooth_loss):.4f}'\"; p.write_text(s.replace(old,new)); print('patched:', p)"`

### Executing program
* set parameters and run in params_and_main.py
*  `Note` :  to run the script with Mlflow you need to adjust the mlflow_config file.

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
