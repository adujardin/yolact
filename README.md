# **Y**ou **O**nly **L**ook **A**t **C**oefficien**T**s

A simple, fully convolutional model for real-time instance segmentation
 - [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)
 - [YOLACT++: Better Real-time Instance Segmentation](https://arxiv.org/abs/1912.06218)

![Example 0](data/yolact_example_0.png)

# Installation

## Dependencies

This project depends on the following libraries:

* Python 3
* CUDA
* ZED SDK (Python API)
* Pytorch

## Getting Started

### ZED SDK Installation

Install the [ZED SDK](https://www.stereolabs.com/developers/release/) and the ZED [Python API](https://www.stereolabs.com/docs/getting-started/python-development/).

### Pytorch and python dependencies Installation

#### Using Conda (recommended)

The CUDA version must match the one used for the ZED SDK, in that case CUDA **10.0**.
A dedicated environment can be created to setup Pytorch, but don't forget to activate it, especially when installing MaskRCNN.

```bash
conda create --name pytorch1 -y
conda activate pytorch1
```

```bash
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
conda install --yes --file requirements.txt
```

#### Using Pip

```bash
pip3 install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt
```

For more information please refer to this page https://pytorch.org/get-started/locally/.

### YOLACT Installation

Compile deformable convolutional layers (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_1.0)) for Yolact++

```bash
cd external/DCNv2
python setup.py build develop
```

## Running YOLACT with the ZED camera

```bash
# Live mode
python zed_yolact.py --trained_model=weights/yolact_plus_resnet50_54_800000.pth

# SVO path
python zed_yolact.py --trained_model=weights/yolact_darknet53_54_800000.pth --svo_path="/path/to/svo"
```

# Models

YOLACT++ models:

| Image Size | Backbone      | FPS (Titan Xp) | mAP  | Weights                                                                                                              |  |
|:----------:|:-------------:|:----:|:----:|----------------------------------------------------------------------------------------------------------------------|--------|
| 550        | Resnet50-FPN  | 33.5 | 34.1 | [yolact_plus_resnet50_54_800000.pth](https://drive.google.com/file/d/1ZPu1YR2UzGHQD0o1rEqy-j5bmEm3lbyP/view?usp=sharing)  | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EcJAtMiEFlhAnVsDf00yWRIBUC4m8iE9NEEiV05XwtEoGw) |
| 550        | Resnet101-FPN | 27.3 | 34.6 | [yolact_plus_base_54_800000.pth](https://drive.google.com/file/d/15id0Qq5eqRbkD-N3ZjDZXdCvRyIaHpFB/view?usp=sharing) | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EVQ62sF0SrJPrl_68onyHF8BpG7c05A8PavV4a849sZgEA)

Older YOLACT models:

| Image Size | Backbone      | FPS (Titan Xp) | mAP  | Weights                                                                                                              |  |
|:----------:|:-------------:|:----:|:----:|----------------------------------------------------------------------------------------------------------------------|--------|
| 550        | Resnet50-FPN  | 42.5 | 28.2 | [yolact_resnet50_54_800000.pth](https://drive.google.com/file/d/1yp7ZbbDwvMiFJEq4ptVKTYTI2VeRDXl0/view?usp=sharing)  | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/EUVpxoSXaqNIlssoLKOEoCcB1m0RpzGq_Khp5n1VX3zcUw) |
| 550        | Darknet53-FPN | 40.0 | 28.7 | [yolact_darknet53_54_800000.pth](https://drive.google.com/file/d/1dukLrTzZQEuhzitGkHaGjphlmRJOjVnP/view?usp=sharing) | [Mirror](https://ucdavis365-my.sharepoint.com/:u:/g/personal/yongjaelee_ucdavis_edu/ERrao26c8llJn25dIyZPhwMBxUp2GdZTKIMUQA3t0djHLw)



To evalute the model, put the corresponding weights file in the `./weights` directory and run one of the following commands. The name of each config is everything before the numbers in the file name (e.g., `yolact_base` for `yolact_base_54_800000.pth`).

## YOLACT Original README

See the original [README](README_origin.md)