# GES: Generalized Exponential Splatting for Efficient Radiance Field Rendering
[arXiv](https://arxiv.org/abs/0000.00000) | [webpage](https://abdullahamdi.com/ges/)

<img src="docs/static/magic123.gif" width="800" />

[Abdullah Hamdi](https://abdullahamdi.com/) <sup>1</sup>, [Luke Melas-Kyriazi](https://lukemelas.github.io/) <sup>1</sup>, [Guocheng Qian](https://guochengqian.github.io/) <sup>2,4</sup>, [Jinjie Mai](https://cemse.kaust.edu.sa/people/person/jinjie-mai) <sup>2</sup>, [Ruoshi Liu](https://ruoshiliu.github.io/) <sup>3</sup>, [Carl Vondrick](https://www.cs.columbia.edu/~vondrick/) <sup>3</sup>, [Bernard Ghanem](https://www.bernardghanem.com/) <sup>2</sup>, [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/) <sup>1</sup>

<sup>1</sup> [King Abdullah University of Science and Technology (KAUST)](https://www.kaust.edu.sa/),
<sup>2</sup> [Snap Inc.](https://www.snap.com/),
<sup>3</sup> [Visual Geometry Group, University of Oxford](http://www.robots.ox.ac.uk/~vgg/)



The code is heavily based on 3D Gaussian Splatting for Real-Time Radiance Field Rendering (https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), thanks for the amazing authors.

## Overview

The codebase has 2 main components:
- A PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs
- A script to help you turn your own images into optimization-ready SfM data sets

The components have different requirements w.r.t. both hardware and software. They have been tested on Windows 10 and Ubuntu Linux 22.04. Instructions for setting up and running each of them are found in the sections below.

## Optimizer

The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)
- Please see FAQ for smaller VRAM configurations

### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used Visual Studio 2019 for Windows)
- CUDA SDK 11 for PyTorch extensions, install *after* Visual Studio (we used 11.8, **known issues with 11.6**)
- C++ Compiler and CUDA SDK must be compatible

### Setup

#### Local Setup

Our default, provided install method is based on Conda package and environment management:
```shell
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate ges
```
Please note that this process assumes that you have CUDA SDK **11** installed, not **12**. For modifications, see below.


### Running



For example, let's assume you download `tandt_db` dataset, please run following to reproduce Gaussian splatting results.
```
python train_gaussian.py -s ./tandt_db/tandt/train -m ./outputs/train --eval 
```

And run following to reproduce our GES results:

```
python train_ges.py -s ./tandt_db/tandt/train -m ./outputs/train --eval 
```

You can also run this example script for better control of the aurguments:

```
bash train_ges.sh

```

To reproduce all the results in our paper, prepare your datasets according to the script, and then run it:

```
bash example_full_eval.sh # for gaussian raw
bash ges_full_eval.sh # for our GES implementation
```
