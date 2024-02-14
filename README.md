# GES: Generalized Exponential Splatting for Efficient Radiance Field Rendering
[arXiv](https://arxiv.org/abs/0000.00000) | [webpage](https://abdullahamdi.com/ges/)

<img src="assets/teaser.png" width="300" />

[Abdullah Hamdi](https://abdullahamdi.com/) <sup>1</sup>, [Luke Melas-Kyriazi](https://lukemelas.github.io/) <sup>1</sup>, [Guocheng Qian](https://guochengqian.github.io/) <sup>2,4</sup>, [Jinjie Mai](https://cemse.kaust.edu.sa/people/person/jinjie-mai) <sup>2</sup>, [Ruoshi Liu](https://ruoshiliu.github.io/) <sup>3</sup>, [Carl Vondrick](https://www.cs.columbia.edu/~vondrick/) <sup>3</sup>, [Bernard Ghanem](https://www.bernardghanem.com/) <sup>2</sup>, [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/) <sup>1</sup>

<sup>1</sup> [Visual Geometry Group, University of Oxford](http://www.robots.ox.ac.uk/~vgg/)
<sup>2</sup> [KAUST](https://www.kaust.edu.sa/),
<sup>3</sup> [Columbia University](https://www.columbia.edu/),
<sup>4</sup> [Snap Inc.](https://www.snap.com/),


## Overview

We provide a PyTorch implementation of our Generalized Exponential Splatting (GES) method, as well as the Gaussian Splatting method for comparison. We also provide the code to reproduce the results in our paper. The code is heavily based on [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). 

### Hardware Requirements

- CUDA-ready GPU with Compute Capability 7.0+
- 24 GB VRAM (to train to paper evaluation quality)

### Software Requirements
- Conda (recommended for easy setup)
- C++ Compiler for PyTorch extensions (we used Visual Studio 2019 for Windows)
- CUDA SDK 11 for PyTorch extensions, install *after* Visual Studio (we used 11.8, **known issues with 11.6**)
- C++ Compiler and CUDA SDK must be compatible

### Setup

Our default, provided install method is based on Conda package and environment management:
```shell
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate ges
```
Please note that this process assumes that you have CUDA SDK **11** installed, not **12**. For modifications, see below.


## Running
Download the datasets from the [original repository](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), and place them in the `tandt_db` and `nerf_360` directories.


For example, let's assume you download `tandt_db` dataset, please run following to reproduce Gaussian splatting results.
```
python train_gaussian.py -s ./tandt_db/tandt/train -m ./outputs/train --eval 
```

And run following to reproduce our GES results:

```
python train_ges.py -s ./tandt_db/tandt/train -m ./outputs/train --eval 
```



To reproduce all the results in our paper, prepare your datasets according to the script, and then run it:

```
bash ges_full_eval.sh # for our GES implementation
```

## Numerical Simulation for Generlized Exponential Function (GEF)
Check the notebook `simulation.ipynb` for the numerical simulation of the Generalized Exponential Function (GEF), that GES is based upon.

## Cite
If you find our work useful in your research, please consider citing:

```bibtex

```
