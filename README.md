## PIXEL: Physics-Informed Cell Representations for Fast and Accurate PDE Solvers
### AAAI 2023 (Accept)
### DLDE-II, NeurIPS 2022 Workshop (Spotlight)
[[Project page]](https://namgyukang.github.io/PIXEL/) [[Paper-Arxiv]](https://arxiv.org/abs/2207.12800) [[Paper-Workshop]](https://openreview.net/forum?id=t49TL3qzma)

## Quick Start

## 1. Installation

### Clone PIXEL repo

```
git clone https://github.com/NamGyuKang/PIXEL.git
cd PIXEL
```

### Create environment

#### We implemented the 2D, and 3D customized CUDA kernel of the triple backward grid sampler that supports cosine, linear, and smoothstep kernel [(Thomas Müller)](https://nvlabs.github.io/instant-ngp/) and third-order gradients $u_{xxc}, u_{yyc}$ with second-order gradients [(Tymoteusz Bleja)](https://github.com/tymoteuszb/smooth-sampler.git). As a result, the runtime and the memory requirement were significantly reduced. You can find our customized CUDA kernel code at https://github.com/NamGyuKang/CosineSampler.

The code is tested with Python (3.8, 3.9) and PyTorch (1.11, 11.2) with CUDA (>=11.3). 
You can create an anaconda environment with those requirements by running:

```
conda env create -f pixel_environment.yml
conda activate pixel
```
You need to install CosineSampler at https://github.com/NamGyuKang/CosineSampler.

## 2. Run

You can directly run PIXEL with command.txt contents.

# Citation
If you use this code in your research, please consider citing:

```
@article{kang2023pixel,
title={PIXEL: Physics-Informed Cell Representations for Fast and Accurate PDE Solvers},
author={Kang, Namgyu and Lee, Byeonghyeon and Hong, Youngjoon and Yun, Seok-Bae and Park, Eunbyung},
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
year={2023}}
                    
```
