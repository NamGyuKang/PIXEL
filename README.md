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
if you use CUDA 11.3, Pytorch 1.11, Python 3.9,
  - conda env create -f CUDA_11_3_Pytorch_1_11_Py_3_9.yml
or with CUDA 11.6, Pytorch 1.12, Python 3.8,
  - conda env create -f CUDA_11_6_Pytorch_1_12_Py_3_8.yml
  
-  conda activate pixel
```

PIXEL can also be run without CosineSampler, which is a speed-acceleration and memory-efficient interpolation function code using CUDA  
(3-order gradient available code).  
You can run the Python based code of the interpolation function which doesn't use CUDA with the '--cuda-off' command.  
With this option, you can calculate the high-order gradient (>= 4-order) by the PyTorch's Autograd.

## 2. Run
You can run PIXEL directly using the following code. It is the part of the command.txt file.
```
[burgers equation]
## Forward problem ##
# Pixel 96 multicell
python main.py --network base --pde burgers_1d --in-dim 2 --out-dim 1 --use-cell --n-cells 96 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.01 --hidden-dim 16 --num-layers 2 --max-iter 39001 --seed 200 --tag sample_tag
# Pixel 64 multicell
python main.py --network base --pde burgers_1d --in-dim 2 --out-dim 1 --use-cell --n-cells 64 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.01 --hidden-dim 16 --num-layers 2 --max-iter 39001 --seed 500 --tag sample_tag
# Pixel 16 multicell
python main.py --network base --pde burgers_1d --in-dim 2 --out-dim 1 --use-cell --n-cells 16 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.01 --hidden-dim 16 --num-layers 2 --max-iter 39001 --seed 500 --tag sample_tag

# PINN
python main.py --network base --pde burgers_1d --in-dim 2 --out-dim 1 --num-train 100000 --num-init 100000 --f-scale 0.01 --hidden-dim 40 --num-layers 9 --max-iter 39001 --seed 200 --tag sample_tag

## Inverse problem ##
# PIXEL
python main.py --network base --pde burgers_1d --in-dim 2 --out-dim 1 --use-cell --n-cells 192 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.0005 --hidden-dim 16 --num-layers 2 --max-iter 41 --problem inverse --seed 500 --tag sample_tag
# PINN
python main.py --network base --pde burgers_1d --in-dim 2 --out-dim 1 --num-train 100000 --num-init 100000 --f-scale 0.0005 --hidden-dim 40 --num-layers 9 --max-iter 41 --problem inverse --seed 500 --tag sample_tag


# 3D - Helmholtz
## Forward problem ##
# PIXEL
python main.py --network base --pde helmholtz_3d --a1 7.0 --a2 7.0 --a3 7.0 --in-dim 3 --out-dim 1 --use-cell --n-cells 16 --cell-dim 4 --cell-size 16 --interp cosine --num-train 400000 --num-init 400000 --num-test 100 --random-f --hidden-dim 16 --num-layers 2 --max-iter 11001 --f-scale 0.01 --seed 200 --tag sample_tag 
# PINN
python main.py --network base --pde helmholtz_3d --a1 7.0 --a2 7.0 --a3 7.0 --in-dim 3 --out-dim 1 --num-train 400000 --num-init 400000 --num-test 100 --random-f --hidden-dim 100 --num-layers 8 --max-iter 11001 --f-scale 0.01 --seed 200 --tag sample_tag 


# 3D - Navier-Stokes
## Inverse problem ##
# PIXEL
python main.py --network base --pde navier_stokes_3d --in-dim 3 --out-dim 3 --use-cell --n-cells 150 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 1.25 --hidden-dim 16 --num-layers 2 --max-iter 501 --seed 300 --problem inverse --tag sample_tag 
# PINN
python main.py --network base --pde navier_stokes_3d --in-dim 3 --out-dim 3 --num-train 100000 --num-init 100000 --num-test 250 --hidden-dim 20 --num-layers 10 --max-iter 1001 --f-scale 1.25 --seed 300 --problem inverse --tag sample_tag
```


# Citation
If you use this code for research, please consider citing:
```
@article{kang2023pixel,
title={PIXEL: Physics-Informed Cell Representations for Fast and Accurate PDE Solvers},
author={Kang, Namgyu and Lee, Byeonghyeon and Hong, Youngjoon and Yun, Seok-Bae and Park, Eunbyung},
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
year={2023}}
                    
```
