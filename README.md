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
You can run PIXEL directly using the following code.
```
[convection equation]
## Forward problem ##
# PIXEL (--ic-func sin_x), 96 multicell
python main.py --network base --pde convection_1d --ic-func sin_x --beta 30 --in-dim 2 --out-dim 1 --use-cell --n-cells 96 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.005 --hidden-dim 16 --num-layers 2 --max-iter 18001 --seed 500 --tag sample_tag
# PIXEL (--ic-func sin_x), 64 multicell
python main.py --network base --pde convection_1d --ic-func sin_x --beta 30 --in-dim 2 --out-dim 1 --use-cell --n-cells 64 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.005 --hidden-dim 16 --num-layers 2 --max-iter 18001 --seed 500 --tag sample_tag
# PIXEL (--ic-func sin_x), 16 multicell
python main.py --network base --pde convection_1d --ic-func sin_x --beta 30 --in-dim 2 --out-dim 1 --use-cell --n-cells 16 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.005 --hidden-dim 16 --num-layers 2 --max-iter 18001 --seed 500 --tag sample_tag

# PINN
python main.py --network base --pde convection_1d --ic-func sin_x --beta 30 --in-dim 2 --out-dim 1 --num-train 100000 --num-init 100000  --hidden-dim 50 --num-layers 4 --max-iter 18001 --f-scale 0.005 --seed 500 --tag sample_tag

## Inverse problem ##
# PIXEL
python main.py --network base --pde convection_1d --ic-func sin_x --beta 30 --in-dim 2 --out-dim 1 --use-cell --n-cells 192 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.005 --hidden-dim 16 --num-layers 2 --max-iter 51  --problem inverse --seed 500 --tag sample_tag
# PINN
python main.py --network base --pde convection_1d --ic-func sin_x --beta 30 --in-dim 2 --out-dim 1 --num-train 100000 --num-init 100000 --f-scale 0.005 --hidden-dim 50 --num-layers 4 --max-iter 51  --problem inverse --seed 500 --tag sample_tag





[reaction-diffusion equation]
## Forward problem ##
# PIXEL 96 multicell
python main.py --network base --pde rd_1d --nu 3.0 --rho 5.0 --ic-func gauss --in-dim 2 --out-dim 1 --use-cell --n-cells 96 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.01 --hidden-dim 16 --num-layers 2 --max-iter 10001 --seed 400  --tag sample_tag
# PIXEL 64 multicell
python main.py --network base --pde rd_1d --nu 3.0 --rho 5.0 --ic-func gauss --in-dim 2 --out-dim 1 --use-cell --n-cells 64 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.01 --hidden-dim 16 --num-layers 2 --max-iter 10001 --seed 500  --tag sample_tag
# PIXEL 16 multicell
python main.py --network base --pde rd_1d --nu 3.0 --rho 5.0 --ic-func gauss --in-dim 2 --out-dim 1 --use-cell --n-cells 16 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.01 --hidden-dim 16 --num-layers 2 --max-iter 10001 --seed 500  --tag sample_tag

# PINN
python main.py --network base --pde rd_1d --nu 3.0 --rho 5.0 --ic-func gauss --in-dim 2 --out-dim 1 --num-train 100000 --num-init 100000 --f-scale 0.01 --hidden-dim 50 --num-layers 4 --max-iter 10001 --seed 400 --tag sample_tag

## Inverse problem ##
# PIXEL 
python main.py --network base --pde rd_1d --nu 3.0 --rho 5.0 --ic-func gauss --in-dim 2 --out-dim 1 --use-cell --n-cells 192 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.005 --hidden-dim 16 --num-layers 2 --max-iter 201 --problem inverse --seed 500 --tag sample_tag
# PINN
python main.py --network base --pde rd_1d --nu 3.0 --rho 5.0 --ic-func gauss --in-dim 2 --out-dim 1 --num-train 100000 --num-init 100000 --f-scale 0.005 --hidden-dim 50 --num-layers 4 --max-iter 201 --problem inverse --seed 500 --tag sample_tag





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





[allen-cahn equation]
## Forward problem ##
# Pixel 96 multicell
python main.py --network base --pde ac_1d --in-dim 2 --out-dim 1 --use-cell --n-cells 96 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.1 --hidden-dim 16 --num-layers 2 --max-iter 500001 --seed 500 --tag sample_tag
# Pixel 64 multicell
python main.py --network base --pde ac_1d --in-dim 2 --out-dim 1 --use-cell --n-cells 64 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.1 --hidden-dim 16 --num-layers 2 --max-iter 500001 --seed 500 --tag sample_tag
# Pixel 16 multicell
python main.py --network base --pde ac_1d --in-dim 2 --out-dim 1 --use-cell --n-cells 16 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.1 --hidden-dim 16 --num-layers 2 --max-iter 500001 --seed 500 --tag sample_tag

# PINN
python main.py --network base --pde ac_1d --in-dim 2 --out-dim 1 --num-train 100000 --num-init 100000 --f-scale 0.1 --hidden-dim 128 --num-layers 7 --max-iter 500001  --seed 500 --tag sample_tag

## Inverse problem ##
# PIXEL
python main.py --network base --pde ac_1d --in-dim 2 --out-dim 1 --use-cell --n-cells 192 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --random-f --f-scale 0.1 --hidden-dim 16 --num-layers 2 --max-iter 41 --problem inverse --seed 500 --tag sample_tag
# PINN
python main.py --network base --pde ac_1d --in-dim 2 --out-dim 1 --num-train 100000 --num-init 100000 --f-scale 0.1 --hidden-dim 128 --num-layers 7 --max-iter 41  --problem inverse --seed 500 --tag sample_tag





[helmholtz_2d] f-scale [high-frequency forward : 0.00001, low-frequency forward : 0.0001, inverse : 0.00001]
## Forward problem ##

(high-frequency) 
# Pixel 96 multicell
python main.py --network base --pde helmholtz_2d --a1 10.0 --a2 10.0 --in-dim 2 --out-dim 1 --use-cell --n-cells 96 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --num-test 250 --random-f --hidden-dim 16 --num-layers 2 --max-iter 1001 --f-scale 0.00001 --seed 100 --tag sample_tag
# Pixel 64 multicell
python main.py --network base --pde helmholtz_2d --a1 10.0 --a2 10.0 --in-dim 2 --out-dim 1 --use-cell --n-cells 64 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --num-test 250 --random-f --hidden-dim 16 --num-layers 2 --max-iter 1001 --f-scale 0.00001 --seed 100 --tag sample_tag
# Pixel 16 multicell
python main.py --network base --pde helmholtz_2d --a1 10.0 --a2 10.0 --in-dim 2 --out-dim 1 --use-cell --n-cells 16 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --num-test 250 --random-f --hidden-dim 16 --num-layers 2 --max-iter 1001 --f-scale 0.00001 --seed 100 --tag sample_tag

(low-frequency)
# Pixel 96 multicell
python main.py --network base --pde helmholtz_2d --a1 4.0 --a2 1.0 --in-dim 2 --out-dim 1 --use-cell --n-cells 96 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --num-test 250 --random-f --hidden-dim 16 --num-layers 2 --max-iter 6001 --f-scale 0.0001 --seed 500 --tag sample_tag
# Pixel 64 multicell
python main.py --network base --pde helmholtz_2d --a1 4.0 --a2 1.0 --in-dim 2 --out-dim 1 --use-cell --n-cells 64 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --num-test 250 --random-f --hidden-dim 16 --num-layers 2 --max-iter 6001 --f-scale 0.0001 --seed 500 --tag sample_tag
# Pixel 16 multicell
python main.py --network base --pde helmholtz_2d --a1 4.0 --a2 1.0 --in-dim 2 --out-dim 1 --use-cell --n-cells 16 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --num-test 250 --random-f --hidden-dim 16 --num-layers 2 --max-iter 6001 --f-scale 0.0001 --seed 500 --tag sample_tag

#PINN - high-frequency helmholtz
python main.py --network base --pde helmholtz_2d --a1 10.0 --a2 10.0 --in-dim 2 --out-dim 1 --num-train 100000 --num-init 100000 --num-test 250 --hidden-dim 100 --num-layers 8 --max-iter 1001 --f-scale 0.00001 --seed 400 --tag sample_tag

#PINN - low-frequency helmholtz
python main.py --network base --pde helmholtz_2d --a1 4.0 --a2 1.0 --in-dim 2 --out-dim 1 --num-train 100000 --num-init 100000 --num-test 250 --hidden-dim 100 --num-layers 8 --max-iter 6001 --f-scale 0.0001 --seed 500 --tag sample_tag


## Inverse problem ##
# PIXEL
python main.py --network base --pde helmholtz_2d --a1 4.0 --a2 1.0 --in-dim 2 --out-dim 1 --use-cell --n-cells 16 --cell-dim 4 --cell-size 16 --interp cosine --num-train 100000 --num-init 100000 --num-test 250 --random-f --hidden-dim 16 --num-layers 2 --max-iter 350 --f-scale 0.00001 --problem inverse --seed 300 --tag sample_tag
# PINN
python main.py --network base --pde helmholtz_2d --a1 4.0 --a2 1.0 --in-dim 2 --out-dim 1 --num-train 100000 --num-init 100000 --num-test 250 --hidden-dim 100 --num-layers 8 --max-iter 350 --f-scale 0.00001 --problem inverse --seed 500 --tag sample_tag


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
If you use this code in your research, please consider citing:

```
@article{kang2023pixel,
title={PIXEL: Physics-Informed Cell Representations for Fast and Accurate PDE Solvers},
author={Kang, Namgyu and Lee, Byeonghyeon and Hong, Youngjoon and Yun, Seok-Bae and Park, Eunbyung},
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
year={2023}}
                    
```
