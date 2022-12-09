import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from utils.grid_sample import *

from cosine_sampler_2d import CosineSampler2d
from cosine_sampler_3d import CosineSampler3d

class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        self.args = args
        # parameters
        self.num_layers = args.num_layers
        
        # using cell
        self.use_cell = args.use_cell
        self.n_cells = args.n_cells
        self.cell_dim = args.cell_dim
        self.cell_size_t = args.cell_size_t
        self.cell_size_x = args.cell_size_x
        self.cell_size_y = args.cell_size_y
        self.cell_size_t_max = args.cell_size_t_max
        self.cell_size_x_max = args.cell_size_x_max
        self.cell_size_y_max = args.cell_size_y_max
        self.cell_factor = args.cell_factor
        self.interp = args.interp
        self.problem = args.problem
        
        
        if self.problem == 'inverse':
            self.lambda_1 = torch.nn.Parameter(torch.zeros(1))
            if self.args.pde == 'navier_stokes_3d':
                self.lambda_2 = torch.nn.Parameter(torch.zeros(1))
        # Network dimension
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.in_dim = args.in_dim

        ''' see the Section "Neural network and Grid representations" in the paper.
                    we made parameterized cells and did an initialization. '''
        if self.use_cell:
            if self.in_dim == 2:
                '''see the Section "Multigrid representations" in the paper. 
                        the first dimension of cells (self.n_cells) is the Multigrid dimension. '''
                self.cells = torch.nn.Parameter(torch.rand(self.n_cells, self.cell_dim, self.cell_size_t, self.cell_size_x))
                self.cells.data.uniform_(-1e-5,1e-5)
                self.cells.requires_grad = True
            elif self.in_dim == 3:
                self.cells = torch.nn.Parameter(torch.rand(self.n_cells, self.cell_dim, self.cell_size_t, self.cell_size_x, self.cell_size_y))
                self.cells.data.uniform_(-1e-5,1e-5)
                self.cells.requires_grad = True
        
        if args.activation=='relu':
            self.activation_fn = nn.ReLU()
        elif args.activation=='leaky_relu':
            self.activation_fn = nn.LeakyReLU()
        elif args.activation=='sigmoid':
            self.activation_fn = nn.Sigmoid()
        elif args.activation=='softplus':
            self.activation_fn = nn.Softplus()
        elif args.activation=='tanh':
            self.activation_fn = nn.Tanh()
        elif args.activation=='gelu':
            self.activation_fn = nn.GELU()
        elif args.activation =='logsigmoid':
            self.activation_fn = nn.LogSigmoid()
        elif args.activation =='hardsigmoid':
            self.activation_fn = nn.Hardsigmoid()
        elif args.activation =='elu':
            self.activation_fn = nn.ELU()
        elif args.activation =='celu':
            self.activation_fn = nn.CELU()            
        elif args.activation =='selu':
            self.activation_fn = nn.SELU() 
        elif args.activation =='silu':
            self.activation_fn = nn.SiLU()     
        else:
            raise NotImplementedError
      
        if self.num_layers==0:
            return
        
        ''' see the Section "Neural network and Grid Representations" in the paper.
                    we built the Neural network. '''
        self.net = []
        input_dim = self.cell_dim if self.use_cell else self.in_dim
        if self.num_layers < 2:
            self.net.append(self.activation_fn)
            self.net.append(torch.nn.Linear(input_dim, self.out_dim))
        else:
            self.net.append(torch.nn.Linear(input_dim, self.hidden_dim))
            self.net.append(self.activation_fn)
            for i in range(self.num_layers-2): 
                self.net.append(torch.nn.Linear(self.hidden_dim, self.hidden_dim))
                self.net.append(self.activation_fn)
            self.net.append(torch.nn.Linear(self.hidden_dim, self.out_dim))
        
        # deploy layers
        self.net = nn.Sequential(*self.net)


    ''' If you want to grow-up size of cells, using function "grow". It is not used in current experiments.'''
    def grow(self, scale=2.0):
        if self.cell_factor:
            t_len = self.cell_size_t.shape[-1]
            x_len = self.cell_size_x.shape[-1]
        else:
            t_len = self.cells.shape[-2]
            x_len = self.cells.shape[-1]
        scale_t = scale if t_len < self.cell_size_t_max else 1.0
        scale_x = scale if x_len < self.cell_size_x_max else 1.0

        if scale_t > 1.0 or scale_x > 1.0:
            new_cell = F.interpolate(self.cells, scale_factor=(scale_t, scale_x), mode='bicubic', align_corners=True)
            self.cells = torch.nn.Parameter(new_cell)
            print('Cell grow: {}'.format(self.cells.size()))
            sys.stdout.flush()
            return True
        return False

    def forward(self, x):
        if self.use_cell:
            if self.cell_factor:
                cells = torch.matmul(self.cells_t.view(self.n_cells,self.cell_dim,self.cell_size_t,1),
                        self.cells_x.view(self.n_cells,self.cell_dim,1,self.cell_size_x))
                cells = cells.sum(0).unsqueeze(0)
                feats = grid_sample_2d(cells, x, step=self.interp, offset=False)
            else:
                if self.in_dim==2:
                    x = x.repeat([self.cells.shape[0],1,1,1])
                    '''1) see the Section "Mesh-agnostic representations through interpolation" in the paper. 
                            "grid_sample_2d" function is an interpolation function
                    2) also, see the Section "Multigrid representations". in the paper.
                            we can shift the input coordinates by using "offset=True". '''
                    if self.args.cuda_off:
                        feats = grid_sample_2d(self.cells, x, step=self.interp, offset=True)
                    else:
                        feats = CosineSampler2d.apply(self.cells, x, 'zeros', True, 'cosine', True)
                elif self.in_dim==3:
                    if self.args.cuda_off:
                        x = x.repeat([self.cells.shape[0],1,1,1])
                        feats = grid_sample_3d(self.cells, x, step=self.interp, offset=True)
                    else:
                        x = x.repeat([self.cells.shape[0],1,1,1,1])
                        feats = CosineSampler3d.apply(self.cells, x, 'zeros', True, 'cosine', True)
            # summation of multigrid dimension.
            x = feats.sum(0).view(self.cell_dim,-1).t()
            ''' now first dimension is 'cell_dim' which is represent by 'c' in the section "Multigrid representations" in the paper.
                        cell_dim is the input dimension of the neural network. '''
        if self.num_layers > 0:
            out = self.net(x)        
        else:
            '''if you don't use the neural network, then below code will be executed.
                        However, all experiments of this paper used neural network.'''
            x = feats.mean(0)
            out = x.mean(0).squeeze().view(-1, 1)
            
        return out
