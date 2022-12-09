import numpy as np
import torch
import torch.optim as optim
import os
import sys
from matplotlib import rc
from tqdm import tqdm
from test_pde import *
from data_generator import *
from physics_informed_loss import *
from ground_truth import *

rc('text', usetex=False)

def pde_test(pde, t_test, x_test, u_test, lambda_1, net_u_2d, problem, it, loss_list, output_path, tag):
    if pde == 'burgers':
        Burgers_test(pde, t_test, x_test, u_test, lambda_1, net_u_2d, problem, it, loss_list, output_path, tag)
    elif pde == 'convection':
        Convection_test(pde, t_test, x_test, u_test, lambda_1, net_u_2d, problem, it, loss_list, output_path, tag)
    elif pde == 'reaction-diffusion':
        ReactionDiffusion_test(pde, t_test, x_test, u_test, lambda_1, net_u_2d, problem, it, loss_list, output_path, tag)
    elif pde == '2d_helmholtz':
        num_test = 250
        Helmholtz_2d_test(pde, t_test, x_test, u_test, lambda_1, net_u_2d, problem, it, loss_list, output_path, tag, num_test)

class PIXEL():
    def __init__(self, network, args, PDE):
        self.args = args
        # deep neural networks
        self.dnn = network
        # random sampling at every iteration
        self.random_f = args.random_f
        # number of points
        self.num_train = args.num_train
        self.num_test = args.num_test
        self.num_ic = args.num_init
        self.num_bc = args.num_init
        self.output_path = "results/figures/{}".format(args.tag)
        self.tag = args.tag
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.f_scale = args.f_scale
        self.u_scale = args.u_scale
        
        # TV regularization coefficient
        self.lamb = args.lamb
        
        
        self.use_cell = args.use_cell
        
        # optimizers: using the same settings
        self.optim = args.optim
        self.lr = args.lr
        self.max_iter = args.max_iter
        self.set_optimizer()
        
        self.iter = 0
        self.loss_list = []
        self.loss_b = 0
        self.loss_tv = 0

        self.exist_pde_source_term = False
        self.boundary_condition = False
        self.number_of_boundary = 0
        self.boundary_gradient_condition = False
        self.mixed_boundary_condition = False

        self.pde = PDE
        
        if self.pde == 'burgers':
            ''' PDE attribution '''
            self.exist_pde_source_term = False
            self.boundary_condition = False
            self.number_of_boundary = 0
            self.boundary_gradient_condition = False
            self.mixed_boundary_condition = False
            ''' load data '''
            self.t_train_f, self.x_train_f, self.t_train, self.x_train, self.u_train= generate_Burgers_train_data(self.num_train, self.num_ic, self.num_bc)
            ''' it will be used in forward's estimation L2 relative error function '''
            self.x_inverse_data, self.t_inverse_data, self.u_inverse_data, self.u_test, self.t_test, self.x_test = load_Burgers_ground_truth()
            ''' bergurs' PDE parameter, e.g. viscosity '''
            self.nu = 0.01/np.pi

        elif self.pde == 'convection':
            ''' PDE attribution '''
            self.exist_pde_source_term = False
            self.boundary_condition = True
            self.number_of_boundary = 2
            self.boundary_gradient_condition = False
            self.mixed_boundary_condition = False
            ''' PDE initial function '''
            if args.ic_func == 'sin_x':
                self.ic_func = lambda x: np.sin(x)
            elif args.ic_func == 'sin_4x':
                self.ic_func = lambda x: np.sin(4*x)
            else:
                raise NotImplementedError()
            ''' PDE parameter '''
            self.nu = 0.0
            self.beta = args.beta
            ''' load data '''
            self.t_train_f, self.x_train_f, self.t_train, self.x_train, self.u_train, self.t_bc1_train, self.x_bc1_train, self.t_bc2_train, self.x_bc2_train = generate_Convection_train_data(self.num_train, self.num_ic, self.num_bc, self.ic_func)
            self.t_test, self.x_test, self.u_test = generate_Convection_test_data(self.nu, self.beta, self.ic_func)
            if self.args.problem == 'inverse':
                self.t_inverse_data, self.x_inverse_data, self.u_inverse_data = generate_Convection_inverse_data(self.x_test, self.t_test, self.nu, self.beta, self.ic_func)
        
        elif self.pde == 'reaction-diffusion':
            ''' PDE attribution '''
            self.exist_pde_source_term = False
            self.boundary_condition = True
            self.number_of_boundary = 2
            self.boundary_gradient_condition = True
            self.mixed_boundary_condition = False
            ''' PDE initial function '''
            if args.ic_func == 'sin_x':
                self.ic_func = lambda x: np.sin(x)
            elif args.ic_func == 'sin_4x':
                self.ic_func = lambda x: np.sin(4*x)
            elif args.ic_func == 'gauss':
                self.ic_func = lambda x: np.exp(-np.power((x - np.pi)/(np.pi/4), 2.)/2.)
            else:
                raise NotImplementedError()
            ''' PDE parameter '''
            self.rho = args.rho
            self.nu = args.nu
            self.beta = args.beta
            ''' load data '''
            self.t_test, self.x_test, self.u_test = generate_Reaction_diffusion_test_data(self.nu, self.rho, self.ic_func)
            self.t_train_f, self.x_train_f, self.t_train, self.x_train, self.u_train, self.t_bc1_train, self.x_bc1_train, self.t_bc2_train, self.x_bc2_train = generate_Reaction_diffusion_train_data(self.num_train, self.num_ic, self.num_bc, self.ic_func)
            if self.args.problem == 'inverse':
                self.t_flat, self.x_flat, self.t_inverse_data, self.x_inverse_data, self.u_inverse_data = generate_Reaction_diffusion_inverse_data(self.x_test, self.t_test, self.nu, self.rho, self.ic_func)
        
        
        elif self.pde == 'allen-cahn':
            ''' PDE attribution '''
            self.exist_pde_source_term = False
            self.boundary_condition = True
            self.number_of_boundary = 2
            self.boundary_gradient_condition = True
            self.mixed_boundary_condition = True
            ''' PDE parameter '''
            self.nu = 0.0001
            ''' load data '''
            self.t_train_f, self.x_train_f, self.t_train, self.x_train, self.u_train, self.t_bc1_train, self.x_bc1_train, self.t_bc2_train, self.x_bc2_train = generate_Allen_cahn_train_data(self.num_train, self.num_ic, self.num_bc)
            ''' it will be used in forward's estimation L2 relative error function '''
            self.t_inverse_data, self.x_inverse_data, self.u_inverse_data, self.t_test, self.x_test, self.u_test, self.test_t_flat, self.test_x_flat, self.T, self.X, self.u_sol = load_AllenCahn_ground_truth()
            ''' For weighted loss '''
            self.b_scale = args.b_scale
            self.u_scale = args.u_scale
            self.use_b_loss = args.use_b_loss

        elif self.pde == '2d_helmholtz':
            ''' PDE attribution '''
            self.exist_pde_source_term = True
            self.boundary_condition = True
            self.number_of_boundary = 1
            self.boundary_gradient_condition = False
            self.mixed_boundary_condition = False
            ''' PDE parameter '''
            self.a1 = args.a1
            self.a2 = args.a2
            self.coefficient = args.lambda_1
            ''' load data '''
            self.t_test, self.x_test, self.u_test = generate_Helmholtz_2d_test_data(self.num_test, self.a1, self.a2)
            self.t_train_f, self.x_train_f, self.u_train_f, self.t_train, self.x_train, self.u_train = generate_Helmholtz_2d_train_data(self.num_train, self.num_bc, self.a1, self.a2, self.coefficient)
            if self.args.problem == 'inverse':
                self.t_inverse_data, self.x_inverse_data, self.u_inverse_data = generate_Helmholtz_2d_inverse_data(self.a1, self.a2)

        elif self.pde == '3d_helmholtz':
            ''' PDE attribution '''
            self.exist_pde_source_term = True
            self.boundary_condition = False
            self.boundary_gradient_condition = False
            self.mixed_boundary_condition = False
            ''' PDE parameter '''
            self.a1 = args.a1
            self.a2 = args.a2
            self.a3 = args.a3
            self.coefficient = args.lambda_1
            ''' load data '''
            self.x_test, self.y_test, self.z_test, self.u_test = generate_Helmholtz_3d_test_data(self.num_test, self.a1, self.a2, self.a3)
            self.x_train_f, self.y_train_f, self.z_train_f, self.u_train_f, self.x_train, self.y_train, self.z_train, self.u_train = generate_Helmholtz_3d_train_data(self.num_train, self.num_bc, self.a1, self.a2, self.a3, self.coefficient)
            if self.args.problem == 'inverse':
                self.x_inverse_data, self.t_inverse_data, self.z_inverse_data, self.u_inverse_data =  generate_Helmholtz_3d_inverse_data(self.a1, self.a2, self.a3)
       
        elif self.pde == '3d_navier_stokes':
            self.t_train_f, self.x_train_f, self.y_train_f, self.t_train, self.x_train, self.y_train, self.txt_u, self.txt_v = generate_Navier_Stokes_inverse_data(self.num_train)
        

    def set_optimizer(self):
        if self.optim == 'lbfgs':
            self.optimizer = optim.LBFGS(
                self.dnn.parameters(), 
                lr=self.lr, 
                #max_iter=self.max_iter, 
                #max_eval=50000,
                #history_size=50,
                #tolerance_grad=1e-6,
                #tolerance_change=1.0 * np.finfo(float).eps,
                line_search_fn="strong_wolfe"       # can be "strong_wolfe"
            )
        elif self.optim == 'adam':
            self.optimizer = optim.Adam(self.dnn.parameters(), lr = self.lr)
        else:
            raise NotImplementedError()

    def net_u_2d(self, t, x, pde):
        if self.use_cell:
            ''' normalize to [-1, 1] '''
            if pde == 'burgers' or pde == 'allen-cahn':
                t = t*2-1
            elif pde == 'convection' or pde == 'reaction-diffusion':
                t = t*2-1
                x = (x-np.pi)/np.pi
            x = torch.cat([t, x], dim=-1).unsqueeze(0).unsqueeze(0)
        else:
            x = torch.cat([t, x], dim=1)
        u = self.dnn(x)
        
        return u

    def net_u_3d_helmholtz(self, x, y, z):
        if self.use_cell:
            ''' normalize to [-1, 1] '''
            if self.args.cuda_off:
                x = torch.cat([x, y, z], dim=-1).unsqueeze(0).unsqueeze(0)
            else:
                x = torch.cat([x, y, z], dim=-1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            
        else:
            x = torch.cat([x, y, z], dim=-1).view(-1, 3)
        u = self.dnn(x)
        return u
    
    def net_u_3d_navier_stokes(self, t, x, y):
        t = t*0.1-1
        x = (x-1)*(2/7)-1
        y = y*0.5
        if self.args.use_cell :
            input = torch.cat([t, x, y], dim= -1).unsqueeze(0).unsqueeze(0)
        else:
            input = torch.cat([t, x, y], dim= -1)
        
        uvp = self.dnn(input)

        return uvp

    def net_f_2d(self, t, x, pde):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u_2d(t, x, pde)
        
        if self.args.problem == 'forward':
            lambda_1 = None
        elif self.args.problem == 'inverse':
            lambda_1 = self.dnn.lambda_1
            
        if pde == 'burgers':
            f = Burgers(u, t, x, self.nu, lambda_1, self.args.problem)
        elif pde == 'convection':
            f = Convection(u, t, x, self.beta, lambda_1, self.args.problem)
        elif pde == 'reaction-diffusion':
            f = ReactionDiffusion(u, t, x, self.nu, self.rho, lambda_1, self.args.problem)
        elif pde == 'allen-cahn':
            f = AllenCahn(u, t, x, self.nu, lambda_1, self.args.problem)
        elif pde == '2d_helmholtz':
            f = Helmholtz_2d(u, t, x, self.coefficient, lambda_1, self.args.problem)

        return f
    
    def net_f_3d_helmholtz(self, x, y, z):
        """ The pytorch autograd version of calculating residual """
        u = self.net_u_3d_helmholtz(x, y, z)

        if self.args.problem == 'forward':
            lambda_1 = None
        elif self.args.problem == 'inverse':
            lambda_1 = self.dnn.lambda_1

        f = Helmholtz_3d(u, x, y, z, self.coefficient, lambda_1, self.args.problem)
        
        return f 

    def net_f_3d_navier_stokes(self, t, x, y):
        """ The pytorch autograd version of calculating residual """
        uvp = self.net_u_3d_navier_stokes(t, x, y)

        f = Navier_Stokes_3d(uvp, t, x, y, self.dnn.lambda_1, self.dnn.lambda_2)
        
        return f 

    def tv(self):
        return self.dnn.tv()


    def loss_func_2d(self):
        self.optimizer.zero_grad()
        f_pred = self.net_f_2d(self.t_train_f, self.x_train_f, self.pde)
        loss_f = torch.mean(f_pred ** 2)
        if self.exist_pde_source_term:
            loss_f = torch.mean((self.u_train_f - f_pred)**2)

        if self.args.problem == 'forward':
            ''' initial condition '''
            u_pred = self.net_u_2d(self.t_train, self.x_train, self.pde)
            loss_u = torch.mean((self.u_train - u_pred) ** 2)
            
            if self.boundary_condition:
                ''' boundary condition '''
                if self.number_of_boundary == 1:
                    u_bc1_pred = self.net_u_2d(self.t_train, self.x_train, self.pde)
                    loss_b = torch.mean((self.u_train - u_bc1_pred) ** 2)
                elif self.number_of_boundary ==2:            
                    u_bc1_pred = self.net_u_2d(self.t_bc1_train, self.x_bc1_train, self.pde)
                    u_bc2_pred = self.net_u_2d(self.t_bc2_train, self.x_bc2_train, self.pde)
                    loss_b = torch.mean((u_bc1_pred - u_bc2_pred) ** 2)

                if self.boundary_gradient_condition:
                    ''' boundary gradient condition '''
                    u_bc1_x = torch.autograd.grad(u_bc1_pred, self.x_bc1_train, grad_outputs=torch.ones_like(u_bc1_pred), retain_graph=True, create_graph=True)[0]
                    u_bc2_x = torch.autograd.grad(u_bc2_pred, self.x_bc2_train, grad_outputs=torch.ones_like(u_bc2_pred), retain_graph=True, create_graph=True)[0]
                    loss_b = torch.mean((u_bc1_x - u_bc2_x) ** 2)
                    
                    if self.mixed_boundary_condition:
                        ''' Summation (boundary condition, 1st order boundary gradient condition) '''
                        if self.use_b_loss:
                            loss_b = torch.mean((u_bc1_pred - u_bc2_pred)**2) + torch.mean((u_bc1_x - u_bc2_x)**2)
                        else:
                            loss_b = torch.zeros(1)

            
            if self.pde == 'burgers' or self.pde == '2d_helmholtz':
                scaled_loss = loss_u + self.f_scale*loss_f + self.lamb*self.loss_tv
            elif self.pde == 'convection':
                scaled_loss = loss_u + loss_b + self.f_scale*loss_f + self.lamb*self.loss_tv
            elif self.pde == 'reaction-diffusion':
                scaled_loss = loss_u + self.f_scale*(loss_b + loss_f) + self.lamb*self.loss_tv
            elif self.pde == 'allen-cahn':
                scaled_loss = loss_u + self.b_scale*loss_b + self.f_scale*loss_f + self.lamb*self.loss_tv

        elif self.args.problem == 'inverse':
            u_data_pred = self.net_u_2d(self.t_inverse_data, self.x_inverse_data, self.pde)
            if self.pde == 'convection' or self.pde == 'reaction-diffusion':
                u_data_pred = u_data_pred.view(self.u_inverse_data.shape)
            loss_data = torch.mean((self.u_inverse_data - u_data_pred)**2)
            scaled_loss = loss_data + self.f_scale*loss_f
        
        scaled_loss.backward()
        # for loggin purpose

        self.loss_f = loss_f.item()
        if self.args.problem == 'forward':
            if self.pde != 'burgers' and self.pde != '2d_helmholtz':
                self.loss_b = loss_b.item()
            self.loss_u = loss_u.item()
        elif self.args.problem == 'inverse':
            self.loss_data = loss_data.item()

        return scaled_loss

    def loss_func_3d(self):
        self.optimizer.zero_grad()
        
        if self.args.problem == 'forward':
            f_pred = self.net_f_3d_helmholtz(self.x_train_f, self.y_train_f, self.z_train_f)
            if self.exist_pde_source_term:
                loss_f = torch.mean((self.u_train_f - f_pred) ** 2)
            else:
                loss_f = torch.mean(f_pred ** 2)

            ''' initial condition '''
            u_pred = self.net_u_3d_helmholtz(self.x_train, self.y_train, self.z_train)
            loss_u = torch.mean((self.u_train - u_pred) ** 2)
            scaled_loss = self.u_scale *loss_u + self.f_scale*loss_f
            scaled_loss.backward()
            self.loss_f = loss_f.item()

        elif self.args.problem == 'inverse':
            uvp = self.net_u_3d_navier_stokes(self.t_train, self.x_train, self.y_train)
            u = uvp[:, 0:1]
            v = uvp[:, 1:2]
            f_u_pred, f_v_pred = self.net_f_3d_navier_stokes(self.t_train_f, self.x_train_f, self.y_train_f)
            loss_u = torch.mean((self.txt_u - u.view(self.txt_u.shape)) ** 2)
            loss_v = torch.mean((self.txt_v - v.view(self.txt_v.shape)) ** 2)
            loss_f_u = torch.mean(f_u_pred ** 2)
            loss_f_v = torch.mean(f_v_pred ** 2)
            scaled_loss = self.u_scale*(loss_u+loss_v) + self.f_scale*(loss_f_u+loss_f_v) 
            scaled_loss.backward(retain_graph =True)
            loss = loss_u+loss_v+loss_f_u+loss_f_v

        # for loggin purpose
        if self.args.problem == 'forward':
            self.loss_u = loss_u.item()
        elif self.args.problem == 'inverse':
            self.loss_u = loss_u.item()
            self.loss_v = loss_v.item()
            self.loss_f_u = loss_f_u.item()
            self.loss_f_v = loss_f_v.item()
            self.loss = loss.item()
        return scaled_loss



    def train(self):
        # Backward and optimize
        for it in tqdm(range(self.max_iter)):
            self.dnn.train()
            self.it = it
        
            if self.optim == 'lbfgs':
                if self.pde[:2] == '3d':
                    self.optimizer.step(self.loss_func_3d)
                else:
                    self.optimizer.step(self.loss_func_2d)
                if self.args.problem == 'forward':
                    if it % 15 ==0:
                        print('Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_b: %.5e, Loss_f: %.5e, Loss_tv: %.5e'%(
                            it+1, self.loss_u+self.loss_b+self.loss_f, self.loss_u, self.loss_b, self.loss_f, self.loss_tv))
                elif self.args.problem == 'inverse':
                    if it % 1 ==0:
                        if self.pde[:2] != '3d':
                            print('Iter %d, lambda: %.5e, Loss: %.5e, Loss_data: %.5e, Loss_f: %.5e, Loss_tv: %.5e'%(
                                it+1, self.dnn.lambda_1, self.loss_data+self.loss_f, self.loss_data, self.loss_f, self.loss_tv))
                        else:
                            print('Iter %d, lda1: %.5e, lda2: %.5e, Loss: %.5e, Loss_u: %.5e, Loss_v: %.5e, Loss_f_u: %.5e, Loss_f_v: %.5e'%(
                                it+1, self.dnn.lambda_1.item(), self.dnn.lambda_2.item(), self.loss, self.loss_u, self.loss_v, self.loss_f_u, self.loss_f_v))

                sys.stdout.flush()
            else:
                self.optimizer.zero_grad()
                ''' Adam optimizer for 2d PDEs '''
                u_pred = self.net_u_2d(self.t_train, self.x_train, self.pde)
                f_pred = self.net_f_2d(self.t_train_f, self.x_train_f, self.pde)
                loss_u = torch.mean((self.u_train - u_pred) ** 2)
                loss_f = torch.mean(f_pred ** 2)
                loss = loss_u + loss_f
                if self.boundary_condition:
                    ''' boundary condition '''
                    u_bc1_pred = self.net_u_2d(self.t_bc1_train, self.x_bc1_train, self.pde)
                    u_bc2_pred = self.net_u_2d(self.t_bc2_train, self.x_bc2_train, self.pde)
                    loss_b = torch.mean((u_bc1_pred - u_bc2_pred) ** 2)

                    if self.boundary_gradient_condition:
                        ''' boundary gradient condition '''
                        u_bc1_x = torch.autograd.grad(u_bc1_pred, self.x_bc1_train, grad_outputs=torch.ones_like(u_bc1_pred), retain_graph=True, create_graph=True)[0]
                        u_bc2_x = torch.autograd.grad(u_bc2_pred, self.x_bc2_train, grad_outputs=torch.ones_like(u_bc2_pred), retain_graph=True, create_graph=True)[0]
                        loss_b = torch.mean((u_bc1_x - u_bc2_x) ** 2)
                        
                        if self.mixed_boundary_condition:
                            ''' Summation (boundary condition, 1st order boundary gradient condition) '''
                            if self.use_b_loss:
                                loss_b = torch.mean((u_bc1_pred - u_bc2_pred)**2) + torch.mean((u_bc1_x - u_bc2_x)**2)
                            else:
                                loss_b = torch.zeros(1)
                    loss += loss_b
                    self.loss_b = loss_b.item()
                loss.backward()
                self.optimizer.step()
                if it % 100 == 0:
                    print('Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (
                            it, loss.item(), loss_u.item(), loss_f.item()))
                    sys.stdout.flush()

            if it % 1== 0:
                self.test(it, self.pde)

            self.iter += 1

            # Every interation, we randomly sample collocation points
            if self.random_f:
                if self.pde == 'burgers':
                    self.t_train_f, self.x_train_f, self.t_train, self.x_train, self.u_train = generate_Burgers_train_data(self.num_train, self.num_ic, self.num_bc)
                elif self.pde == 'convection':
                    self.t_train_f, self.x_train_f, self.t_train, self.x_train, self.u_train, self.t_bc1_train, self.x_bc1_train, self.t_bc2_train, self.x_bc2_train = generate_Convection_train_data(self.num_train, self.num_ic, self.num_bc, self.ic_func)
                elif self.pde == 'reaction-diffusion':
                    self.t_train_f, self.x_train_f, self.t_train, self.x_train, self.u_train, self.t_bc1_train, self.x_bc1_train, self.t_bc2_train, self.x_bc2_train = generate_Reaction_diffusion_train_data(self.num_train, self.num_ic, self.num_bc, self.ic_func)
                elif self.pde == 'allen-cahn':
                    self.t_train_f, self.x_train_f, self.t_train, self.x_train, self.u_train, self.t_bc1_train, self.x_bc1_train, self.t_bc2_train, self.x_bc2_train = generate_Allen_cahn_train_data(self.num_train, self.num_ic, self.num_bc)
                elif self.pde == '2d_helmholtz':
                    self.t_train_f, self.x_train_f, self.u_train_f, self.t_train, self.x_train, self.u_train = generate_Helmholtz_2d_train_data(self.num_train, self.num_bc, self.a1, self.a2, self.coefficient)
                elif self.pde == '3d_helmholtz':
                    self.x_train_f, self.y_train_f, self.z_train_f, self.u_train_f, self.x_train, self.y_train, self.z_train, self.u_train = generate_Helmholtz_3d_train_data(self.num_train, self.num_bc, self.a1, self.a2, self.a3, self.coefficient)
                elif self.pde == '3d_naver_stokes':
                     self.t_train_f, self.x_train_f, self.y_train_f, self.t_train, self.x_train, self.y_train, self.txt_u, self.txt_v = generate_Navier_Stokes_inverse_data() 





    def test(self, it, pde):
        self.dnn.eval()
        if self.args.problem == 'forward':
            lambda_1 = None
        elif self.args.problem == 'inverse':
            lambda_1 = self.dnn.lambda_1

        if self.pde == 'allen-cahn':
            AllenCahn_test(pde, self.test_t_flat, self.test_x_flat, self.T, self.X, self.u_test, self.u_sol, lambda_1, self.net_u_2d, self.args.problem, it, self.loss_list, self.output_path, self.tag)
        elif self.pde == '3d_helmholtz':
            Helmholtz_3d_test(pde, self.x_test, self.y_test, self.z_test, self.u_test, lambda_1, self.net_u_3d_helmholtz, self.args.problem, it, self.loss_list, self.output_path, self.tag, self.num_test)
        elif self.pde == '3d_navier_stokes':
            Navier_Stokes_3d_test(self.dnn.lambda_1, self.dnn.lambda_2, self.net_f_3d_navier_stokes, self.net_u_3d_navier_stokes, self.t_train_f, self.x_train_f, self.y_train_f, it, self.loss_list, self.output_path, self.tag, self.num_test, self.num_train)
        else:
            pde_test(pde, self.t_test, self.x_test, self.u_test, lambda_1, self.net_u_2d, self.args.problem, it, self.loss_list, self.output_path, self.tag)
        