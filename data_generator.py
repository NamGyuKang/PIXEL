import torch
import numpy as np
from ground_truth import reaction, diffusion, helmholtz_2d_exact_u, helmholtz_2d_source_term, helmholtz_3d_exact_u, helmholtz_3d_source_term

''' Contents : 1. Generate Train data
               2. Generate Test data    '''

''' 1. Generate Train data '''
def generate_Burgers_train_data(num_train, num_ic, num_bc):
    t = np.random.rand(num_train, 1)
    x = np.random.rand(num_train, 1)
    ''' x -> [-1, 1], t -> [0, 1] '''
    x = 2 * x - 1 
    t_train_f = torch.tensor(t, requires_grad= True).float()
    x_train_f = torch.tensor(x, requires_grad= True).float()

    # create IC
    t_ic = np.zeros((num_ic, 1))                # t_ic =  0
    x_ic = 2 * np.random.rand(num_ic, 1) - 1  # x_ic = -1 ~ +1

    # create BC
    t_bc = np.random.rand(num_bc, 1) # t_bc =  0 ~ +1
    x_bc = np.random.rand(num_bc, 1) # x_bc = -1 or +1
    x_bc = 2 * np.round(x_bc) - 1    

    t_train = torch.tensor(np.concatenate((t_ic, t_bc)), requires_grad= True).float()
    x_train = torch.tensor(np.concatenate((x_ic, x_bc)), requires_grad= True).float()

    # tx_ic = 2 * np.random.rand(num_ic, 2) - 1      # x_ic = -1 ~ +1

    # create output values for IC and BCs
    u_ic = np.sin(-np.pi * x_ic)        # u_ic = -sin(pi*x_ic)
    u_bc = np.zeros((num_bc, 1))        # u_bc = 0
    u_train = torch.tensor(np.concatenate((u_ic, u_bc))).float()

    return t_train_f, x_train_f, t_train, x_train, u_train



def generate_Convection_train_data(num_train, num_ic, num_bc, ic_func):
    # collocation points
    t = np.random.rand(num_train, 1)
    x = np.random.rand(num_train, 1)    
    x =  2*np.pi*x      # x -> [0, 2*pi], t -> [0, 1]
    t_train_f = torch.tensor(t, requires_grad=True).float()
    x_train_f = torch.tensor(x, requires_grad=True).float()

    # create IC
    t_ic = np.zeros((num_ic, 1))
    x_ic = 2*np.pi*np.random.rand(num_ic, 1)      # x_ic =  0 ~ 2*pi
    u_ic = ic_func(x_ic)
    t_ic_train = torch.tensor(t_ic, requires_grad=True).float()
    x_ic_train = torch.tensor(x_ic, requires_grad=True).float()
    u_ic_train = torch.tensor(u_ic).float()

    # create BC
    tx_bc1 = np.random.rand(num_bc, 2)             # t_bc =  0 ~ 1
    tx_bc1[..., 1] = 2*np.pi                            # x = 2*pi
    tx_bc2 = np.copy(tx_bc1)
    tx_bc2[..., 1] = 0                                   # x = 0
    t_bc1_train = torch.tensor(tx_bc1[...,0:1]).float()
    x_bc1_train = torch.tensor(tx_bc1[...,1:2]).float()
    t_bc2_train = torch.tensor(tx_bc2[...,0:1]).float()
    x_bc2_train = torch.tensor(tx_bc2[...,1:2]).float()

    return t_train_f, x_train_f, t_ic_train, x_ic_train, u_ic_train, t_bc1_train, x_bc1_train, t_bc2_train, x_bc2_train

def generate_Reaction_diffusion_train_data(num_train, num_ic, num_bc, ic_func):
    # collocation points
    t = np.random.rand(num_train, 1)
    x = np.random.rand(num_train, 1)

    x =  2*np.pi*x       # x -> [0, 2*pi], t -> [0, 1]
    t_train_f = torch.tensor(t, requires_grad=True).float()
    x_train_f = torch.tensor(x, requires_grad=True).float()

    # create IC
    t_ic = np.zeros((num_ic, 1))                  # t_ic =  0
    x_ic = 2*np.pi*np.random.rand(num_ic, 1)      # x_ic =  0 ~ 2*pi
    u_ic = ic_func(x_ic)
    t_ic_train = torch.tensor(t_ic, requires_grad=True).float()
    x_ic_train = torch.tensor(x_ic, requires_grad=True).float()
    u_ic_train = torch.tensor(u_ic).float()

    # create BC
    t_bc1 = np.random.rand(num_bc, 1)      # t_bc1 =  0 ~ 1
    x_bc1 = np.ones((num_bc, 1))*2*np.pi   # x_bc1 = 2*pi
    t_bc2 = np.copy(t_bc1)                 # t_bc2 =  0 ~ 1
    x_bc2 = np.zeros((num_bc, 1))          # x_bc2 = 0

    t_bc1_train = torch.tensor(t_bc1, requires_grad=True).float()
    x_bc1_train = torch.tensor(x_bc1, requires_grad=True).float()
    t_bc2_train = torch.tensor(t_bc2, requires_grad=True).float()
    x_bc2_train = torch.tensor(x_bc2, requires_grad=True).float()

    return t_train_f, x_train_f, t_ic_train, x_ic_train, u_ic_train, t_bc1_train, x_bc1_train, t_bc2_train, x_bc2_train

def generate_Allen_cahn_train_data(num_train, num_ic, num_bc):
    t = np.random.rand(num_train, 1)
    x = np.random.rand(num_train, 1)
    x = 2*x-1
    t_train_f = torch.tensor(t, requires_grad=True).float()
    x_train_f = torch.tensor(x, requires_grad=True).float()

    # create IC
    tx_ic = 2 * np.random.rand(num_ic, 2) - 1      # x_ic = -1 ~ +1
    tx_ic[..., 0] = 0                                   # t_ic =  0
    tx_ic_tensor = torch.tensor(tx_ic).float()
    t_train = tx_ic_tensor[:, 0:1]
    x_train = tx_ic_tensor[:, 1:2]
    # x^2cos(pi*x)
    x_temp = tx_ic[..., 1, np.newaxis]
    u_ic_train = torch.tensor((x_temp**2)*np.cos(np.pi*x_temp)).float()

    # create BC
    t_bc1 = np.random.rand(num_bc, 1)       # t_bc =  0 ~ 1
    x_bc1 = np.ones((num_bc, 1))            # x = 1
    t_bc2 = np.copy(t_bc1)                  # t_bc =  0 ~ 1
    x_bc2 = -1 * np.ones((num_bc, 1))       # x = -1
    
    t_bc1_train = torch.tensor(t_bc1, requires_grad=True).float()
    x_bc1_train = torch.tensor(x_bc1, requires_grad=True).float()
    t_bc2_train = torch.tensor(t_bc2, requires_grad=True).float()
    x_bc2_train = torch.tensor(x_bc2, requires_grad=True).float()

    return t_train_f, x_train_f, t_train, x_train, u_ic_train, t_bc1_train, x_bc1_train, t_bc2_train, x_bc2_train

def generate_Helmholtz_2d_train_data(num_train, num_bc, a1, a2, coefficient):
    # colocation points
    yc = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
    xc = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
    with torch.no_grad():
        uc = helmholtz_2d_source_term(yc, xc, a1, a2, coefficient)
    # requires grad
    yc.requires_grad = True
    xc.requires_grad = True
    # boundary points
    north = torch.empty((num_bc, 1), dtype=torch.float32).uniform_(-1., 1.)
    west = torch.empty((num_bc, 1), dtype=torch.float32).uniform_(-1., 1.)
    south = torch.empty((num_bc, 1), dtype=torch.float32).uniform_(-1., 1.)
    east = torch.empty((num_bc, 1), dtype=torch.float32).uniform_(-1., 1.)
    yb = torch.cat([
        torch.ones((num_bc, 1)), west,
        torch.ones((num_bc, 1)) * -1, east
        ])
    xb = torch.cat([
        north, torch.ones((num_bc, 1)) * -1,
        south, torch.ones((num_bc, 1))
        ])
    ub = helmholtz_2d_exact_u(yb, xb, a1, a2)
    return yc, xc, uc, yb, xb, ub

def generate_Helmholtz_3d_train_data(num_train, num_bc, a1, a2, a3, coefficient):
    # colocation points
    yc = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
    xc = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
    zc = torch.empty((num_train, 1), dtype=torch.float32).uniform_(-1., 1.)
    with torch.no_grad():
        uc = helmholtz_3d_source_term(yc, xc, zc, a1, a2, a3, coefficient)
    # requires grad
    yc.requires_grad = True
    xc.requires_grad = True
    zc.requires_grad = True

    xb = [
        torch.ones(num_bc, 1),
        torch.ones(num_bc, 1)*-1,
        torch.empty((num_bc, 1), dtype= torch.float32).uniform_(-1, 1),
        torch.empty((num_bc, 1), dtype= torch.float32).uniform_(-1, 1),
        torch.empty((num_bc, 1), dtype= torch.float32).uniform_(-1, 1),
        torch.empty((num_bc, 1), dtype= torch.float32).uniform_(-1, 1)
    ]
    yb = [
        torch.empty((num_bc, 1), dtype= torch.float32).uniform_(-1, 1),
        torch.empty((num_bc, 1), dtype= torch.float32).uniform_(-1, 1),
        torch.ones(num_bc, 1),
        torch.ones(num_bc, 1)*-1,
        torch.empty((num_bc, 1), dtype= torch.float32).uniform_(-1, 1),
        torch.empty((num_bc, 1), dtype= torch.float32).uniform_(-1, 1),
    ]
    zb = [
        torch.empty((num_bc, 1), dtype= torch.float32).uniform_(-1, 1),
        torch.empty((num_bc, 1), dtype= torch.float32).uniform_(-1, 1),
        torch.empty((num_bc, 1), dtype= torch.float32).uniform_(-1, 1),
        torch.empty((num_bc, 1), dtype= torch.float32).uniform_(-1, 1),
        torch.ones(num_bc, 1),
        torch.ones(num_bc, 1)*-1,
    ]

    xb = torch.concat(xb).view(-1,)
    yb = torch.concat(yb).view(-1,)
    zb = torch.concat(zb).view(-1,)
    ub = helmholtz_3d_exact_u(xb, yb, zb, a1, a2, a3)
    

    return xc, yc, zc, uc, xb, yb, zb, ub
        


''' 2. Generate Test data '''
def generate_Convection_test_data(nu, beta, ic_func):
    number_x = 256
    number_t = 100
    h = 2*np.pi/number_x
    x = np.arange(0, 2*np.pi, h) # not inclusive of the last point
    t = np.linspace(0, 1, number_t).reshape(-1, 1)
    X, T = np.meshgrid(x, t)

    initial_u = ic_func(x)

    source = 0
    F = (np.copy(initial_u)*0)+source # F is the same size as initial_u

    complex_pos = 1j * np.arange(0, number_x/2+1, 1)
    complex_neg = 1j * np.arange(-number_x/2+1, 0, 1)
    complex = np.concatenate((complex_pos, complex_neg))
    complex2 = complex * complex

    initial_uhat = np.fft.fft(initial_u)
    nu_factor = np.exp(nu * complex2 * T - beta * complex * T)
    B = initial_uhat - np.fft.fft(F)*0 # at t=0, second term goes away
    uhat = B*nu_factor + np.fft.fft(F)*T # for constant, fft(p) dt = fft(p)*T
    u = np.real(np.fft.ifft(uhat))

    t_test = t
    x_test = x
    u_test = u

    return t_test, x_test, u_test

def generate_Reaction_diffusion_test_data(nu, rho, ic_func):
    number_x = 256
    number_t = 100
    length = 2*np.pi
    T = 1
    dx = length/number_x
    dt = T/number_t
    x = np.arange(0, length, dx) # not inclusive of the last point
    t = np.linspace(0, T, number_t).reshape(-1, 1)
    X, T = np.meshgrid(x, t)
    u = np.zeros((number_x, number_t))

    complex_pos = 1j * np.arange(0, number_x/2+1, 1)
    complex_neg = 1j * np.arange(-number_x/2+1, 0, 1)
    complex = np.concatenate((complex_pos, complex_neg))
    complex2 = complex * complex

    # call u0 this way so array is (n, ), so each row of u should also be (n, )
    initial_u = ic_func(x)
    u[:,0] = initial_u
    u_ = initial_u
    for i in range(number_t-1):
        u_ = reaction(u_, rho, dt)
        u_ = diffusion(u_, nu, dt, complex2)
        u[:,i+1] = u_
    t_test = t
    x_test = x
    u_test = u.T

    return t_test, x_test, u_test


def generate_Helmholtz_2d_test_data(num_test, a1, a2):
    # test points
    y = torch.linspace(-1, 1, num_test)
    x = torch.linspace(-1, 1, num_test)
    y, x = torch.meshgrid([y, x], indexing='ij')
    y_test = y.reshape(-1, 1)
    x_test = x.reshape(-1, 1)
    u_test = helmholtz_2d_exact_u(y_test, x_test, a1, a2)
    return y_test, x_test, u_test

def generate_Helmholtz_3d_test_data(num_test, a1, a2, a3):
    # test points
    y = torch.linspace(-1, 1, num_test)
    x = torch.linspace(-1, 1, num_test)
    z = torch.linspace(-1, 1, num_test)
    
    x, y, z = torch.meshgrid([x, y, z], indexing='ij')
    y_test = y.reshape(-1, 1)
    x_test = x.reshape(-1, 1)
    z_test = z.reshape(-1, 1)
    
    u_test = helmholtz_3d_exact_u(x_test, y_test, z_test, a1, a2, a3)

    return x_test, y_test, z_test, u_test