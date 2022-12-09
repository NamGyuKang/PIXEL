import scipy.io
import numpy as np
import torch

def load_Burgers_ground_truth():
    data = scipy.io.loadmat('data/Burgers/burgers_shock.mat')
    u_test = data['usol']
    t_test = data['t']
    x_test = data['x']

    u_gt = np.real(data['usol']).T # (100, 256)  -> after transpose (t, x)
    x_gt = np.asarray(data['x']).flatten()[:,None] # (256, 1)
    t_gt = np.asarray(data['t']).flatten()[:,None] # (100, 1)
    
    data_u = torch.tensor(u_gt.reshape(-1, 1)).float()
    
    X, T = np.meshgrid(x_gt,t_gt)
    
    data_x = torch.tensor(X.reshape(-1, 1), requires_grad = True).float()
    data_t = torch.tensor(T.reshape(-1, 1), requires_grad = True).float()

    return data_x, data_t, data_u, u_test, t_test, x_test


def generate_Convection_inverse_data(x_test, t_test, nu, beta, ic_func):
    number_x = 256
    number_t = 100
    h = 2*np.pi/number_x
    x = np.arange(0, 2*np.pi, h) # not inclusive of the last point
    
    t = np.linspace(0, 1, number_t).reshape(-1, 1)
    X, T = np.meshgrid(x, t)
    
    initial_u = ic_func(x)

    source = 0
    F = (np.copy(initial_u)*0)+source # F is the same size as u0

    complex_pos = 1j * np.arange(0, number_x/2+1, 1)
    complex_neg = 1j * np.arange(-number_x/2+1, 0, 1)
    complex = np.concatenate((complex_pos, complex_neg))
    complex2 = complex * complex

    initial_uhat = np.fft.fft(initial_u)
    nu_factor = np.exp(nu * complex2 * T - beta * complex * T)
    B = initial_uhat - np.fft.fft(F)*0 # at t=0, second term goes away
    uhat = B*nu_factor + np.fft.fft(F)*T # for constant, fft(p) dt = fft(p)*T
    u = np.real(np.fft.ifft(uhat))

    u_inverse_data = torch.tensor(u).float()    

    x_inverse_np, t_inverse_np = np.meshgrid(x_test, t_test)

    t_inverse_data = torch.tensor(t_inverse_np.flatten(), requires_grad =True).float().view(-1, 1)
    x_inverse_data = torch.tensor(x_inverse_np.flatten(), requires_grad =True).float().view(-1, 1)

    return t_inverse_data, x_inverse_data, u_inverse_data


def reaction(u, rho, dt):
    """ du/dt = rho*u*(1-u)
    """
    A = u * np.exp(rho * dt)
    B = (1 - u)
    u = A / (A+B)
    return u

def diffusion(u, nu, dt, complex2):
    """ du/dt = nu*d2u/dx2
    """
    A = np.exp(nu * complex2 * dt)
    u_hat = np.fft.fft(u)
    u_hat *= A
    u = np.real(np.fft.ifft(u_hat))
    return u



def generate_Reaction_diffusion_inverse_data(x_test, t_test, nu, rho, ic_func):
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

    u_inverse_data = torch.tensor(u.T).float()    

    t_flat = t.flatten()
    x_flat = x.flatten()
    x_inverse_np, t_inverse_np = np.meshgrid(x_test, t_test)

    t_inverse_data = torch.tensor(t_inverse_np.flatten(), requires_grad =True).float().view(-1, 1)
    x_inverse_data = torch.tensor(x_inverse_np.flatten(), requires_grad =True).float().view(-1, 1)

    return t_flat, x_flat, t_inverse_data, x_inverse_data, u_inverse_data

def load_AllenCahn_ground_truth():
    data = scipy.io.loadmat('data//Allen-Cahn/AC.mat')
    u_sol = data['uu']
    test_t_flat = data['tt'][0]
    test_x_flat = data['x'][0]


    test_t, test_x = np.meshgrid(test_t_flat, test_x_flat)
    
    test_tx = torch.tensor(np.stack([test_t.flatten(), 
        test_x.flatten()], axis=-1)).float()
    test_u_sol_tensor = torch.tensor(u_sol.flatten().reshape(-1,1)).float()
    
    test_t_tensor = torch.tensor(test_t.flatten(), requires_grad = True).float().view(-1, 1)
    test_x_tensor = torch.tensor(test_x.flatten(), requires_grad = True).float().view(-1, 1)

    test_u_sol= u_sol.flatten().reshape(-1,1)
        
    x_test = test_tx[:,0:1]
    t_test = test_tx[:,1:2]

    return test_t_tensor, test_x_tensor, test_u_sol_tensor, t_test, x_test, test_u_sol, test_t_flat, test_x_flat, test_t, test_x, u_sol

def helmholtz_2d_exact_u(y, x, a1, a2):
    return torch.sin(a1*torch.pi*y) * torch.sin(a2*torch.pi*x)

def helmholtz_2d_source_term(y, x, a1, a2, coefficient):
    u_gt = helmholtz_2d_exact_u(y, x, a1, a2)
    u_yy = -(a1*torch.pi)**2 * u_gt
    u_xx = -(a2*torch.pi)**2 * u_gt
    return  u_yy + u_xx + coefficient*u_gt

def generate_Helmholtz_2d_inverse_data(a1, a2):
    # test points
    y = torch.linspace(-1, 1, 700) 
    x = torch.linspace(-1, 1, 700)
    y, x = torch.meshgrid([y, x], indexing='ij')
    y_test = y.reshape(-1, 1)
    x_test = x.reshape(-1, 1)
    u_test = helmholtz_2d_exact_u(y_test, x_test, a1, a2)
    return y_test, x_test, u_test    

def helmholtz_3d_exact_u(x, y, z, a1, a2, a3):
        return torch.sin(a1*torch.pi*y) * torch.sin(a2*torch.pi*x) * torch.sin(a3*torch.pi*z)

def helmholtz_3d_source_term(x, y, z, a1, a2, a3, coefficient):
    u_gt = helmholtz_3d_exact_u(x, y, z, a1, a2, a3)
    u_yy = -(a1*torch.pi)**2 * u_gt
    u_xx = -(a2*torch.pi)**2 * u_gt
    u_zz = -(a3*torch.pi)**2 * u_gt
    return  u_yy + u_xx + u_zz + coefficient*u_gt

def generate_Helmholtz_3d_inverse_data(a1, a2, a3):
    # test points
    y = torch.linspace(-1, 1, 100) 
    x = torch.linspace(-1, 1, 100)
    z = torch.linspace(-1, 1, 100)
    x, y, z = torch.meshgrid([x, y, z], indexing='ij')
    y_test = y.reshape(-1, 1)
    x_test = x.reshape(-1, 1)
    z_test = z.reshape(-1, 1)
    
    u_test = helmholtz_3d_exact_u(x_test, y_test, z_test, a1, a2, a3)
    return x_test, y_test, z_test, u_test

def load_Navier_Stokes_ground_truth():
    exact_u = np.loadtxt('data/Navier-Stokes/NS_exact_u_solution.txt').reshape(-1,1)  # (5000, 200)
    exact_v = np.loadtxt('data/Navier-Stokes/NS_exact_v_solution.txt').reshape(-1,1)  # (5000, 200)
    exact_x = np.loadtxt('data/Navier-Stokes/NS_exact_x_solution.txt').reshape(5000, 1)  # (5000, 1)
    exact_y = np.loadtxt('data/Navier-Stokes/NS_exact_y_solution.txt').reshape(5000, 1)  # (5000, 1)
    exact_t = np.loadtxt('data/Navier-Stokes/NS_exact_t_solution.txt').reshape(200, 1)   # (200, 1)
    exact_t = (np.tile(exact_t, (1,5000)).T).reshape(-1,1)
    exact_x = np.tile(exact_x, (1,200)).reshape(-1,1)
    exact_y = np.tile(exact_y, (1,200)).reshape(-1,1)
    
    txt_u_sample = torch.tensor(exact_u).float()
    txt_v_sample = torch.tensor(exact_v).float()
    txt_t_sample = torch.tensor(exact_t, requires_grad=True).float()
    txt_x_sample = torch.tensor(exact_x, requires_grad=True).float()
    txt_y_sample = torch.tensor(exact_y, requires_grad=True).float()
    
    return txt_t_sample, txt_x_sample, txt_y_sample, txt_u_sample, txt_v_sample

def generate_Navier_Stokes_inverse_data(num_train):
    txt_t_sample, txt_x_sample, txt_y_sample, txt_u_sample, txt_v_sample = load_Navier_Stokes_ground_truth()
    idx = np.random.permutation(1000000)[:100000]
    t_train = txt_t_sample[idx]
    x_train = txt_x_sample[idx]
    y_train = txt_y_sample[idx]
    txt_u = txt_u_sample[idx]
    txt_v = txt_v_sample[idx]

    txy = np.random.uniform(0, 20, (num_train, 3))
    txy[..., 1] = 0.35*txy[..., 1]+1
    txy[..., 2] = 0.2*txy[..., 2]-2
    
    txy_f = torch.tensor(txy, requires_grad=True).float() 
    
    
    t_train_f = txy_f[:, 0:1]
    x_train_f = txy_f[:, 1:2]
    y_train_f = txy_f[:, 2:3]

    return t_train_f, x_train_f, y_train_f, t_train, x_train, y_train, txt_u, txt_v