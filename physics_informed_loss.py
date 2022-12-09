import torch

def Burgers(u, t, x, nu, inverse_lambda, problem):
    """ The pytorch autograd version of calculating residual """
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    
    if problem == 'forward':
        f = u_t + u * u_x - nu * u_xx
    elif problem == 'inverse':
        f = u_t + u * u_x - inverse_lambda * u_xx
    
    return f


def Convection(u, t, x, beta, inverse_lambda, problem):
    """ The pytorch autograd version of calculating residual """
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    
    if problem == 'forward':
        f = u_t + beta*u_x
    elif problem == 'inverse':
        f = u_t + inverse_lambda*u_x

    return f

def ReactionDiffusion(u, t, x, nu, rho, inverse_lambda, problem):
    """ The pytorch autograd version of calculating residual """   
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    
    if problem == 'forward':
        f = u_t - nu*u_xx - rho*u + rho*u**2
    elif problem == 'inverse':
        f = u_t - inverse_lambda*u_xx - rho*u + rho*u**2   
    return f


def AllenCahn(u, t, x, nu, inverse_lambda, problem):
    """ The pytorch autograd version of calculating residual """
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]

    if problem == 'forward':
        f = u_t + 5*(u**3) - 5*u - nu*u_xx 
    elif problem == 'inverse':
        f = u_t + inverse_lambda*(u**3) - 5*u - nu*u_xx 
    
    return f

def Helmholtz_2d(u, y, x, coefficient, inverse_lambda, problem):
    """ The pytorch autograd version of calculating residual """
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), True, True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), True, True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), True, True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), True, True)[0]
    
    if problem == 'forward':
        f =  u_yy + u_xx + coefficient*u
    elif problem == 'inverse':
        f =  u_yy + u_xx + inverse_lambda*u
    return f

def Helmholtz_3d(u, x, y, z, coefficient, inverse_lambda, problem):
    """ The pytorch autograd version of calculating residual """
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), True, True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), True, True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), True, True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), True, True)[0]
    u_z = torch.autograd.grad(u, z, torch.ones_like(u), True, True)[0]
    u_zz = torch.autograd.grad(u_z, z, torch.ones_like(u_z), True, True)[0]
    
    if problem == 'forward':
        f =  u_yy + u_xx + u_zz + coefficient*u
    elif problem == 'inverse':
        f =  u_yy + u_xx + u_zz + inverse_lambda*u
    return f

def Navier_Stokes_3d(uvp, t, x, y, inverse_lambda_1, inverse_lambda_2):
    """ The pytorch autograd version of calculating residual """
    u = uvp[:,0:1]
    v = uvp[:,1:2]
    p = uvp[:,2:3]
    
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), retain_graph=True, create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), retain_graph=True, create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), retain_graph=True, create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), retain_graph=True, create_graph=True)[0]

    f_u = u_t + inverse_lambda_1*(u*u_x + v*u_y) + p_x - inverse_lambda_2*(u_xx + u_yy)
    f_v = v_t + inverse_lambda_1*(u*v_x + v*v_y) + p_y - inverse_lambda_2*(v_xx + v_yy)
   
    return f_u, f_v
