
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
rc('text', usetex=False)

def Burgers_plot(pde, it, u_pred, t, x, u_test, t_flat, x_flat, net_u, output_path, tag):
    # image plot
    if it % 15 ==0:
        fig = plt.figure(figsize=(20,8))
        gs = GridSpec(2, 4)
        plt.subplot(gs[0, 0:2])
        plt.pcolormesh(t, x, u_pred, cmap='rainbow',shading='auto')
        plt.xlabel('t')
        plt.ylabel('x')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label('u(t,x)')
        cbar.mappable.set_clim(-1, 1)

        plt.subplot(gs[0, 2:4])
        plt.pcolormesh(t, x, u_test, cmap='rainbow',shading='auto')
        plt.xlabel('t')
        plt.ylabel('x')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label('gt')
        cbar.mappable.set_clim(-1, 1)
        
        u_pred2 = []
        # plot u(t=const, x) cross-sections
        t_cross_sections = [0, 0.5, 0.75, 1.0]
        t_idx = [0, int(u_test.T.shape[0]/2), int(3*(u_test.T.shape[0]/4)),-1]
    
        for i, t_cs in enumerate(t_cross_sections):
            plt.subplot(gs[1, i])
            t_flat = np.full(x_flat.shape, t_cs).flatten()
            tx = torch.tensor(np.stack([t_flat, x_flat], axis=-1)).float()
            u_pred = net_u(tx[:,0:1], tx[:,1:2], pde)
            u_pred2.append(u_pred.detach().cpu().numpy().reshape(x_flat.shape))

            plt.plot(x_flat, u_test.T[t_idx[i]], 'b-', linewidth = 2, label = 'Exact')   
            plt.plot(x_flat, u_pred2[i], 'r--', linewidth = 2, label = 'Predict')   
            plt.title('t={}'.format(t_cs))
            plt.xlabel('x')
            plt.ylabel('u(t,x)')
        plt.savefig(output_path + "/{}_{}_fig_.png".format(tag, it))
        plt.close(fig)


def Convection_plot(pde, it, u_pred, t, x, u_test, t_flat, x_flat, net_u, output_path, tag):
    # image plot
    if it % 15 ==0:        
        fig = plt.figure(figsize=(14,8))
        gs = GridSpec(2, 2)
        plt.subplot(gs[0, 0])
        plt.pcolormesh(t, x, u_pred, cmap='rainbow',shading='auto')
        plt.title('Convection', fontsize = 20)
        plt.xlabel('t')
        plt.ylabel('x')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label('u(t,x)')
        cbar.mappable.set_clim(-1, 1)
        

        plt.subplot(gs[0, 1])
        plt.pcolormesh(t, x, u_test.T, cmap='rainbow',shading='auto')
        plt.xlabel('t')
        plt.ylabel('x')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label('gt')
        cbar.mappable.set_clim(-1, 1)

        u_pred2 = []
        # plot u(t=const, x) cross-sections
        t_cross_sections = [0, 1.0]
        test_idx = [0, -1]

        for i, t_cs in enumerate(t_cross_sections):
            plt.subplot(gs[1, i])
            t_flat = np.full(x_flat.shape, t_cs).flatten()
            tx = torch.tensor(np.stack([t_flat, x_flat], axis=-1)).float()
            u_pred = net_u(tx[:,0:1], tx[:,1:2], pde)
            u_pred2.append(u_pred.detach().cpu().numpy().reshape(x_flat.shape))
            plt.plot(x_flat, u_test[test_idx[i]], 'b-', linewidth = 4, label = 'Exact')   
            plt.plot(x_flat, u_pred2[i], 'r--', linewidth = 4, label = 'Prediction')            
            plt.title('t={}'.format(t_cs))
            plt.xlabel('x')
            plt.ylabel('u(t,x)')

        plt.savefig(output_path + "/{}_{}.png".format(tag, it))
        plt.close(fig)


def ReactionDiffusion_plot(pde, it, u_pred, t, x, u_test, t_flat, x_flat, net_u, output_path, tag):  
    # image plot
    if it % 15 ==0:
        fig = plt.figure(figsize=(14,8))
        gs = GridSpec(2, 2)
        plt.subplot(gs[0, 0])
        plt.pcolormesh(t, x, u_pred, cmap='rainbow',shading='auto')
        plt.xlabel('t')
        plt.ylabel('x')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label('u(t,x)')
        
        
        plt.subplot(gs[0, 1])
        plt.pcolormesh(t, x, u_test.T, cmap='rainbow',shading='auto')
        plt.xlabel('t')
        plt.ylabel('x')
        cbar = plt.colorbar(pad=0.05, aspect=10)
        cbar.set_label('gt')
        u_pred2 = []
        # plot u(t=const, x) cross-sections
        t_cross_sections = [0, 0.4]
        for i, t_cs in enumerate(t_cross_sections):
            plt.subplot(gs[1, i])
            t_flat = np.full(x_flat.shape, t_cs).flatten()
            tx = torch.tensor(np.stack([t_flat, x_flat], axis=-1)).float()
            u_pred = net_u(tx[:,0:1], tx[:,1:2], pde)
            u_pred2.append(u_pred.detach().cpu().numpy().reshape(x_flat.shape))
            plt.plot(x_flat, u_pred2[i])
            plt.title('t={}'.format(t_cs))
            plt.xlabel('x')
            plt.ylabel('u(t,x)')

        plt.savefig(output_path + "/{}_{}.png".format(tag, it))

        
        plt.close(fig)


def AllenCahn_plot(pde, it, t, x, u_sol, u_pred, t_flat, x_flat, net_u, output_path, tag):
    # image plot
    fig = plt.figure(figsize=(14,12))
    gs = GridSpec(3, 4)
    plt.subplot(gs[0,:2])
    plt.pcolormesh(t, x, u_sol, cmap='rainbow',shading='auto')
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('GT')
    cbar.mappable.set_clim(-1, 1)
    
    plt.subplot(gs[0,2:])
    plt.pcolormesh(t, x, u_pred, cmap='rainbow',shading='auto')
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('Pred')
    cbar.mappable.set_clim(-1, 1)

    

    # plot u(t=const, x) cross-sections
    n_cs = 4
    t_interval = int((t_flat.shape[0]-1)/(n_cs-1))
    for i in range(n_cs):
        plt.subplot(gs[1, i])
        t_idx = int(t_interval*i)
        t_value = t_flat[t_idx]
        plt.plot(x_flat, u_sol[:,t_idx])
        plt.title('t={}'.format(t_value))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
        plt.subplot(gs[2, i])
        tx = torch.tensor(np.stack([np.full(x_flat.shape, t_value), x_flat], axis=-1)).float()
        u_pred2 = net_u(tx[:,0:1], tx[:,1:2], pde)
        u_pred2 = u_pred2.detach().cpu().numpy().reshape(x_flat.shape)
        plt.plot(x_flat, u_pred2)
        plt.title('t={}'.format(t_value))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')

    plt.savefig(output_path + "/{}_{}.png".format(tag, it))
    plt.close(fig)

        
def Helmholtz_2d_plot(it, y, x, u, u_gt, num_test, output_path, tag):
    # ship back to cpu
    y = y.cpu().numpy().reshape(num_test, num_test)
    x = x.cpu().numpy().reshape(num_test, num_test)
    u = u.cpu().numpy().reshape(num_test, num_test)
    u_gt = u_gt.cpu().numpy().reshape(num_test, num_test)

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].set_aspect('equal')
    col0 = axes[0].pcolormesh(x, y, u_gt, cmap='rainbow', shading='auto')
    axes[0].set_xlabel('x', fontsize=12, labelpad=12)
    axes[0].set_ylabel('y', fontsize=12, labelpad=12)
    axes[0].set_title('Exact U', fontsize=18, pad=18)
    div0 = make_axes_locatable(axes[0])
    cax0 = div0.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col0, cax=cax0)

    axes[1].set_aspect('equal')
    col1 = axes[1].pcolormesh(x, y, u, cmap='rainbow', shading='auto')
    axes[1].set_xlabel('x', fontsize=12, labelpad=12)
    axes[1].set_ylabel('y', fontsize=12, labelpad=12)
    axes[1].set_title('Predicted U', fontsize=18, pad=18)
    div1 = make_axes_locatable(axes[1])
    cax1 = div1.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(col1, cax=cax1)

    axes[2].set_aspect('equal')
    col2 = axes[2].pcolormesh(x, y, np.abs(u-u_gt), cmap='rainbow', shading='auto')
    axes[2].set_xlabel('x', fontsize=12, labelpad=12)
    axes[2].set_ylabel('y', fontsize=12, labelpad=12)
    axes[2].set_title('Absolute error', fontsize=18, pad=18)
    div2 = make_axes_locatable(axes[2])
    cax2 = div2.append_axes('right', size='5%', pad=0.05)
    cbar = plt.colorbar(col2, cax=cax2)
    cbar.mappable.set_clim(0, 1)

    
    plt.tight_layout()
    if it % 25 ==0:
        fig.savefig(output_path + "/{}_{}.png".format(tag, it))
    plt.clf()
    plt.close(fig)

    
def Helmholtz_3d_plot(it, x, y, z, u, u_gt, num_test, output_path, tag):
    if z.numel() != u.numel():
        x, y, z = torch.meshgrid(x.view(-1), y.view(-1), z.view(-1), indexing='ij')

    # ship back to cpu
    y = y.cpu().numpy().reshape(num_test, num_test, num_test)
    x = x.cpu().numpy().reshape(num_test, num_test, num_test)
    z = z.cpu().numpy().reshape(num_test, num_test, num_test)
    u = u.cpu().numpy().reshape(num_test, num_test, num_test)
    u_gt = u_gt.cpu().numpy().reshape(num_test, num_test, num_test)

    # plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=u, s=0.5, cmap='seismic')

    fig.savefig(output_path + "/{}_{}.png".format(tag, it))



def NavierStokes_plot(it, num_test, net_u, output_path, tag):
    t_flat = np.ones(num_test) * 10 #np.linspace(0, 20, num_test) #  #
    x_flat = np.linspace(1, 8,  num_test) # 0 8 -> -15 25
    y_flat = np.linspace(-2, 2,  num_test) # -2 2 -> -8 8
    
    # t, x, y = np.meshgrid(t_flat, x_flat, y_flat)
    # T = torch.tensor(t_flat).float().requires_grad_(True).view(t_flat.shape[0],1)
    # X = torch.tensor(x_flat).float().requires_grad_(True).view(x_flat.shape[0],1)
    # Y = torch.tensor(y_flat).float().requires_grad_(True).view(y_flat.shape[0],1)                
    
    x, y = np.meshgrid(x_flat, y_flat)
    t = np.tile(t_flat, (1, num_test)).reshape(num_test, num_test)

    T = torch.tensor(t.flatten().reshape(t.flatten().shape[0],1),requires_grad = True).float() # 
    X = torch.tensor(x.flatten().reshape(x.flatten().shape[0],1),requires_grad = True).float() # requires_grad = True
    Y = torch.tensor(y.flatten().reshape(y.flatten().shape[0],1),requires_grad = True).float() # requires_grad = True

    
    
    uvp = net_u(T,X,Y)
        
    u_pred = uvp[:, 0:1]
    v_pred = uvp[:, 1:2]
    p_pred = uvp[:, 2:3]

    u_pred = u_pred.detach().cpu().numpy().reshape(x.shape)
    v_pred = v_pred.detach().cpu().numpy().reshape(x.shape)
    p_pred = p_pred.reshape(X.shape).detach().cpu().numpy().reshape(x.shape)
    # image plot
    fig = plt.figure(figsize=(14,8))
    gs = GridSpec(3, 1)
    plt.subplot(gs[0, :])
    
    plt.pcolormesh(x, y, u_pred, cmap='rainbow',shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    # cbar.mappable.set_clim(-1, 1)
    
    plt.subplot(gs[1, :])
    plt.pcolormesh(x, y, v_pred, cmap='rainbow',shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('v(t,x)')
    # cbar.mappable.set_clim(-1, 1)

    plt.subplot(gs[2, :])
    plt.pcolormesh(x, y, p_pred, cmap='rainbow',shading='auto')
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('p(t,x)')
    # cbar.mappable.set_clim(-1, 1)

    # plot u(t=const, x) cross-sections
    # t_cross_sections = [0, 0.5, 1.0]
    # for i, t_cs in enumerate(t_cross_sections):
    #     plt.subplot(gs[1, i])
    #     tx = torch.tensor(np.stack([np.full(t_flat.shape, t_cs), x_flat], axis=-1)).float()
    #     u_pred = net_u(tx[:,0:1], tx[:,1:2])
    #     u_pred = u_pred.detach().cpu().numpy().reshape(t_flat.shape)
    #     plt.plot(x_flat, u_pred)
    #     plt.title('t={}'.format(t_cs))
    #     plt.xlabel('x')
    #     plt.ylabel('u(t,x)')
    #plt.tight_layout()
    plt.savefig(output_path + "/{}_{}.png".format(tag, it))
    plt.show()
    plt.close(fig)    