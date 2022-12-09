import numpy as np
import torch
import sys
from plot import *
from ground_truth import generate_Navier_Stokes_inverse_data

def save_loss_list(problem, loss_list, it, output_path, save_it = 50):
    if problem =='inverse':
        save_it = 5
    if it % save_it ==0:
        np.save(output_path + "/loss_{}.png".format(it), loss_list)


def Burgers_test(pde, t_test, x_test, u_test, lambda_1, net_u, problem, it, loss_list, output_path, tag):
    t_flat = t_test.flatten()
    x_flat = x_test.flatten()
    t, x = np.meshgrid(t_test, x_test)
    tx = torch.tensor(np.stack([t.flatten(), x.flatten()], axis=-1)).float()
    u_pred = net_u(tx[:,0:1], tx[:,1:2], pde)
    
    u_pred = u_pred.detach().cpu().numpy().reshape(t.shape)
    u_gt = u_test.reshape(-1,1)

    l2_loss = np.linalg.norm(u_gt-u_pred.reshape(-1,1))/np.linalg.norm(u_gt)

    if problem == 'forward':
        if it % 15 ==0:
            # logger.error('Iter %d, l2_Loss: %.5e', it+1, l2_loss)   
            print('[Test Iter:%d, 12_Loss: %.5e]'%(it, l2_loss))
        loss_list.append(l2_loss)
    
    elif problem == 'inverse':
        if it % 1 ==0:
            print('[Test Iter:%d, lambda_1: %.5e, l2_Loss: %.5e]'%(it, lambda_1, l2_loss))
            # logger.error('Iter %d, lambda: %.5e, l2_Loss: %.5e', it+1, lambda_1, l2_loss)   

        loss_list.append(lambda_1.item())

    save_loss_list(problem, loss_list, it, output_path)

    if it % 15 == 0 :
        Burgers_plot(pde, it, u_pred, t, x, u_test, t_flat, x_flat, net_u, output_path, tag)








def Convection_test(pde, t_test, x_test, u_test, lambda_1, net_u, problem, it, loss_list, output_path, tag):
    t_flat = t_test.flatten()
    x_flat = x_test.flatten()
    t, x = np.meshgrid(t_test, x_test)
    tx = torch.tensor(np.stack([t.flatten(), x.flatten()], axis=-1),
                        requires_grad=True).float()
    u_pred = net_u(tx[:,0:1], tx[:,1:2], pde)
    u_pred = u_pred.detach().cpu().numpy().reshape(t.shape)

    u_gt = u_test.T.reshape(-1,1)
    l2_loss = np.linalg.norm(u_gt-u_pred.reshape(-1,1))/np.linalg.norm(u_gt)

    if problem == 'forward':
        if it % 15 ==0:
            print('[Test Iter:%d, Loss: %.5e]'%(it, l2_loss))
            # logger.error('Iter %d, FLOPs: %.5e, l2_Loss: %.5e' , it+1, flops_total, l2_loss) 
        loss_list.append(l2_loss)

    elif problem == 'inverse':  
        print('[Test Iter:%d, lambda: %.5e, Loss: %.5e]'%(it, lambda_1, l2_loss))    
        # logger.error('(TEST) Iter %d, lambda: %.5e, l2_Loss: %.5e' , it+1, lambda_1, l2_loss)    
        loss_list.append(lambda_1.item())

    save_loss_list(problem, loss_list, it, output_path)

    if it % 15 == 0 :
        Convection_plot(pde, it, u_pred, t, x, u_test, t_flat, x_flat, net_u, output_path, tag)




    
def ReactionDiffusion_test(pde, t_test, x_test, u_test, lambda_1, net_u, problem, it, loss_list, output_path, tag):
    t_flat = t_test.flatten()
    x_flat = x_test.flatten()
    t, x = np.meshgrid(t_test, x_test)
    tx = torch.tensor(np.stack([t.flatten(), x.flatten()], axis=-1),
                        requires_grad=True).float()
    u_pred = net_u(tx[:,0:1], tx[:,1:2], pde)
    u_pred = u_pred.detach().cpu().numpy().reshape(t.shape)

    u_gt = u_test.T.reshape(-1,1)
    l2_loss = np.linalg.norm(u_gt-u_pred.reshape(-1,1))/np.linalg.norm(u_gt)

    if problem == 'forward':
        if it % 5 == 0 :
            print('[Test Iter:%d, Loss: %.5e]'%(it, l2_loss))
            # logger.error('Iter %d, l2_loss: %.5e' , it+1, l2_loss)
        loss_list.append(l2_loss)

    elif problem == 'inverse':
        print('[Test Iter:%d, lambda_1: %.5e, Loss: %.5e]'%(it, lambda_1, l2_loss))
        loss_list.append(lambda_1.item())    

    save_loss_list(problem, loss_list, it, output_path)
    
    if it % 15 == 0 :
        ReactionDiffusion_plot(pde, it, u_pred, t, x, u_test, t_flat, x_flat, net_u, output_path, tag)

def AllenCahn_test(pde, test_t_flat, test_x_flat, test_t, test_x, test_u_sol, u_sol, inverse_lambda, net_u, problem, it, loss_list, output_path, tag):
    t_flat = test_t_flat
    x_flat = test_x_flat
    t = test_t
    x = test_x

    tx = torch.tensor(np.stack([t.flatten(), x.flatten()], axis=-1),
                        requires_grad=True).float()
    u_pred = net_u(tx[:,0:1], tx[:,1:2], pde)
    u_pred = u_pred.detach().cpu().numpy()
    u_pred_test = u_pred.flatten().reshape(-1, 1)
    u_pred = u_pred.reshape(t.shape)
    

    loss_test = np.linalg.norm(test_u_sol-u_pred_test)/np.linalg.norm(test_u_sol)


    if problem == 'forward':
        if it % 15 ==0:
            # logger.error('Iter %d, l2_Loss: %.5e', it+1, loss_test.item())   
            print('[Test Iter:%d, 12_Loss: %.5e]'%(it, loss_test.item()))
        loss_list.append(loss_test.item())
    elif problem == 'inverse':
        if it % 1 ==0:
            print('[Test Iter:%d, lambda_1: %.5e, l2_Loss: %.5e]'%(it, inverse_lambda, loss_test.item()))
            # logger.error('Iter %d, lambda: %.5e, l2_Loss: %.5e', it+1, lambda_1, loss_test.item())   

        loss_list.append(inverse_lambda.item())
        
    save_loss_list(problem, loss_list, it, output_path)

    if it % 15 == 0 :
        AllenCahn_plot(pde, it, t, x, u_sol, u_pred, t_flat, x_flat, net_u, output_path, tag)

    
def Helmholtz_2d_test(pde, y_test, x_test, u_test, inverse_lambda, net_u, problem, it, loss_list, output_path, tag, num_test):
    u_pred = net_u(y_test, x_test, pde)
    u_pred_arr = u_pred.detach().cpu().numpy()
    u_test_arr = u_test.detach().cpu().numpy()
    
    l2_loss = np.linalg.norm(u_pred_arr - u_test_arr) / np.linalg.norm(u_test_arr)
    if problem == 'forward':
        loss_list.append(l2_loss)
        if it % 15 ==0 :
            print('[Test Iter:%d, Loss: %.5e]'%(it, l2_loss))
            # logger.error('Iter %d, l2_Loss: %.5e', it+1, l2_loss)
            
    elif problem == 'inverse':
        print('[Test Iter:%d, lambda: %.5e, Loss: %.5e]'%(it, inverse_lambda, l2_loss))
        # logger.error('Iter %d, lambda: %.5e, l2_Loss: %.5e', it+1, lambda_1, l2_loss)   
        loss_list.append(inverse_lambda.item())
    
    sys.stdout.flush()
    save_loss_list(problem, loss_list, it, output_path)

    if it % 15 == 0 :
        Helmholtz_2d_plot(it, y_test, x_test, u_pred.detach(), u_test, num_test, output_path, tag)
        
def Helmholtz_3d_test(pde, x_test, y_test, z_test, u_test, inverse_lambda, net_u, problem, it, loss_list, output_path, tag, num_test):
    u_pred = net_u(x_test, y_test, z_test)
    u_pred_arr = u_pred.detach().cpu().numpy()
    u_test_arr = u_test.detach().cpu().numpy()
    
    
    l2_loss = np.linalg.norm(u_pred_arr - u_test_arr) / np.linalg.norm(u_test_arr)
    if problem == 'forward':
        loss_list.append(l2_loss)
        if it % 15 ==0 :
            print('[Test Iter:%d, Loss: %.5e]'%(it, l2_loss))
            
    elif problem == 'inverse':
        print('[Test Iter:%d, lambda: %.5e, Loss: %.5e]'%(it, inverse_lambda, l2_loss))
        loss_list.append(inverse_lambda.item())

    sys.stdout.flush()
    save_loss_list(problem, loss_list, it, output_path)

    if it % 10000 == 0 :
        Helmholtz_3d_plot(it, x_test, y_test, z_test, u_pred.detach(), u_test, num_test, output_path, tag)

def Navier_Stokes_3d_test(lambda_1, lambda_2, net_f_3d, net_u_3d, t_train_f, x_train_f, y_train_f, it, loss_list, output_path, tag, num_test, num_train):
    t_train_f, x_train_f, y_train_f, t_train, x_train, y_train, txt_u, txt_v  = generate_Navier_Stokes_inverse_data(num_train) #, t_index = 0, t_interval= 1)
    uvp = net_u_3d(t_train, x_train, y_train)
    u = uvp[:,0:1]
    v = uvp[:,1:2]
        
    f_u_pred, f_v_pred = net_f_3d(t_train_f, x_train_f, y_train_f)
    
    loss_u = torch.mean((txt_u - u.view(txt_u.shape)) ** 2)
    loss_v = torch.mean((txt_v - v.view(txt_v.shape)) ** 2)
    loss_f_u = torch.mean(f_u_pred ** 2)
    loss_f_v = torch.mean(f_v_pred ** 2)
    loss = loss_u + loss_v + loss_f_u + loss_f_v
    
    print('Test (Iter %d,  l1: %.5e, l2: %.5e, Loss: %.5e, Loss_u: %.5e, Loss_v: %.5e, Loss_f_u: %.5e, Loss_f_v: %.5e' % (it, lambda_1.item(), lambda_2.item(), loss.item(), loss_u.item(), loss_v.item(), loss_f_u.item(), loss_f_v.item()))
    loss_list.append(lambda_1.item())
    loss_list.append(lambda_2.item())
    save_loss_list('inverse', loss_list, it, output_path, save_it = 5)

    if it % 15 == 0 :
        NavierStokes_plot(it, num_test, net_u_3d, output_path, tag)