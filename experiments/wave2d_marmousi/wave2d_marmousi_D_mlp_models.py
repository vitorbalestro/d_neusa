import argparse
import torch
import torch.nn as nn
import time
import os
import numpy as np
import json

from models.PINN import PINN
from models.QRes import QRes
from models.FLS import FLS



from basis.cosine2d import CosineBasis2D

from utils.directory_manager import DirectoryManager
from utils.grid import Random3DGrid
from utils.trainers import AdamTrainer
from utils.metrics import Metrics
from utils.plots import experiment_plot_2d
from scipy.interpolate import RegularGridInterpolator

from solvers.wave import wave_solver


def temporal_derivative(model,t,x,y):
    u = model(t,x,y)

    return torch.autograd.grad(u,t,grad_outputs=torch.ones_like(u),create_graph=False)[0]

def initial_condition(sigma,center,x,y):
    return torch.exp(-((x-center[0])**2+(y-center[1])**2)/(2*sigma**2))

def neumann_loss(model, x, y, t, axis):
    if axis == "x":
        x = x.requires_grad_()
        u = model(t, x, y)
        du_dn = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]

    elif axis == "y":
        y = y.requires_grad_()
        u = model(t, x, y)
        du_dn = torch.autograd.grad(
            u,
            y,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
        )[0]

    return torch.mean(du_dn**2)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

def instantiate_model(model_name):
    if model_name == "PINN":
        model = PINN(in_dim=3, out_dim=1, hidden_dim=128,num_layer=4)
    elif model_name == "QRes":
        model = QRes(in_dim=3, out_dim=1, hidden_dim=128,num_layer=4)
    elif model_name == "FLS":
        model = FLS(in_dim=3, out_dim=1, hidden_dim=128,num_layer=4)
    model.apply(init_weights)
    model = model.to(device)
    return model

def loss_function(model, training_grid, sigma,center,c_squared_interp, device, u0_model=initial_condition,ut0_model=None):
    training_grid.update_random_grid()
    training_grid.to(device)

    x_ic = training_grid.x_ic
    y_ic = training_grid.y_ic
    t_ic = training_grid.t_ic.detach().requires_grad_()

    x_res = training_grid.x_res.detach().requires_grad_()
    y_res = training_grid.y_res.detach().requires_grad_()
    t_res = training_grid.t_res.detach().requires_grad_()
    
    u0 = u0_model(sigma,center,x_ic,y_ic)

    u_ic = model(t_ic,x_ic,y_ic)
    u_t_ic = torch.autograd.grad(u_ic, t_ic, grad_outputs=torch.ones_like(u_ic), create_graph=True)[0]

    if ut0_model == None:
        ut0 = torch.zeros_like(u_t_ic)
    else:
        ut0 = ut0_model(sigma,center,x_ic,y_ic)

    loss_ic = torch.mean((u_ic-u0)**2) + torch.mean((u_t_ic-ut0)**2)


        
    loss_bc = (
        neumann_loss(model, training_grid.x_lb_bc, training_grid.y_bc, training_grid.t_bc, "x")
        + neumann_loss(model, training_grid.x_ub_bc, training_grid.y_bc, training_grid.t_bc, "x")
        + neumann_loss(model, training_grid.x_bc, training_grid.y_lb_bc, training_grid.t_bc, "y")
        + neumann_loss(model, training_grid.x_bc, training_grid.y_ub_bc, training_grid.t_bc, "y")
    ) / 4.0

    u = model(t_res,x_res,y_res)
    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_res, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_t = torch.autograd.grad(u,t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]

    x_res = x_res.detach().cpu().numpy()
    y_res = y_res.detach().cpu().numpy()
    points = np.stack([x_res,y_res],axis=-1)

    c_squared = c_squared_interp(points)

    c_squared = torch.tensor(c_squared).to(device)

    loss_res = torch.mean((u_tt - c_squared*(u_xx+u_yy))**2)

    return (1e3*loss_ic, loss_bc, loss_res)


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Wave 2d Marmousi D-MLP models')
    parser.add_argument('--model', type=str, default='PINN')    # options: 'PINN', 'FLS', 'QRes', 'PINNsFormer
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')     
    parser.add_argument('--steps', type=int, default=10000)  
    parser.add_argument('--lr', type=float, default=0.001)
    
    args = parser.parse_args()
    model_name = args.model
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr

    torch.manual_seed(seed)

    model_name_str = "D_" + model_name

    directory_manager = DirectoryManager(
        equation="wave2d_marmousi",
        model=model_name_str,
        dir_name=dir_name,
        ablation=False,
        specifier=False,
        seed=seed
    )

    directory_manager.config()
    
    # domain and grid
    domain_x = [-2.0,2.0]
    domain_y = [-2.0,2.0]
    domain_t = [0.0,2.0]

    # extended grid for training
    domain_x_ext = [-4.0,4.0]
    domain_y_ext = [-4.0,4.0]
    M = 401
    N = 401
    
    # resolution of predicted and reference solutions
    n_x = 201
    n_y = 201
    n_t = 201
    
    # size of training samples
    n_ic = 500
    n_bc = 200
    n_res = 5000
    

    # initial condition parameters
    sigma = torch.tensor(0.1,dtype=torch.float32).to(device)
    center = (0.0,-0.5)
    
    # load velocity profile (and interpolate to the training grid)
    velocity_profile = np.loadtxt("data/wave2d_marmousi/velocity_profile.txt",dtype=np.float32)  # 401 x 401
    c_squared = torch.tensor(velocity_profile * velocity_profile)
    c_squared = c_squared.numpy()
    xs = np.linspace(*domain_x_ext,M)
    ys = np.linspace(*domain_y_ext,N)
    c_squared_interp = RegularGridInterpolator((xs,ys),c_squared) 
    
    # time domain decomposition
    n_slices = 20
    assert (n_t - 1) % n_slices == 0
    offset = (n_t - 1) // n_slices       # quantity of time steps considered in each sub-network
    time_slices = [offset*i for i in range(1,n_slices+1)]
    ts = torch.linspace(*domain_t,n_t)

    model_list  = [instantiate_model(model_name) for _ in range(n_slices)]
    model_list = nn.ModuleList(model_list)

    # training
    start_time = time.time()

    current_domain_t = [ts[0].item(),ts[time_slices[0]].item()]
    training_grid = Random3DGrid(domain_x_ext, domain_y_ext, current_domain_t, n_ic, n_bc, n_res)
    starter_steps = steps 
    trainer = AdamTrainer(params = list(model_list[0].parameters()),
                        adam_lr = lr,
                        adam_steps = starter_steps,
                        learning_rate_scheduler = True,
                        loss_function = lambda : loss_function(model_list[0],training_grid,sigma,center,c_squared_interp,device)
                    )
    
    trainer.train()
    

    for i in range(1,n_slices):
        current_domain_t = [ts[time_slices[i-1]].item(),ts[time_slices[i]].item()]
        training_grid = Random3DGrid(domain_x_ext, domain_y_ext, current_domain_t, n_ic, n_bc, n_res)
        current_t0 = current_domain_t[0] * torch.ones(n_ic).to(device)
        current_t0 = current_t0.unsqueeze(-1).requires_grad_()

        model_list[i].load_state_dict(model_list[i-1].state_dict()) # each model is initialized as the previous model
        
        # the initial condition is the previous model prediction on its last time step
        for param in model_list[i-1].parameters():
            param.requires_grad = False
        u0_model = lambda sigma,center,x,y : model_list[i-1](current_t0,x,y)
        ut0_model = lambda sigma,center,x,y: temporal_derivative(model_list[i-1],current_t0,x,y)

        trainer = AdamTrainer(params=list(model_list[i].parameters()),
                            adam_lr=lr,
                            adam_steps=steps,
                            learning_rate_scheduler=True,
                            loss_function=lambda: loss_function(model_list[i],training_grid,sigma,center,c_squared_interp,device,u0_model,ut0_model)
                        )
        
        trainer.train()
        
        
    end_time = time.time()
    training_time = end_time - start_time
    
    # predicted solution
    u_pred = []
    basis = CosineBasis2D(M,N,domain_x_ext,domain_y_ext,device=device)
    xs = basis.x[100:301]
    ys = basis.y[100:301]
    
    for i in range(0,n_slices):
        if i == 0:
            t = ts[i:time_slices[i]]
        elif i == n_slices - 1:
            t = ts[time_slices[i-1]:time_slices[i]+1]
        else:
            t = ts[time_slices[i-1]:time_slices[i]]
            
        T,X,Y = torch.meshgrid(t,xs,ys,indexing='ij')

        t_inf = T.flatten().unsqueeze(1)
        x_inf = X.flatten().unsqueeze(1)
        y_inf = Y.flatten().unsqueeze(1)

        model_list[i] = model_list[i].to('cpu')
        with torch.no_grad():
            current_u_pred = model_list[i](t_inf,x_inf,y_inf)
            current_u_pred = current_u_pred.reshape(len(t),n_x,n_y)

        u_pred.append(current_u_pred)
    
    u_pred = torch.cat(u_pred,dim=0).detach().cpu().numpy()

    # reference solution
    X,Y = torch.meshgrid(basis.x,basis.y,indexing='ij')
    
    grid = torch.stack((X,Y),dim=-1)    # shape = (M,N,2)
    xs = grid[:,:,0:1]
    ys = grid[:,:,1:2]
    
    u0 = initial_condition(sigma.to('cpu'),center,xs,ys).squeeze(-1)
    velocity_profile = np.loadtxt("data/wave2d_marmousi/velocity_profile.txt",dtype=np.float32)
    c = torch.tensor(velocity_profile)
    
    gt_time_steps = 401
    ts_gt = torch.linspace(*domain_t,gt_time_steps)
    u_ref_ps = wave_solver(
            c=c,
            u0=u0,
            basis=basis,
            ts=ts_gt,
            device=device
    )
    
    u_ref = u_ref_ps[:,100:301,100:301]
    time_idx = range(0,gt_time_steps,2)
    u_ref = u_ref[time_idx,:,:]
 
    
    # compute metrics
    # metrics
    metrics = Metrics(u_pred,u_ref)
    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    # save solutions

    np.save(os.path.join(directory_manager.save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(directory_manager.save_dir_solutions, "u_ref.npy"),u_ref)


    metrics_dict = {
        "model": model_name_str,
        "relative_l2_error": float(relative_l2_error_value),
        "relative_l1_error": float(relative_l1_error_value),
        "steps": int(steps),
        "training_time": float(training_time)
    }

    with open(os.path.join(directory_manager.save_dir_metrics, "metrics.json"), "w") as f:
            json.dump(metrics_dict, f, indent=4)    
    
    
    # plot
    experiment_plot_2d(
        ts=ts,
        u_pred=u_pred,
        u_ref=u_ref,
        figs_dir=directory_manager.figs_dir,
        fig_name="wave2d_marmousi_plot.png",
        extent=[*domain_x,*domain_y],
        cmap="seismic",
    )