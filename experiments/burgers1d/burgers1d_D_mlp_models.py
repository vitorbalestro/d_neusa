import argparse
import torch
import torch.nn as nn
import time
import os
import numpy as np
import json
from math import pi
from basis.fourier1d import FourierBasis1D

from models.PINN import PINN
from models.QRes import QRes
from models.FLS import FLS

from utils.directory_manager import DirectoryManager
from utils.grid import Regular2DGrid
from utils.trainers import AdamTrainer
from utils.metrics import Metrics
from utils.plots import experiment_plot_1d

from solvers.burgers1d import burgers1d_solver


def initial_condition(x):
    u0 = -torch.sin(pi*x)
    return u0


def loss_function(model, training_grid, nu, u0_model=initial_condition):
    x = training_grid.x
    t0 = training_grid.t0
    u0 = u0_model(x)

    u0_pred = model(t0,x)
    ic_weight = 1e2
    loss_ic = torch.mean((u0_pred-u0)**2) * ic_weight

    t = training_grid.t
    x_lb = training_grid.x_lb
    x_ub = training_grid.x_ub

    loss_bc = torch.mean(model(t,x_lb)**2) + torch.mean(model(t,x_ub)**2)

    x_res = training_grid.grid_column_x.requires_grad_()
    t_res = training_grid.grid_column_t.requires_grad_()

    u = model(t_res,x_res)

    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    loss_res = torch.mean((u_t + u*u_x - nu * u_xx)**2)

    return (loss_ic, loss_bc, loss_res)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


def instantiate_model(model_name):
    if model_name == "PINN":
        model = PINN(in_dim=2, out_dim=1, hidden_dim=128,num_layer=4)
    elif model_name == "QRes":
        model = QRes(in_dim=2, out_dim=1, hidden_dim=128,num_layer=4)
    elif model_name == "FLS":
        model = FLS(in_dim=2, out_dim=1, hidden_dim=128,num_layer=4)
    model.apply(init_weights)
    model = model.to(device)
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser('pinn_train_module')
    parser.add_argument('--model', type=str, default='PINN')    
    parser.add_argument('--dir', type=str,default='free_experiment')   
    parser.add_argument('--seed', type=int,default= 42)    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=20000)  
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
        equation="burgers1d",
        model=model_name_str,
        dir_name=dir_name,
        ablation=False,
        specifier=False,
        seed=seed
    )

    directory_manager.config()
    
    domain_x = [-1.0,1.0]
    domain_t = [0.0,1.0]
    
    n_x = 201
    n_t = 201

    xs = torch.linspace(*domain_x,n_x)
    ts = torch.linspace(*domain_t,n_t)

    diffusion = 0.01/pi        # equation diffusion coefficient nu: u_t = -u*u_x + nu*u_xx

    # time domain decomposition
    n_slices = 20
    assert (n_t - 1) % n_slices == 0
    offset = (n_t - 1) // n_slices       # quantity of time steps considered in each sub-network
    time_slices = [offset*i for i in range(1,n_slices+1)]

    model_list  = [instantiate_model(model_name) for _ in range(n_slices)]
    model_list = nn.ModuleList(model_list)

    start_time = time.time()

    current_domain_t = [ts[0].item(),ts[time_slices[0]].item()]
    training_grid = Regular2DGrid(domain_x,current_domain_t,n_x,offset+1).to(device)

    starter_steps = steps 
    trainer = AdamTrainer(params = list(model_list[0].parameters()),
                        adam_lr = lr,
                        adam_steps = starter_steps,
                        learning_rate_scheduler = True,
                        loss_function = lambda : loss_function(model_list[0],training_grid,diffusion)
                    )

    trainer.train()


    for i in range(1,n_slices):
        current_domain_t = [ts[time_slices[i-1]].item(),ts[time_slices[i]].item()]
        training_grid = Regular2DGrid(domain_x,current_domain_t,n_x,offset+1).to(device)
        current_t0 = training_grid.t0

        model_list[i].load_state_dict(model_list[i-1].state_dict()) # each model is initialized as the previous model
        
        # the initial condition is the previous model prediction on its last time step
        for param in model_list[i-1].parameters():
            param.requires_grad = False
        u0_model = lambda x : model_list[i-1](current_t0,x)

        trainer = AdamTrainer(params=list(model_list[i].parameters()),
                            adam_lr=lr,
                            adam_steps=steps,
                            learning_rate_scheduler=True,
                            loss_function=lambda: loss_function(model_list[i], training_grid, diffusion, u0_model)
                        )
        
        trainer.train()
        
        
    end_time = time.time()
    training_time = end_time - start_time


    # predicted solution
    u_pred = []
    basis = FourierBasis1D(n_x,domain_x)
    x = basis.x

    for i in range(0,n_slices):
        if i == 0:
            t = ts[i:time_slices[i]]
        elif i == n_slices - 1:
            t = ts[time_slices[i-1]:time_slices[i]+1]
        else:
            t = ts[time_slices[i-1]:time_slices[i]]
            
        T,X = torch.meshgrid(t,x,indexing='ij')

        t_inf = T.flatten().unsqueeze(1)
        x_inf = X.flatten().unsqueeze(1)
        model_list[i] = model_list[i].to('cpu')
        with torch.no_grad():
            current_u_pred = model_list[i](t_inf,x_inf)
            current_u_pred = current_u_pred.reshape(len(t),n_x)

        u_pred.append(current_u_pred)
    
    u_pred = torch.cat(u_pred,dim=0).cpu().numpy()

    # reference solution
    u_ref = burgers1d_solver(
        n_x=n_x,
        n_t=n_t,
        domain_x=domain_x,
        domain_t=domain_t,
        diffusion=diffusion,
        device=device
    )
        
    # metrics
    metrics = Metrics(u_pred,u_ref)
    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    # save solutions

    np.save(os.path.join(directory_manager.save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(directory_manager.save_dir_solutions, "u_ref.npy"),u_ref)

    metrics_dict = {
            "model": str(model_name_str),
            "relative_l2_error": float(relative_l2_error_value),
            "relative_l1_error": float(relative_l1_error_value),
            "steps": int(steps),
            "training_time": float(training_time)
    }

    with open(os.path.join(directory_manager.save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)    
    
    # plot
    experiment_plot_1d(
        xs=xs,
        ts=ts,
        u_ref=u_ref,
        u_pred=u_pred,
        figs_dir=directory_manager.figs_dir,
        fig_name="burgers1d_plot.png"
    )