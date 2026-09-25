import argparse
import torch
import torch.nn as nn
import time
import os
import numpy as np
import json
from math import pi

from models.PINN import PINN
from models.QRes import QRes
from models.FLS import FLS

from utils.directory_manager import DirectoryManager
from utils.grid import Regular2DGrid
from utils.trainers import AdamTrainer
from utils.metrics import Metrics
from utils.transport_plot import transport_plot_1d


def standard_loss_function(model, training_grid, a, c):
    x = training_grid.x
    t = training_grid.t

    t0 = training_grid.t0
    x_lb = training_grid.x_lb.requires_grad_()
    x_ub = training_grid.x_ub.requires_grad_()

    x_res = training_grid.grid_column_x.requires_grad_()
    t_res = training_grid.grid_column_t.requires_grad_()

    u0_pred = model(t0,x)
    u0 = torch.sin(2*(torch.pi)*x) + a * torch.sin(20*(torch.pi)*x)

    loss_ic = 1e2*torch.mean((u0-u0_pred)**2)

    u_b_lb = model(t,x_lb)
    u_b_lb_x = torch.autograd.grad(u_b_lb, x_lb, grad_outputs=torch.ones_like(u_b_lb), create_graph=True)[0]
    
    u_b_ub = model(t,x_ub)
    u_b_ub_x = torch.autograd.grad(u_b_ub, x_ub, grad_outputs=torch.ones_like(u_b_ub), create_graph=True)[0]

    loss_bc = torch.mean((u_b_lb-u_b_ub)**2) + torch.mean((u_b_lb_x-u_b_ub_x)**2)

    u = model(t_res,x_res)
    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    loss_res = torch.mean((u_t + c*u_x)**2)

    return (loss_ic, loss_bc, loss_res)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Allen-Cahn 1D MLP models')
    parser.add_argument('--model', type=str, default='PINN')    # options: 'PINN', 'FLS', 'QRes'
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=5000000)  
    parser.add_argument('--lr', type=float, default=0.001)  
    parser.add_argument('--causal', action='store_true')
    parser.add_argument('--n_causal', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=100)
      
    args = parser.parse_args()
    model_name = args.model
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr
    causal = args.causal
    n_causal = args.n_causal
    epsilon = args.epsilon


    torch.manual_seed(seed)

    model_name_str = model_name
    if causal:
        model_name_str = "Causal_" + model_name_str

    directory_manager = DirectoryManager(
        equation="transport1d",
        model=model_name_str,
        dir_name=dir_name,
        ablation=False,
        specifier=False,
        seed=seed,
        checkpoints=True
    )

    directory_manager.config()

    # instantiate model
    if model_name == "PINN":
        model = PINN(in_dim=2, out_dim=1, hidden_dim=128,num_layer=4)
    elif model_name == "QRes":
        model = QRes(in_dim=2, out_dim=1, hidden_dim=128,num_layer=4)
    elif model_name == "FLS":
        model = FLS(in_dim=2, out_dim=1, hidden_dim=128,num_layer=4)
    model.apply(init_weights)
    model = model.to(device)


    # parameters
    a = 0.3
    c = 1.0

    # domain and grid
    domain_x = [0.0, 1.0]
    n_x = 201
    domain_t = [0.0, 1.0]
    n_t = 201


    training_grid = Regular2DGrid(domain_x, domain_t, n_x, n_t)
    training_grid.to(device)

    checkpoint_steps = [
        1_000,
        2_000,
        5_000,
        10_000,
        20_000,
        50_000,
        100_000,
        200_000,
        500_000,
        1_000_000,
        2_000_000,
        3_000_000,
        4_000_000,
        5_000_000,
    ]

    checkpoints_dir = directory_manager.checkpoints_dir

    # training
    trainer = AdamTrainer(
                params=list(model.parameters()),
                adam_lr = lr,
                adam_steps = steps,
                learning_rate_scheduler=False,
                loss_function=(
                        lambda: standard_loss_function(model,training_grid, a, c)
                ),
                model=model,
                checkpoint_steps=checkpoint_steps,
                checkpoints_dir=checkpoints_dir
            )
    
        
    start_time = time.time()
    trainer.train()                 
    end_time = time.time()

    training_time = end_time - start_time

    # predicted solution 
    xs = torch.linspace(*domain_x,n_x)
    ts = torch.linspace(*domain_t,n_t)
    
    T,X = torch.meshgrid(ts,xs,indexing='ij')

    t_inf = T.flatten().unsqueeze(1).to(device)
    x_inf = X.flatten().unsqueeze(1).to(device)

    start_time_inf = time.time()
    u_pred = model(t_inf,x_inf).reshape(n_t,n_x).detach().cpu().numpy()
    end_time_inf = time.time()

    inference_time = end_time_inf - start_time_inf

    # reference solution
    u_ref = torch.sin(2*pi*(X-c*T)) + a * torch.sin(20*pi*(X-c*T))
    u_ref = u_ref.cpu().numpy()

    # low frequency component
    u_low = torch.sin(2*pi*(X-c*T))
    u_low = u_low.cpu().numpy()
    
    # save solutions
    np.save(os.path.join(directory_manager.save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(directory_manager.save_dir_solutions, "u_ref.npy"),u_ref)

    # metrics
    metrics = Metrics(u_pred, u_ref)
    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()
    
    metrics_dict = {
        "model": model_name_str,
        "n_causal": n_causal if causal else None,
        "epsilon": epsilon if causal else None,
        "relative_l2_error": float(relative_l2_error_value),
        "relative_l1_error": float(relative_l1_error_value),
        "steps": int(steps),
        "training_time": float(training_time),
        "model_inference_time": inference_time
    }

    with open(os.path.join(directory_manager.save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4) 

    # plot
    transport_plot_1d(
        xs=xs,
        ts=ts,
        u_ref=u_ref,
        u_pred=u_pred,
        u_low=u_low,
        figs_dir=directory_manager.figs_dir,
        fig_name="transport1d_plot.png"
    )