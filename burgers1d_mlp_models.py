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

from utils.grid import Regular2DGrid
from utils.trainers import AdamTrainer
from utils.metrics import Metrics
from utils.plots import create_graphs_grid, create_time_evolution_gif


from data_loaders.burgers1d_data_loader import Burgers1DGroundTruthLoader

def loss_function(model, training_grid, nu):
    x = training_grid.x
    t0 = training_grid.t0
    u0 = -torch.sin(pi*x)

    loss_ic = torch.mean((model(x,t0)-u0)**2)

    t = training_grid.t
    x_lb = training_grid.x_lb
    x_ub = training_grid.x_ub

    loss_bc = torch.mean(model(x_lb,t)**2) + torch.mean(model(x_ub,t)**2)

    x_res = training_grid.grid_column_x.requires_grad_()
    t_res = training_grid.grid_column_t.requires_grad_()

    u = model(x_res,t_res)

    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_t = torch.autograd.grad(u, t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    loss_res = torch.mean((u_t + u*u_x - nu * u_xx)**2)

    return (loss_ic, loss_bc, loss_res)


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser('pinn_train_module')
    parser.add_argument('--model', type=str, default='PINN')    # options: 'PINN', 'FLS', 'QRes'
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
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

    save_dir = os.path.join("results", "burgers1d", dir_name, model_name, f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    # initialize model
    if model_name == "PINN":
        model = PINN(in_dim=2, out_dim=1)
    elif model_name == "QRes":
        model = QRes(in_dim=2, out_dim=1)
    elif model_name == "FLS":
        model = FLS(in_dim=2, out_dim=1)
    model.apply(init_weights)
    model = model.to(device)

    # domain and grid
    domain_x = [-1.0,1.0]
    domain_t = [0.0,1.0]
    n_x = 201
    n_t = 201
    training_grid = Regular2DGrid(domain_x, domain_t, n_x,n_t)
    training_grid = training_grid.to(device)

    # diffusion constat
    nu = 0.01/pi

    # training
    trainer = AdamTrainer(params=list(model.parameters()),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function=lambda: loss_function(model, training_grid, nu)
                    )


    start_time = time.time()
    trainer.train()
    end_time = time.time()

    training_time = end_time - start_time

    # load ground-truth data
    data_loader = Burgers1DGroundTruthLoader(model_name)

    x_gt,t_gt = data_loader.get_ground_truth_grid()
    u_gt = data_loader.get_ground_truth_solution()

    n_x_gt, n_t_gt = data_loader.get_grid_size()

    x,t = data_loader.get_ground_truth_grid_as_tensors()

    # get predicted solution
    model = model.to('cpu')
    with torch.no_grad():
        u_pred = model(x,t).squeeze().numpy()

    metrics = Metrics(u_pred,u_gt)
    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    save_dir_solutions = os.path.join(save_dir, "solutions")
    os.makedirs(save_dir_solutions,exist_ok=True)

    np.save(os.path.join(save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(save_dir_solutions, "u_gt.npy"),u_gt)

    
    metrics_dict = {
        "relative_l2_error": relative_l2_error_value,
        "relative_l1_error": relative_l1_error_value,
        "training_time": training_time
    }

    save_dir_metrics = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir_metrics,exist_ok=True)

    with open(os.path.join(save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # #### PLOTS ####
    u_gt = u_gt.reshape(n_t_gt, n_x_gt)
    u_pred = u_pred.reshape(n_t_gt, n_x_gt)

    u_plot = [u_gt[0,:], u_gt[25,:], u_gt[50,:], u_gt[99,:]]
    u_pred_plot = [u_pred[0,:], u_pred[25,:], u_pred[50,:], u_pred[99,:]]

    x_plot = x_gt
    titles = [f"t={t_gt[0].item()}", f"t={t_gt[25].item()}", f"t={t_gt[50].item()}", f"t={t_gt[99].item()}"]

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    create_graphs_grid(x_plot,u_plot,u_pred_plot,titles,figs_dir)
    create_time_evolution_gif(x_gt,t_gt,u_gt,u_pred,figs_dir)