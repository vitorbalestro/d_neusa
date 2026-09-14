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
from solvers.allen_cahn_2d import allen_cahn_2d_solver

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


def causal_loss_function(model,training_grid,diffusion,beta,device,n_causal,epsilon):
    training_grid.update_random_grid()
    training_grid.to(device)

    x_ic = training_grid.x_ic
    y_ic = training_grid.y_ic
    t_ic = training_grid.t_ic

    u0 = torch.tanh((0.35 - torch.sqrt((x_ic-0.5)**2+(y_ic-0.5)**2))/(2*0.025))
    
    u_ic = model(t_ic,x_ic,y_ic)

    loss_ic = torch.mean((u_ic-u0)**2) 
    
    loss_bc = (
        neumann_loss(model, training_grid.x_lb_bc, training_grid.y_bc, training_grid.t_bc, "x")
        + neumann_loss(model, training_grid.x_ub_bc, training_grid.y_bc, training_grid.t_bc, "x")
        + neumann_loss(model, training_grid.x_bc, training_grid.y_lb_bc, training_grid.t_bc, "y")
        + neumann_loss(model, training_grid.x_bc, training_grid.y_ub_bc, training_grid.t_bc, "y")
    ) / 4.0

    x_res_all = training_grid.x_res
    y_res_all = training_grid.y_res
    t_res_all = training_grid.t_res

    t_min = training_grid.domain_t[0]
    t_max = training_grid.domain_t[1]

    time_edges = torch.linspace(
        t_min,
        t_max,
        n_causal + 1,
        device=device
    )

    residual_losses = []

    for i in range(n_causal):

        t_left = time_edges[i]
        t_right = time_edges[i + 1]

        if i == n_causal - 1:
            mask = (
                (t_res_all >= t_left)
                &
                (t_res_all <= t_right)
            )
        else:
            mask = (
                (t_res_all >= t_left)
                &
                (t_res_all < t_right)
            )

        mask = mask.squeeze(-1)

        x_res = x_res_all[mask].detach().requires_grad_()
        y_res = y_res_all[mask].detach().requires_grad_()
        t_res = t_res_all[mask].detach().requires_grad_()

        if x_res.numel() == 0:
            residual_losses.append(
                torch.tensor(0.0, device=device)
            )
            continue

        u = model(t_res,x_res,y_res)
        u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y_res, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_t = torch.autograd.grad(u,t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        

        residual = (
            u_t
            - diffusion * (u_xx + u_yy)
            - beta * u
            + beta * u**3
        )

        loss_i = torch.mean(residual**2)

        residual_losses.append(loss_i)

    residual_losses = torch.stack(residual_losses)

    weights = torch.ones_like(residual_losses)

    for i in range(1, n_causal):

        weights[i] = torch.exp(
            -epsilon * torch.sum(residual_losses[:i])
        )

    weights = weights.detach()

    loss_res = torch.mean(
        weights * residual_losses
    )

    return (
        1e3 * loss_ic,
        loss_bc,
        loss_res
    )


def standard_loss_function(model, training_grid, diffusion, beta, device):
    training_grid.update_random_grid()
    training_grid.to(device)

    x_ic = training_grid.x_ic
    y_ic = training_grid.y_ic
    t_ic = training_grid.t_ic

    x_res = training_grid.x_res.detach().requires_grad_()
    y_res = training_grid.y_res.detach().requires_grad_()
    t_res = training_grid.t_res.detach().requires_grad_()
    
    u0 = torch.tanh((0.35 - torch.sqrt((x_ic-0.5)**2+(y_ic-0.5)**2))/(2*0.025))

    u_ic = model(t_ic,x_ic,y_ic)

    loss_ic = torch.mean((u_ic-u0)**2) 
    
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
    

    x_res = x_res.detach().cpu().numpy()
    y_res = y_res.detach().cpu().numpy()
    
    loss_res = torch.mean((u_t - diffusion*(u_xx+u_yy)- beta * u + beta * u**3)**2)

    return (1e3*loss_ic, loss_bc, loss_res)

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Allen-Cahn 2D baseline models')
    parser.add_argument('--model', type=str, default='PINN')    # options: 'PINN', 'FLS', 'QRes', 'PINNsFormer
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')     
    parser.add_argument('--steps', type=int, default=50000)  
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--causal', action='store_true')
    parser.add_argument('--n_causal', type=int, default=20)
    parser.add_argument('--epsilon', type=float, default=1)
          
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
        equation="allen_cahn_2d",
        model=model_name_str,
        dir_name=dir_name,
        ablation=False,
        specifier=False,
        seed=seed
    )

    directory_manager.config()

    # instantiate model
    if model_name == "PINN":
        model = PINN(in_dim=3, out_dim=1, hidden_dim=128,num_layer=4)
    elif model_name == "QRes":
        model = QRes(in_dim=3, out_dim=1, hidden_dim=128,num_layer=4)
    elif model_name == "FLS":
        model = FLS(in_dim=3, out_dim=1, hidden_dim=128,num_layer=4)
    model.apply(init_weights)
    model = model.to(device)

    # equation parameters: u_t = nu*(Delta u) - beta*u**3 + beta*u
    diffusion = 0.00625
    beta = 10

    # domain and grid
    domain_x = [0.0,1.0]
    domain_y = [0.0,1.0]
    domain_t = [0.0,10.0]

    # size of training samples
    n_ic = 2000
    n_bc = 1000
    n_res = 20000

    # training grid
    training_grid = Random3DGrid(domain_x, domain_y, domain_t, n_ic, n_bc, n_res)

    # training
    trainer = AdamTrainer(
                params=list(model.parameters()),
                adam_lr=lr,
                adam_steps=steps,
                learning_rate_scheduler=True,
                loss_function=(
                    lambda: causal_loss_function(model,training_grid,diffusion,beta,device,n_causal,epsilon)
                    if causal
                    else standard_loss_function(model, training_grid, diffusion, beta, device)
                )
            )


    start_time = time.time()
    trainer.train()
    end_time = time.time()

    training_time = end_time - start_time

    # predicted solution 
    n_x = 201
    n_y = 201
    n_t = 201

    basis = CosineBasis2D(n_x,n_y,domain_x,domain_y,device=device)
    xs = basis.x
    ys = basis.y
    
    X, Y = torch.meshgrid(xs, ys, indexing="ij")

    x = X.reshape(-1, 1).to(device)
    y = Y.reshape(-1, 1).to(device)
    ts = torch.linspace(*domain_t,n_t)
    t = torch.empty_like(x).to(device)
    

    outputs = []

    start_time_inf = time.time()
    with torch.no_grad():
        for tk in ts:
            t.fill_(tk)

            u = model(t, x, y)
            outputs.append(u.reshape(n_x, n_y).cpu())

    u_pred = torch.stack(outputs).numpy()
    end_time_inf = time.time()

    inference_time = end_time_inf - start_time_inf
    
    # reference solution
    X,Y = torch.meshgrid(basis.x,basis.y,indexing='ij')
    grid = torch.stack((X,Y),dim=-1)
    xs = grid[:,:,0:1]
    ys = grid[:,:,1:2]
    u0 = torch.tanh((0.35 - torch.sqrt((xs-0.5)**2+(ys-0.5)**2))/(2*0.025))
    u0 = u0.squeeze(-1)
    n_t = 20001

    u_ref = allen_cahn_2d_solver(
        basis=basis,
        n_t=n_t,
        domain_t=domain_t,
        u0=u0,
        diffusion=diffusion,
        beta=beta,
        device=device
    )
    u_ref = u_ref[::100,:,:]

    metrics = Metrics(u_pred,u_ref)
    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

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
    

    experiment_plot_2d(
        ts=ts,
        u_pred=u_pred,
        u_ref=u_ref,
        figs_dir=directory_manager.figs_dir,
        fig_name="allen_cahn_2d_plot.png",
        extent=[*domain_x,*domain_y],
        cmap="coolwarm",
    )





