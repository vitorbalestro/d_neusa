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

from basis.cosine3d import CosineBasis3D

from utils.directory_manager import DirectoryManager
from utils.grid import Random4DGrid
from utils.trainers import AdamTrainer
from utils.metrics import Metrics
from utils.plots import experiment_plot_2d


from solvers.wave import wave_solver
from scipy.interpolate import RegularGridInterpolator


def neumann_loss(model, x, y, z, t, axis):
    if axis == "x":
        x.requires_grad_()
        u = model(t, x, y, z)
        u_n = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    elif axis == "y":
        y.requires_grad_()
        u = model(t, x, y, z)
        u_n = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    elif axis == "z":
        z.requires_grad_()
        u = model(t, x, y, z)
        u_n = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return torch.mean(u_n**2)

def causal_loss_function(model, training_grid, sigma, c_squared_interp, center, device, n_causal, epsilon):
    training_grid.update_random_grid()
    training_grid.to(device)

    x_ic = training_grid.x_ic
    y_ic = training_grid.y_ic
    z_ic = training_grid.z_ic
    t_ic = training_grid.t_ic.detach().requires_grad_()

    u0 = 10*torch.exp(-((x_ic-center[0])**2+(y_ic-center[1])**2+(z_ic-center[2])**2)/(2*sigma**2))
    
    u_ic = model(t_ic,x_ic,y_ic,z_ic)
    u_t_ic = torch.autograd.grad(u_ic, t_ic, grad_outputs=torch.ones_like(u_ic), create_graph=True)[0]

    loss_ic = torch.mean((u_ic-u0)**2) + torch.mean((u_t_ic)**2)
    
    bc_x0_loss = neumann_loss(model, training_grid.x_bc_x0, training_grid.y_bc_x0, training_grid.z_bc_x0, training_grid.t_bc_x0, axis="x")
    bc_x1_loss = neumann_loss(model, training_grid.x_bc_x1, training_grid.y_bc_x1, training_grid.z_bc_x1, training_grid.t_bc_x1, axis="x")
    bc_y0_loss = neumann_loss(model, training_grid.x_bc_y0, training_grid.y_bc_y0, training_grid.z_bc_y0, training_grid.t_bc_y0, axis="y")
    bc_y1_loss = neumann_loss(model, training_grid.x_bc_y1, training_grid.y_bc_y1, training_grid.z_bc_y1, training_grid.t_bc_y1, axis="y")
    bc_z0_loss = neumann_loss(model, training_grid.x_bc_z0, training_grid.y_bc_z0, training_grid.z_bc_z0, training_grid.t_bc_z0, axis="z")
    bc_z1_loss = neumann_loss(model, training_grid.x_bc_z1, training_grid.y_bc_z1, training_grid.z_bc_z1, training_grid.t_bc_z1, axis="z")
    loss_bc = (bc_x0_loss + bc_x1_loss + bc_y0_loss + bc_y1_loss + bc_z0_loss + bc_z1_loss) / 6.0

    x_res_all = training_grid.x_res
    y_res_all = training_grid.y_res
    z_res_all = training_grid.z_res
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
        z_res = z_res_all[mask].detach().requires_grad_()
        t_res = t_res_all[mask].detach().requires_grad_()

        if x_res.numel() == 0:
                    residual_losses.append(
                        torch.tensor(0.0, device=device)
                    )
                    continue
        

        x_res_np = x_res.detach().clone().cpu().numpy()
        y_res_np = y_res.detach().clone().cpu().numpy()
        z_res_np = z_res.detach().clone().cpu().numpy()
        points = np.stack([x_res_np,y_res_np,z_res_np],axis=-1)
    
        c_squared = c_squared_interp(points)
    
        c_squared_torch = torch.tensor(c_squared,dtype=torch.float32,device=device)

        
        u = model(t_res,x_res,y_res,z_res)
        u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y_res, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
        u_z = torch.autograd.grad(u, z_res, grad_outputs=torch.ones_like(u),create_graph=True)[0]
        u_zz = torch.autograd.grad(u_z,z_res,grad_outputs=torch.ones_like(u_z),create_graph=True)[0]
        u_t = torch.autograd.grad(u,t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    
        residual = u_tt - (c_squared_torch)*(u_xx+u_yy+u_zz)
        
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
        1e2 * loss_ic,
        loss_bc,
        loss_res
    )

    
def standard_loss_function(model, training_grid, sigma, c_squared_interp, center, device):
    training_grid.update_random_grid()
    training_grid.to(device)

    x_ic = training_grid.x_ic
    y_ic = training_grid.y_ic
    z_ic = training_grid.z_ic
    t_ic = training_grid.t_ic.detach().requires_grad_()

    x_res = training_grid.x_res.detach().requires_grad_()
    y_res = training_grid.y_res.detach().requires_grad_()
    z_res = training_grid.z_res.detach().requires_grad_()
    t_res = training_grid.t_res.detach().requires_grad_()
    
    u0 = 10*torch.exp(-((x_ic-center[0])**2+(y_ic-center[1])**2+(z_ic-center[2])**2)/(2*sigma**2))

    u_ic = model(t_ic,x_ic,y_ic,z_ic)
    u_t_ic = torch.autograd.grad(u_ic, t_ic, grad_outputs=torch.ones_like(u_ic), create_graph=True)[0]

    loss_ic = torch.mean((u_ic-u0)**2) + torch.mean((u_t_ic)**2)
    
    bc_x0_loss = neumann_loss(model, training_grid.x_bc_x0, training_grid.y_bc_x0, training_grid.z_bc_x0, training_grid.t_bc_x0, axis="x")
    bc_x1_loss = neumann_loss(model, training_grid.x_bc_x1, training_grid.y_bc_x1, training_grid.z_bc_x1, training_grid.t_bc_x1, axis="x")
    bc_y0_loss = neumann_loss(model, training_grid.x_bc_y0, training_grid.y_bc_y0, training_grid.z_bc_y0, training_grid.t_bc_y0, axis="y")
    bc_y1_loss = neumann_loss(model, training_grid.x_bc_y1, training_grid.y_bc_y1, training_grid.z_bc_y1, training_grid.t_bc_y1, axis="y")
    bc_z0_loss = neumann_loss(model, training_grid.x_bc_z0, training_grid.y_bc_z0, training_grid.z_bc_z0, training_grid.t_bc_z0, axis="z")
    bc_z1_loss = neumann_loss(model, training_grid.x_bc_z1, training_grid.y_bc_z1, training_grid.z_bc_z1, training_grid.t_bc_z1, axis="z")
    loss_bc = (bc_x0_loss + bc_x1_loss + bc_y0_loss + bc_y1_loss + bc_z0_loss + bc_z1_loss) / 6.0

    x_res_np = x_res.detach().clone().cpu().numpy()
    y_res_np = y_res.detach().clone().cpu().numpy()
    z_res_np = z_res.detach().clone().cpu().numpy()
    points = np.stack([x_res_np,y_res_np,z_res_np],axis=-1)

    c_squared = c_squared_interp(points)

    c_squared_torch = torch.tensor(c_squared,dtype=torch.float32,device=device)

    u = model(t_res,x_res,y_res,z_res)
    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_res, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z_res, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z,z_res,grad_outputs=torch.ones_like(u_z),create_graph=True)[0]
    u_t = torch.autograd.grad(u,t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]


    loss_res = torch.mean((u_tt - (c_squared_torch)*(u_xx+u_yy+u_zz))**2)

    return (1e2*loss_ic, loss_bc, loss_res)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Wave 3D Utah FORGE MLP models')
    parser.add_argument('--model', type=str, default='PINN')    
    parser.add_argument('--dir', type=str,default='free_experiment')    
    parser.add_argument('--seed', type=int,default= 42)    
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
        equation="wave3d_utah_forge",
        model=model_name_str,
        dir_name=dir_name,
        ablation=False,
        specifier=False,
        seed=seed
    )

    directory_manager.config()

    # instantiate model
    if model_name == "PINN":
        model = PINN(in_dim=4, out_dim=1, hidden_dim=128,num_layer=4)
    elif model_name == "QRes":
        model = QRes(in_dim=4, out_dim=1, hidden_dim=128,num_layer=4)
    elif model_name == "FLS":
        model = FLS(in_dim=4, out_dim=1, hidden_dim=128,num_layer=4)
    model.apply(init_weights)
    model = model.to(device)

    # domain and grid
    domain_x = [-2,2]
    domain_y = [-2,2]
    domain_z = [-2,2]
    domain_t = [0,3]

    n_x = 41
    n_y = 41
    n_z = 41
    n_t = 301

    # size of the training samples
    n_ic = 10000
    n_bc = 2000
    n_res = 70000

    # training grid
    training_grid = Random4DGrid(domain_x, domain_y, domain_z, domain_t, n_ic, n_res, n_bc)

    # initial condition parameters
    center = (-1,0,0)
    sigma = torch.tensor(0.15,dtype=torch.float32).to(device)
    
    # load velocity profile (and create interpolator)
    normalizing_factor = 10
    data_path = "data/utah_forge/Seismic_Velocity_Model_FORGE_Site"

    c_as_array = []

    with open(data_path,'r') as f:
        next(f)
        for line in f:
            cols = line.split()
            c_as_array.append(float(cols[5])/normalizing_factor)     


    c_numpy = np.array(c_as_array).reshape(65,41,41)
    c_numpy = c_numpy[0:41,:,:]
    c = torch.tensor(c_numpy,dtype=torch.float32) # we need this to compute the ground-truth solution below
    c_squared = c_numpy*c_numpy
    xs = np.linspace(*domain_x,n_x)
    ys = np.linspace(*domain_y,n_y)
    zs = np.linspace(*domain_z,n_z)
    c_squared_interp = RegularGridInterpolator((xs,ys,zs),c_squared)

    trainer = AdamTrainer(
                params=list(model.parameters()),
                adam_lr=lr,
                adam_steps=steps,
                learning_rate_scheduler=True,
                loss_function=(
                    lambda: causal_loss_function(model, training_grid, sigma, c_squared_interp, center, device, n_causal, epsilon)
                    if causal
                    else standard_loss_function(model, training_grid, sigma, c_squared_interp, center, device)
                )
            )



    start_time = time.time()
    trainer.train()
    end_time = time.time()

    training_time = end_time - start_time

    
    # grid for solutions
    basis = CosineBasis3D(n_x,n_y,n_z,domain_x,domain_y,domain_z,device=device)
    X,Y,Z = torch.meshgrid(basis.x,basis.y,basis.z, indexing='ij')
    grid = torch.stack((X,Y,Z),dim=-1)
    xs = grid[:,:,:,0]
    ys = grid[:,:,:,1]
    zs = grid[:,:,:,2]
    ts = torch.linspace(*domain_t,n_t)

    # predicted solution
    x = X.reshape(-1, 1).to(device)
    y = Y.reshape(-1, 1).to(device)
    z = Z.reshape(-1, 1).to(device)
    ts = torch.linspace(*domain_t,n_t)
    t = torch.empty_like(x).to(device)

    outputs = []
    
    start_time_inf = time.time()
    with torch.no_grad():
        for tk in ts:
            t.fill_(tk)

            u = model(t, x, y, z)
            outputs.append(u.reshape(n_x, n_y, n_z).cpu())

    u_pred = torch.stack(outputs).numpy()
    end_time_inf = time.time()

    inference_time = end_time_inf - start_time_inf

    # initial condition
    u0 = 10*torch.exp((-((xs-center[0]).cpu())**2-((ys-center[1]).cpu())**2-((zs-center[2]).cpu())**2)/(2*sigma.cpu()**2))

    # reference solution
    u_ref = wave_solver(c,u0,basis,ts,device)

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

    # we plot a slice at z = 0.5
    experiment_plot_2d(
        ts=ts,
        u_pred=u_pred[:,:,:,24],
        u_ref=u_ref[:,:,:,24],
        figs_dir=directory_manager.figs_dir,
        fig_name="wave3d_utah_forge_plot.png",
        extent=[*domain_x,*domain_y],
        cmap="seismic",
        transpose=False,
        interpolation='bicubic'
    )
    
    



