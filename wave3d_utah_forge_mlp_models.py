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

from utils.grid import Random4DGrid
from utils.trainers import AdamTrainer
from utils.metrics import Metrics
from utils.plots import plot_prediction_vs_groundtruth


from pseudo_spectral_methods.wave import wave_groundtruth_generator
from scipy.interpolate import RegularGridInterpolator


def velocity_profile(z, a=1.0, b=-0.5, c=1000, d=0.5):
        return a + b * (torch.sigmoid(c * (z - d)))

def neumann_loss(model, x, y, z, t, axis):
    if axis == "x":
        x.requires_grad_()
        u = model(x, y, z, t)
        u_n = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    elif axis == "y":
        y.requires_grad_()
        u = model(x, y, z, t)
        u_n = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    elif axis == "z":
        z.requires_grad_()
        u = model(x, y, z, t)
        u_n = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return torch.mean(u_n**2)

def loss_function(model, training_grid, sigma, c_squared_interp, center, device):
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

    u_ic = model(x_ic,y_ic,z_ic,t_ic)
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

    u = model(x_res,y_res,z_res,t_res)
    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_res, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_z = torch.autograd.grad(u, z_res, grad_outputs=torch.ones_like(u),create_graph=True)[0]
    u_zz = torch.autograd.grad(u_z,z_res,grad_outputs=torch.ones_like(u_z),create_graph=True)[0]
    u_t = torch.autograd.grad(u,t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]


    loss_res = torch.mean((u_tt - (c_squared_torch)*(u_xx+u_yy+u_zz))**2)

    return (1e3*loss_ic, loss_bc, loss_res)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser('pinn_train_module')
    parser.add_argument('--model', type=str, default='PINN')    # options: 'PINN', 'FLS', 'QRes', 'PINNsFormer
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')     
    parser.add_argument('--steps', type=int, default=5000)  
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scenario',type=str,default='first') # first or second scenarios

    args = parser.parse_args()
    model_name = args.model
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr
    scenario = args.scenario


    torch.manual_seed(seed)

    dir_subtitle = "wave3d_utah_forge_"+scenario+"_scenario"
    save_dir = os.path.join("results", dir_subtitle, dir_name, model_name, f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    if scenario == 'first':
        domain_t = [0,3]
        center = (0,0,0)
        normalizing_factor = 10
    elif scenario == 'second':
        domain_t = [0,1.2]
        center = (-3,0,0)
        normalizing_factor = 2

    if model_name == "PINN":
        model = PINN(in_dim=4, out_dim=1)
    elif model_name == "QRes":
        model = QRes(in_dim=4, out_dim=1)
    elif model_name == "FLS":
        model = FLS(in_dim=4, out_dim=1)
    model.apply(init_weights)
    model = model.to(device)

    # domain
    domain_x = [-4,4]
    domain_y = [-2.5,2.5]
    domain_z = [-2.5,2.5]

    n_x = 65
    n_y = 41
    n_z = 41
    n_t = 301
    
    n_ic = 10000
    n_bc = 2000
    n_res = 70000

    training_grid = Random4DGrid(domain_x, domain_y, domain_z, domain_t, n_ic, n_res, n_bc)
    
    sigma = torch.tensor(0.15,dtype=torch.float32).to(device)

    # load velocity field
    data_path = "data/utah_forge/Seismic_Velocity_Model_FORGE_Site"

    c_as_array = []

    with open(data_path,'r') as f:
        next(f)
        for line in f:
            cols = line.split()
            c_as_array.append(float(cols[5])/normalizing_factor)     


    c_numpy = np.array(c_as_array).reshape(n_x,n_y,n_z)
    c = torch.tensor(c_numpy,dtype=torch.float32) # we need this to compute the ground-truth solution below
    c_squared = c_numpy*c_numpy
    xs = np.linspace(*domain_x,n_x)
    ys = np.linspace(*domain_y,n_y)
    zs = np.linspace(*domain_z,n_z)
    c_squared_interp = RegularGridInterpolator((xs,ys,zs),c_squared)

    trainer = AdamTrainer(params=list(model.parameters()),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function=lambda: loss_function(model, training_grid, sigma, c_squared_interp, center, device)
                    )



    start_time = time.time()
    trainer.train()
    end_time = time.time()

    training_time = end_time - start_time

    
    # loading ground-truth and predicted solutions

    # grid and basis
    basis = CosineBasis3D(n_x,n_y,n_z,domain_x,domain_y,domain_z,device=device)
    X,Y,Z = torch.meshgrid(basis.x,basis.y,basis.z, indexing='ij')
    grid = torch.stack((X,Y,Z),dim=-1)
    x = grid[:,:,:,0:1]
    y = grid[:,:,:,1:2]
    z = grid[:,:,:,2:3]

    # time discretization
    t = torch.linspace(*domain_t,n_t)

    u0 = 10*torch.exp((-((x-center[0]).cpu())**2-((y-center[1]).cpu())**2-((z-center[2]).cpu())**2)/(2*sigma.cpu()**2)).squeeze(-1)

    # pseudo-spectral method for generating ground-truth solution
    u_gt = wave_groundtruth_generator(c,u0,basis,t,device)

    model = model.to("cpu")

    # predicted solution inference
    T,X,Y,Z = torch.meshgrid(t,basis.x,basis.y,basis.z,indexing='ij')
    x = X.flatten().unsqueeze(-1)
    y = Y.flatten().unsqueeze(-1)
    z = Z.flatten().unsqueeze(-1)
    t = T.flatten().unsqueeze(-1)

    chunk_size = 10000
    num_points = x.shape[0]
    u_pred_list = []
    with torch.no_grad():
        for i in range(0,num_points,chunk_size):
            x_chunk = x[i:i+chunk_size]
            y_chunk = y[i:i+chunk_size]
            z_chunk = z[i:i+chunk_size]
            t_chunk = t[i:i+chunk_size]
            u_pred_chunk = model(x_chunk,y_chunk,z_chunk,t_chunk)
            
            u_pred_list.append(u_pred_chunk)


        u_pred = torch.cat(u_pred_list,dim=0)
        u_pred = u_pred.squeeze()
        u_pred = u_pred.reshape(n_t,n_x,n_y,n_z)
        u_pred = u_pred.cpu().numpy()
    

    metrics = Metrics(u_pred,u_gt)
    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    save_dir_solutions = os.path.join(save_dir, "solutions")
    os.makedirs(save_dir_solutions,exist_ok=True)

    np.save(os.path.join(save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(save_dir_solutions, "u_gt.npy"),u_gt)

  
    metrics_dict = {
        "relative_l2_error": float(relative_l2_error_value),
        "relative_l1_error": float(relative_l1_error_value),
        "training_time": float(training_time)
    }

    save_dir_metrics = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir_metrics,exist_ok=True)

    with open(os.path.join(save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    
    # slice at z = 0.5
    u_pred_slice = u_pred[:,:,:,24]
    u_gt_slice = u_gt[:,:,:,24]
    
    save_path = os.path.join(figs_dir,"plots.png")

    ts = torch.linspace(*domain_t,n_t)
    plot_prediction_vs_groundtruth(ts=ts,u_pred=u_pred_slice,u_gt=u_gt_slice,extent=[*domain_x,*domain_y],save_path=save_path,cmap="seismic")
    



