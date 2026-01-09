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

from data_loaders.wave2d_three_layers_increasing_data_loader import Wave2DThreeLayersIncreasingGroundTruthLoader

from utils.grid import Random3DGrid
from utils.trainers import AdamTrainer
from utils.metrics import Metrics

from utils.plots import plot_prediction_vs_groundtruth
from scipy.interpolate import RegularGridInterpolator


def loss_function(model, training_grid, sigma, device):
    training_grid.update_random_grid()
    training_grid.to(device)

    x_ic = training_grid.x_ic
    y_ic = training_grid.y_ic
    t_ic = training_grid.t_ic.detach().requires_grad_()

    x_res = training_grid.x_res.detach().requires_grad_()
    y_res = training_grid.y_res.detach().requires_grad_()
    t_res = training_grid.t_res.detach().requires_grad_()
    
    u0 = torch.exp(-(x_ic**2+y_ic**2)/(2*sigma**2))

    u_ic = model(x_ic,y_ic,t_ic)
    u_t_ic = torch.autograd.grad(u_ic, t_ic, grad_outputs=torch.ones_like(u_ic), create_graph=True)[0]

    loss_ic = torch.mean((u_ic-u0)**2) + torch.mean((u_t_ic)**2)
    
    loss_bc = torch.tensor(0.0)

    c = 1.0 + 0.25*(torch.sigmoid(1000*(y_res-0.5)) + torch.sigmoid(1000*(y_res-0.5-0.5)))


    u = model(x_res,y_res,t_res)
    u_x = torch.autograd.grad(u, x_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_res, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y_res, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    u_t = torch.autograd.grad(u,t_res, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t_res, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]


    loss_res = torch.mean((u_tt - (c**2)*(u_xx+u_yy))**2)

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

    save_dir = os.path.join("results", "wave2d_three_layers_increasing", dir_name, model_name, f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    if model_name == "PINN":
        model = PINN(in_dim=3, out_dim=1)
    elif model_name == "QRes":
        model = QRes(in_dim=3, out_dim=1)
    elif model_name == "FLS":
        model = FLS(in_dim=3, out_dim=1)
    model.apply(init_weights)
    model = model.to(device)

    domain_x = [-2.0,2.0]
    domain_y = [-2.0,2.0]
    domain_t = [0.0,2.0]
    
    n_ic = 2000
    n_bc = 1000
    n_res = 20000
    training_grid = Random3DGrid(domain_x, domain_y, domain_t, n_ic, n_bc, n_res)
    
    sigma = torch.tensor(0.1,dtype=torch.float32).to(device)
    
    trainer = AdamTrainer(params=list(model.parameters()),
                    adam_lr=lr,
                    adam_steps=steps,
                    learning_rate_scheduler=True,
                    loss_function=lambda: loss_function(model, training_grid, sigma, device)
                    )


    start_time = time.time()
    trainer.train()
    end_time = time.time()

    training_time = end_time - start_time

    data_loader = Wave2DThreeLayersIncreasingGroundTruthLoader()
    u_gt = data_loader.get_ground_truth_solution()
    x,y,t = data_loader.get_ground_truth_grid_as_tensors()

    with torch.no_grad():

        x = x.to("cuda:0")
        y = y.to("cuda:0")
        t = t.to("cuda:0")
        model = model.to("cuda:0")
        u_pred = model(x,y,t)
        u_pred = u_pred.squeeze().cpu().numpy()

    


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
        "training_time": training_time,
    }

    save_dir_metrics = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir_metrics,exist_ok=True)

    with open(os.path.join(save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)




