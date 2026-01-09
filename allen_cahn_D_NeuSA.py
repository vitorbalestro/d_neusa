import torch
from torchdyn.core import NeuralODE
import numpy as np
import time
import argparse
import os
import json
from scipy.interpolate import RegularGridInterpolator

from basis.fourier1d import FourierBasis1D
from models.hadamard_layer import HadamardLayer
from models.mlps import MLP
from models.vector_fields import HeatLikeVectorField
from utils.trainers import AdamTrainer
from data_loaders.allen_cahn_data_loader import AllenCahn1DGroundTruthLoader
from utils.metrics import Metrics
from utils.plots import create_graphs_grid, create_time_evolution_gif


def loss_function(hu0,ts,NODE,basis,F,F0):

    hu = NODE(hu0,ts)
    Fhu = F(hu)
    
    # u_t = cu_xx - 5u^3 + 5u
    u = basis.idbt(hu)
    u_t = basis.idbt(Fhu)
    c_u_xx = basis.idbt(F0(hu))

    loss = torch.mean((u_t - c_u_xx + 5*u**3 - 5*u)**2)

    return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Allen-Cahn D-NeuSA')
    parser.add_argument('--dir', type=str, default='free_experiment')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.02)

    args = parser.parse_args()
    dir_name = args.dir
    device = args.device
    seed = args.seed
    steps = args.steps
    lr = args.lr

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)

    save_dir = os.path.join("results","allen_cahn",dir_name,"D_NeuSA",f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    # domain, grids and basis
    domain_x = [-1.0,1.0]
    M = 101

    basis = FourierBasis1D(M,domain_x)
    xs = basis.x

    domain_t = [0.0,1.0]
    time_steps = 1001
    ts = torch.linspace(*domain_t,time_steps)

    # initial condition
    u0 = xs**2 * torch.cos(torch.pi*xs)
    hu0 = basis.dbt(u0)

    # initial vector field
    D2 = basis.D2
    omegas = 0.0001*torch.diag(D2)
    F0 = HadamardLayer(omegas)

    # time domain decomposition
    n_slices = 100
    assert (time_steps - 1) % n_slices == 0
    offset = (time_steps - 1) // n_slices
    time_slices = [offset*i for i in range(1,n_slices+1)]
    
    # neural networs and NODEs
    in_dim = len(xs)
    hidden_dim = 2 * in_dim
    out_dim = in_dim
    num_hidden_layers = 1

    
    NODEs = []
    Fs = []

    for _ in range(n_slices):
        F_neural = MLP(in_dim,hidden_dim,out_dim,num_hidden_layers)
        F = HeatLikeVectorField(F_neural,F0,model_weight=1.0)
        NODE = NeuralODE(F,solver='rk4',sensitivity='autograd',return_t_eval=False)

        F = F.to(device)
        NODE = NODE.to(device)

        Fs.append(F)
        NODEs.append(NODE)
    
    # training
    hus = []

    basis = basis.to(device)
    hu0 = hu0.to(device)
    ts = ts.to(device)
    F0 = F0.to(device)

    start_time = time.time()

    trainer = AdamTrainer(NODEs[0].parameters(),
                        adam_lr = lr,
                        adam_steps = steps,
                        learning_rate_scheduler=True,
                        loss_function = lambda: loss_function(hu0,ts[0:time_slices[0]+1],NODEs[0],basis,Fs[0],F0)
                    )
    

   
    trainer.train_neusa()

    hus.append(NODEs[0](hu0,ts[0:time_slices[0]+1]))

    for i in range(1,n_slices):
        hu_init = hus[i-1][-1].detach()
        hus[i-1] = hus[i-1][:-1,:]
        trainer = AdamTrainer(NODEs[i].parameters(),
                            adam_lr = lr,
                            adam_steps = steps,
                            learning_rate_scheduler=True,
                            loss_function = lambda: loss_function(hu_init,ts[time_slices[i-1]:time_slices[i]+1],NODEs[i],basis,Fs[i],F0)
        )

        
        trainer.train_neusa()
        
        hus.append(NODEs[i](hu_init,ts[time_slices[i-1]:time_slices[i]+1]))
    
    end_time = time.time()
    training_time = end_time - start_time

    # retrieve model solution (and copy the left boundary solution to the right boundary)
    hu = torch.cat(hus,dim=0)
    u_model = basis.idbt(hu).detach().to('cpu').numpy()
    u_model = np.concatenate([u_model, u_model[:,[0]]],axis=-1)
    
    # construct an interpolator based on the model solution
    x_right = torch.tensor(domain_x[-1],device=xs.device).unsqueeze(0)
    xs = torch.cat([xs,x_right]).numpy()
    ts = ts.cpu().numpy()
    interpolator = RegularGridInterpolator((ts,xs),u_model)
    
    # load ground-truth (solution and grid)
    data_loader = AllenCahn1DGroundTruthLoader()

    u_gt = data_loader.get_ground_truth_solution()
    x_gt,t_gt = data_loader.get_ground_truth_grid()
    n_x_gt, n_t_gt = data_loader.get_grid_size()

    
    T,X = np.meshgrid(t_gt,x_gt,indexing='ij')
    points = np.stack([T.ravel(),X.ravel()], axis=-1)

    # interpolate the model solution for the ground-truth grid
    u_pred = interpolator(points).reshape(n_t_gt,n_x_gt)

    # compute and save metrics
    metrics= Metrics(u_pred,u_gt)

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
    

    # #### PLOTS ####

    offset = n_t_gt // 3
    u_plot = [u_gt[0,:], u_gt[offset,:], u_gt[2 * offset,:], u_gt[n_t_gt-1,:]]
    u_pred_plot = [u_pred[0,:], u_pred[offset,:], u_pred[2 * offset,:], u_pred[n_t_gt-1,:]]

    x_plot = x_gt
    titles = [f"t={t_gt[0].item()}", f"t={t_gt[offset].item()}", f"t={t_gt[2 * offset].item()}", f"t={t_gt[n_t_gt-1].item()}"]

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    create_graphs_grid(x_plot,u_plot,u_pred_plot,titles,figs_dir)
    create_time_evolution_gif(x_gt,t_gt,u_gt,u_pred,figs_dir)

    