import torch
from torchdyn.core import NeuralODE
import numpy as np
import time
import argparse
import os
import json

from basis.fourier1d import FourierBasis1D

from training_wrappers.training_wrapper_d_neusa import TrainingWrapperDNeuSA

from models.hadamard_layer import HadamardLayer
from models.mlps import MLP
from models.vector_fields import HeatLikeVectorField

from utils.trainers import AdamTrainer
from utils.metrics import Metrics

from solvers.allen_cahn_1d import allen_cahn_1d_groundtruth_generator

from utils.plots import experiment_plot_1d

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

    parser = argparse.ArgumentParser('Allen-Cahn 1D D-NeuSA')
    parser.add_argument('--dir', type=str, default='free_experiment')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--n_slices', type=int, default=100)
    parser.add_argument('--diffusion', type=float, default=0.0001)

    args = parser.parse_args()
    dir_name = args.dir
    device = args.device
    seed = args.seed
    steps = args.steps
    lr = args.lr
    n_slices = args.n_slices
    diffusion = args.diffusion

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)

    if n_slices != 1:
        save_dir = os.path.join("results","allen_cahn_1d",dir_name,"D_NeuSA",f"seed_{seed}")
    else:
        save_dir = os.path.join("results","allen_cahn_1d",dir_name,"NeuSA",f"seed_{seed}")
    
    # domain, spectral basis and discretizations
    domain_x = [-1.0,1.0]
    n_x = 101
    
    basis = FourierBasis1D(n_x,domain_x)
    xs = basis.x

    domain_t = [0.0,1.0]
    n_t = 1001
    ts = torch.linspace(*domain_t,n_t)

    # initial condition
    u0 = xs**2 * torch.cos(torch.pi*xs)
    hu0 = basis.dbt(u0)

    # initial vector field
    D2 = basis.D2
    omegas = diffusion*torch.diag(D2)
    F0 = HadamardLayer(omegas)

    # neural network parameters
    neural_network_dict = {
        "in_dim": n_x,
        "hidden_dim": 2 * in_dim,
        "out_dim": in_dim,
        "num_hidden_layers": 1
    }

    # optimizer parameters
    optim_dict = {
        "adam_lr": lr,
        "steps": steps,
        "learning_rate_scheduler": True
    }

    # training and inference
    trainer = TrainingWrapperDNeuSA(
        neural_network_dict=neural_network_dict,
        optim_dict=optim_dict,
        ts=ts,
        n_slices=n_slices,
        hu0=hu0,
        basis=basis,
        F0=F0,
        neural_vector_field_builder=HeatLikeVectorField,
        model_weight=1.0,
        loss_function=loss_function,
        device=device
    )

    u_pred = trainer.train()

    # reference solution
    u_ref = allen_cahn_1d_solver(n_x,n_t,domain_x,domain_t)

    # metrics
    metrics = Metrics(u_pred,u_ref)
    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    # save solutions
    save_dir_solutions = os.path.join(save_dir, "solutions")
    os.makedirs(save_dir_solutions,exist_ok=True)

    np.save(os.path.join(save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(save_dir_solutions, "u_ref.npy"),u_gt)

    with open(os.path.join(save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)    
    
    # plot
    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    experiment_plot_1d(xs,ts,u_ref,u_pred,figs_dir)
    