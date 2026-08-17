import torch
from torchdyn.core import NeuralODE
import numpy as np
import time
import argparse
import json
import os

from basis.fourier1d import FourierBasis1D

from training_wrappers.training_wrapper_d_neusa import TrainingWrapperDNeuSA

from models.hadamard_layer import HadamardLayer
from models.mlps import MLP
from models.vector_fields import HeatLikeVectorField

from utils.trainers import AdamTrainer
from utils.metrics import Metrics
from utils.directory_manager import DirectoryManager

from solvers.allen_cahn_1d import allen_cahn_1d_solver

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
    parser.add_argument('--ablation_subdomains',action='store_true')
    parser.add_argument('--n_slices', type=int, default=100)
    parser.add_argument('--ablation_diffusion',action='store_true')
    parser.add_argument('--diffusion', type=float, default=0.0001)
    parser.add_argument('--enhanced_neusa',action='store_true', help="Enables a NeuSA model with number of parameters comparable to that of D-NeuSA.")

    args = parser.parse_args()
    dir_name = args.dir
    device = args.device
    seed = args.seed
    steps = args.steps
    lr = args.lr
    ablation_subdomains = args.ablation_subdomains
    n_slices = args.n_slices
    ablation_diffusion = args.ablation_diffusion
    diffusion = args.diffusion
    enhanced_neusa = args.enhanced_neusa

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)

    assert n_slices != 0
    assert sum([ablation_subdomains,ablation_diffusion,enhanced_neusa]) <= 1

    ablation=False
    specifier=None

    if ablation_subdomains:
        ablation = True
        specifier = f"subdomains_{n_slices}"
    elif ablation_diffusion:
        ablation = True
        specifier = f"diffusion_{diffusion}"
    elif enhanced_neusa:
        ablation = True
        specifier = "enhanced_NeuSA"

    model = "D_NeuSA" if n_slices > 1 else "NeuSA"

    directory_manager = DirectoryManager(
        equation="allen_cahn_1d",
        model=model,
        dir_name=dir_name,
        ablation=ablation,
        specifier=specifier,
        seed=seed
    )

    directory_manager.config()


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
    if n_slices != 1:
        hidden_dim = 2 * n_x
        num_hidden_layers = 1
    elif not enhanced_neusa:
        hidden_dim = 4 * n_x
        num_hidden_layers = 1
    else:
        hidden_dim = 8 * n_x
        num_hidden_layers = 8
        
    neural_network_dict = {
        "in_dim": n_x,
        "hidden_dim": hidden_dim,
        "out_dim": n_x,
        "num_hidden_layers": num_hidden_layers 
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

    np.save(os.path.join(directory_manager.save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(directory_manager.save_dir_solutions, "u_ref.npy"),u_ref)

    metrics_dict = {
            "n_slices": int(n_slices),
            "diffusion_coefficient": float(diffusion),
            "enhanced_neusa": str(enhanced_neusa),
            "relative_l2_error": float(relative_l2_error_value),
            "relative_l1_error": float(relative_l1_error_value),
            "training_time": float(trainer.training_time),
            "num_params": int(trainer.num_params),
            "inference_time": float(trainer.inference_time)
    }

    with open(os.path.join(directory_manager.save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)    
    
    # plot
    experiment_plot_1d(xs,ts,u_ref,u_pred,directory_manager.figs_dir)
    