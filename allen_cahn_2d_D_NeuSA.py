import torch
import numpy as np
import time
import argparse
import os
import time
import json

from basis.cosine2d import CosineBasis2D


from training_wrappers.training_wrapper_d_neusa import TrainingWrapperDNeuSA

from models.hadamard_layer import HadamardLayer
from models.vector_fields import HeatLikeVectorField

from utils.metrics import Metrics
from utils.directory_manager import DirectoryManager

from solvers.allen_cahn_2d import allen_cahn_2d_solver

from utils.plots import experiment_plot_2d

def loss_function(hu0,ts,NODE,basis,F,F0,beta):

    hu = NODE(hu0,ts)
    Fhu = F(hu)

    u = basis.idbt(hu)
    u_t = basis.idbt(Fhu)
    nu_laplacian = basis.idbt(F0(hu))

    loss = torch.mean((u_t - nu_laplacian + beta*u**3 - beta*u)**2)

    return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Allen-Cahn 2D D-NeuSA")
    parser.add_argument('--dir', type=str, default='free_experiment')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--ablation_subdomains',action='store_true')
    parser.add_argument('--n_slices', type=int, default=1000)
    parser.add_argument('--ablation_diffusion',action='store_true')
    parser.add_argument('--diffusion', type=float, default=0.00625)
    parser.add_argument('--discretization_size', type=int, default=20001)

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
    discretization_size = args.discretization_size

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)

    assert n_slices != 0
    assert sum([ablation_subdomains,ablation_diffusion]) <= 1

    # beta parameter (u_t = diffusion * Delta u - beta * u^3 + beta * u)
    beta = 10

    # directory configuration
    ablation=False
    specifier=None

    if ablation_subdomains:
        ablation = True
        specifier = f"subdomains_{n_slices}"
    elif ablation_diffusion:
        ablation = True
        specifier = f"diffusion_{diffusion}"

    model = "D_NeuSA" if n_slices > 1 else "NeuSA"
    
    directory_manager = DirectoryManager(
        equation="allen_cahn_2d",
        model=model,
        dir_name=dir_name,
        ablation=ablation,
        specifier=specifier,
        seed=seed
    )

    directory_manager.config()

    # domain, spectral basis and discretizations
    domain_x = [0.0,1.0]
    n_x = 201
    domain_y = [0.0,1.0]
    n_y = 201
    
    basis = CosineBasis2D(n_x,n_y,domain_x,domain_y)
    X,Y = torch.meshgrid(basis.x,basis.y,indexing='ij')
    grid = torch.stack((X,Y),dim=-1)
    xs = grid[:,:,0]
    ys = grid[:,:,1]
    
    domain_t = [0.0,10.0]
    n_t = discretization_size
    ts = torch.linspace(*domain_t,n_t)

    # initial condition
    u0 = torch.tanh((0.35 - torch.sqrt((xs-0.5)**2+(ys-0.5)**2))/(2*0.025))
    hu0 = basis.dbt(u0)

    # spectral laplacian operator
    D2 = basis.D2
    omegas = diffusion * D2
    F0 = HadamardLayer(omegas)

    # model architecture and parameters
    M = n_x
    N = n_y
    widen_factor = 2
    num_hidden_layers = 0
    model_weight=1.0

    neural_network_dict = {
        "M": M,
        "N": N,
        "widen_factor": widen_factor,
        "num_hidden_layers": num_hidden_layers
    }

    # optimizer parameters
    optim_dict = {
        "adam_lr": lr,
        "steps": steps,
        "learning_rate_scheduler": True
    }

    # loss function kwargs
    loss_kwargs = {
        'beta': beta
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
        model_weight=model_weight,
        space_dimension=2,
        loss_function=loss_function,
        loss_kwargs=loss_kwargs,
        device=device
    )

    u_pred = trainer.train()

    # reference solution
    start_time_solver = time.time()
    u_ref = allen_cahn_2d_solver(
        basis=basis,
        n_t=n_t,
        domain_t=domain_t,
        u0=u0,
        diffusion=diffusion,
        beta=beta,
        device=device
    )
    end_time_solver = time.time()
    solver_time = end_time_solver - start_time_solver

    # metrics
    metrics = Metrics(u_pred,u_ref)
    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    # save solutions

    np.save(os.path.join(directory_manager.save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(directory_manager.save_dir_solutions, "u_ref.npy"),u_ref)

    metrics_dict = {
            "model": model,
            "n_slices": int(n_slices),
            "diffusion_coefficient": float(diffusion),
            "relative_l2_error": float(relative_l2_error_value),
            "relative_l1_error": float(relative_l1_error_value),
            "training_time": float(trainer.training_time),
            "num_params": int(trainer.num_params),
            "model_inference_time": float(trainer.inference_time),
            "solver_inference_time": float(solver_time)
    }

    with open(os.path.join(directory_manager.save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)    


    # plot
    experiment_plot_2d(
        ts=ts,
        u_pred=u_pred,
        u_ref=u_ref,
        figs_dir=directory_manager.figs_dir,
        fig_name="allen_cahn_2d_plot.png",
        extent=[*domain_x,*domain_y],
        cmap="coolwarm",
    )