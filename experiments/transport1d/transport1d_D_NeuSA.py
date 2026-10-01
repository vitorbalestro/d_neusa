import torch
import numpy as np
import argparse
import json
import os
from math import pi
from basis.fourier1d import FourierBasis1D

from training_wrappers.training_wrapper_d_neusa import TrainingWrapperDNeuSA

from models.hadamard_layer import HadamardLayer
from models.vector_fields import HeatLikeVectorField
from utils.metrics import Metrics
from utils.directory_manager import DirectoryManager

from utils.transport_plot import transport_plot_1d

def loss_function(hu0,ts,NODE,basis,F,F0):

    hu = NODE(hu0,ts)
    Fhu = F(hu)

    Dx_u_hat =  hu @ basis.D
    
    # u_t + u_x = 0
    u_t = basis.idbt(Fhu)
    u_x = basis.idbt(Dx_u_hat)

    loss = torch.mean((u_t + u_x)**2)

    return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Transport 1D D-NeuSA')
    parser.add_argument('--dir', type=str, default='free_experiment')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--n_slices', type=int, default=20)
    parser.add_argument('--discretization_size', type=int, default=201)
    parser.add_argument('--warm_start', action="store_true",help="Enables warm start, where each vector field is initialized with the parameters of the previous one.")

    args = parser.parse_args()
    dir_name = args.dir
    device = args.device
    seed = args.seed
    steps = args.steps
    lr = args.lr
    n_slices = args.n_slices
    discretization_size = args.discretization_size
    warm_start = args.warm_start

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)

    assert n_slices != 0

    ablation=False
    specifier=None

    model = "D_NeuSA" if n_slices > 1 else "NeuSA"

    directory_manager = DirectoryManager(
        equation="transport1d",
        model=model,
        dir_name=dir_name,
        ablation=ablation,
        specifier=specifier,
        seed=seed
    )

    directory_manager.config()

    # parameters
    a = 0.3
    c = 1.0

    # domain, spectral basis and discretizations
    domain_x = [0.0,1.0]
    n_x = 201
    
    basis = FourierBasis1D(n_x,domain_x)
    xs = basis.x

    domain_t = [0.0,1.0]
    n_t = discretization_size
    ts = torch.linspace(*domain_t,n_t)

    # initial condition
    u0 = torch.sin(2*(torch.pi)*xs) + a * torch.sin(20*(torch.pi)*xs)
    hu0 = basis.dbt(u0)

    # initial vector field (no initial vector field)
    F0 = HadamardLayer(torch.zeros(n_x))

    # model architecture and parameters
    if n_slices != 1:
        hidden_dim = 2 * n_x
        num_hidden_layers = 1
        model_weight = 1.0
    else:
        hidden_dim = 2 * n_x
        num_hidden_layers = 0
        model_weight = 1.0
        
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
        model_weight=model_weight,
        space_dimension=1,
        loss_function=loss_function,
        device=device,
        warm_start=warm_start
    )

    u_pred = trainer.train()

    # reference solution
    T,X = torch.meshgrid(ts,xs,indexing='ij')
    u_ref = torch.sin(2*pi*(X-c*T)) + a * torch.sin(20*pi*(X-c*T))
    u_ref = u_ref.cpu().numpy()

    # low frequency component
    u_low = torch.sin(2*pi*(X-c*T))
    u_low = u_low.cpu().numpy()
    
    # save solutions
    np.save(os.path.join(directory_manager.save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(directory_manager.save_dir_solutions, "u_ref.npy"),u_ref)


    # metrics
    metrics = Metrics(u_pred,u_ref)
    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    
    metrics_dict = {
            "model": model,
            "n_slices": int(n_slices),
            "warm_start": str(warm_start),
            "relative_l2_error": float(relative_l2_error_value),
            "relative_l1_error": float(relative_l1_error_value),
            "steps": int(steps),
            "training_time": float(trainer.training_time),
            "num_params": int(trainer.num_params),
            "model_inference_time": float(trainer.inference_time),
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
    