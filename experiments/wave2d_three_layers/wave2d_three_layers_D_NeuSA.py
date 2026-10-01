import torch
import numpy as np
import time
import json
import argparse
import os

from basis.cosine2d import CosineBasis2D

from training_wrappers.training_wrapper_d_neusa import TrainingWrapperDNeuSA

from models.hadamard_layer import HadamardLayer
from models.vector_fields import WaveLikeVectorField

from utils.metrics import Metrics
from utils.directory_manager import DirectoryManager
from utils.plots import experiment_plot_2d

from solvers.wave import wave_solver

def loss_function(hu0,ts,NODE,basis,F,F0,c):

    huv = NODE(hu0,ts)
    Fhuv = F(huv)

    hu, _ = huv.chunk(2, dim=-1)
    _, Fhv = Fhuv.chunk(2, dim=-1)

    right_hand_side = c**2 * basis.idbt(F0(hu))
    v_t = basis.idbt(Fhv)

    loss = torch.mean((v_t-right_hand_side)**2)

    return loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Wave 2D Three Layers D-NeuSA')
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')   
    parser.add_argument('--steps', type=int, default=200)  
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--ablation_subdomains',action='store_true')
    parser.add_argument('--n_slices', type=int, default=40)
    parser.add_argument('--enhanced_neusa', action='store_true')
    parser.add_argument('--warm_start', action="store_true",help="Enables warm start, where each vector field is initialized with the parameters of the previous one.")

    args = parser.parse_args()
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr
    n_slices = args.n_slices
    ablation_subdomains = args.ablation_subdomains
    enhanced_neusa = args.enhanced_neusa
    warm_start = args.warm_start

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)

    assert n_slices != 0
    assert sum([ablation_subdomains,enhanced_neusa]) <= 1

    # directory configuration
    ablation=False
    specifier=None

    if ablation_subdomains:
        ablation = True
        specifier = f"subdomains_{n_slices}"
    elif enhanced_neusa:
        ablation = True
        specifier = "enhanced_NeuSA"

    model = "D_NeuSA" if n_slices > 1 else "NeuSA"
    
    directory_manager = DirectoryManager(
        equation="wave2d_three_layers",
        model=model,
        dir_name=dir_name,
        ablation=ablation,
        specifier=specifier,
        seed=seed
    )

    directory_manager.config()

    # reference domain
    domain_x = [-2.0,2.0]
    n_x = 101
    domain_y = [-2.0,2.0]
    n_y = 101
    domain_t = [0.0,2.0]
    n_t = 201
    ts = torch.linspace(*domain_t,n_t)

    # extended domain for training
    domain_x_ext = [-4.0,4.0]
    n_x_ext = 201
    domain_y_ext = [-4.0,4.0]
    n_y_ext = 201

    basis = CosineBasis2D(
        M=n_x_ext,
        N=n_y_ext,
        domain_x=domain_x_ext,
        domain_y=domain_y_ext,
        device=device
    )
    X,Y = torch.meshgrid(basis.x,basis.y,indexing='ij')
    grid = torch.stack((X,Y),dim=-1)
    xs = grid[:,:,0]
    ys = grid[:,:,1]

    # initial condition
    sigma = 0.1
    u0 = torch.exp(-((xs)**2+(ys)**2)/(2*sigma**2))
    hu0 = basis.dbt(u0)     
    hv0 = torch.zeros_like(hu0)
    huv0 = torch.cat([hu0,hv0],dim=-1)

    # velocity field
    c = (1.0 + 0.25 * (torch.sigmoid(1000 * (ys - 0.5)) + torch.sigmoid(1000 * (ys - 1.0)))).to(device)

    # spectral laplacian operator
    D2 = basis.D2
    F0 = HadamardLayer(D2)

    # model architecture and parameters
    M = n_x_ext
    N = n_y_ext
    if not enhanced_neusa:
        num_hidden_layers = 0
        widen_factor = 2
        model_weight=1.0
    else:
        num_hidden_layers=4
        widen_factor=4
        model_weight=1.0

    neural_network_dict={
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
        'c': c
    }

    # training and inference
    trainer = TrainingWrapperDNeuSA(
        neural_network_dict=neural_network_dict,
        optim_dict=optim_dict,
        ts=ts,
        n_slices=n_slices,
        hu0=huv0,
        basis=basis,
        F0=F0,
        neural_vector_field_builder=WaveLikeVectorField,
        model_weight=model_weight,
        space_dimension=2,
        loss_function=loss_function,
        loss_kwargs=loss_kwargs,
        decompose_output=True,
        device=device,
        warm_start=warm_start
    )

    u_pred = trainer.train()

    # crop the predicted solution for the domain of interest
    u_pred = u_pred[:,50:151,50:151]

    # reference solution
    start_time_solver = time.time()
    u_ref = wave_solver(
        c=c,
        u0=u0,
        basis=basis,
        ts=ts,
        device=device
    )
    end_time_solver = time.time()
    solver_time = end_time_solver - start_time_solver

    # crop the reference solution for the domain of interest
    u_ref = u_ref[:,50:151,50:151]

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
            "warm_start": str(warm_start),
            "relative_l2_error": float(relative_l2_error_value),
            "relative_l1_error": float(relative_l1_error_value),
            "training_time": float(trainer.training_time),
            "steps": int(steps),
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
        fig_name="wave2d_three_layers_plot.png",
        extent=[*domain_x,*domain_y],
        cmap="seismic",
    )