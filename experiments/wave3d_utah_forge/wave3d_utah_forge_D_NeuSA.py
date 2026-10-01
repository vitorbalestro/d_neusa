import torch
import numpy as np
import time
import json
import argparse
import os

from basis.cosine3d import CosineBasis3D
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
    hu,_ = huv.chunk(2,dim=-1)
    _,Fhv = Fhuv.chunk(2,dim=-1)

    right_side = c**2 * basis.idbt(F0(hu))
    v_t = basis.idbt(Fhv)

    loss = torch.mean((v_t-right_side)**2)

    return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Wave 3D Utah FORGE D-NeuSA')
    parser.add_argument('--dir', type=str,default='free_experiment')    # name of the directory where the results folder will be saved
    parser.add_argument('--seed', type=int,default= 42)    # random seed
    parser.add_argument('--device', type=str, default='cuda:0')   
    parser.add_argument('--steps', type=int, default=3000)  
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--ablation_subdomains', action='store_true')
    parser.add_argument('--n_slices', type=int, default=60)
    parser.add_argument('--enhanced_neusa', action="store_true")
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
        equation="wave3d_utah_forge",
        model=model,
        dir_name=dir_name,
        ablation=ablation,
        specifier=specifier,
        seed=seed
    )

    directory_manager.config()

    # domain, spectral basis and discretizations
    domain_x = [-2.0,2.0]
    n_x = 41
    domain_y = [-2.0,2.0]
    n_y = 41
    domain_z = [-2.0,2.0]
    n_z = 41
    domain_t = [0.0,3.0]
    n_t = 301

    basis = CosineBasis3D(
        Mx=n_x,
        My=n_y,
        Mz=n_z,
        domain_x=domain_x,
        domain_y=domain_y,
        domain_z=domain_z,
        device=device
    )

    X,Y,Z = torch.meshgrid(basis.x,basis.y,basis.z,indexing='ij')
    grid = torch.stack((X,Y,Z),dim=-1)
    xs = grid[:,:,:,0]
    ys = grid[:,:,:,1]
    zs = grid[:,:,:,2]

    ts = torch.linspace(*domain_t,n_t)

    # initial condition
    sigma = 0.15
    center = (-1,0,0)
    u0 = 10*torch.exp((-(xs-center[0])**2-(ys-center[1])**2-(zs-center[2])**2)/(2*sigma**2))
    hu0 = basis.dbt(u0)
    hv0 = torch.zeros_like(hu0)
    huv0 = torch.cat([hu0,hv0],dim=-1)

    # Utah FORGE velocity field
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
    c_numpy = np.transpose(c_numpy, (2, 1, 0)).copy()

    initial_velocity = c_numpy[20,20,20]
    
    c = torch.tensor(c_numpy,dtype=torch.float32,device=device)

    # spectral laplacian operator
    D2 = basis.D2
    F0 = HadamardLayer(D2)

    # model architecture and parameters
    M = n_x
    N = n_y
    P = n_z
    model_weight = 10.0
    if not enhanced_neusa:
        num_hidden_layers = 0
        widen_factor = 1
    else:
        num_hidden_layers = 2
        widen_factor = 4

    neural_network_dict={
        "M": M,
        "N": N,
        "P": P,
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

    # vector field kwargs dict
    vector_field_kwargs = {
        "initial_velocity": initial_velocity
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
        space_dimension=3,
        loss_function=loss_function,
        loss_kwargs=loss_kwargs,
        vector_field_kwargs=vector_field_kwargs,
        decompose_output=True,
        device=device,
        warm_start=warm_start
    )

    u_pred = trainer.train()

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
            "steps": int(steps),
            "training_time": float(trainer.training_time),
            "num_params": int(trainer.num_params),
            "model_inference_time": float(trainer.inference_time),
            "solver_inference_time": float(solver_time)
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