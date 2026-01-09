"""
This code is setted for the second scenario described in the paper,
where the velocity field is normalized by a factor of 2 and,
the source is near to the top of the spatial domain (we assume that x is the vertical axis) 
and the propagation time is 1.2.

In case you want to run the first scenario, just modify the following:

(1) normalize the velocity by a factor of 10 instead of 2
(2) set the center of the Gaussian initial condition at (0,0,0) instead of (-3,0,0)
(3) let domain_t = [0,3] instead of [0,1.2]
"""


import torch
from torchdyn.core import NeuralODE
import time
import json
import argparse
import os
import numpy as np

from basis.cosine3d import CosineBasis3D
from models.hadamard_layer import HadamardLayer
from models.vector_fields import WaveLikeVectorField
from models.mlps import MLPWithTranspose3D
from utils.trainers import AdamTrainer
from utils.metrics import Metrics
from pseudo_spectral_methods.wave import wave_groundtruth_generator
from utils.plots import plot_prediction_vs_groundtruth


def loss_function(NODE,F,F0,c2,huv0,ts,basis):
    huv = NODE(huv0,ts)
    Fhuv = F(huv)
    hu,_ = huv.chunk(2,dim=-1)
    _,Fhv = Fhuv.chunk(2,dim=-1)

    right_side = c2 * basis.idbt(F0(hu))
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
    parser.add_argument('--scenario',type=str,default='first') # first or second scenarios

    args = parser.parse_args()
    dir_name = args.dir
    device = args.device
    seed=args.seed
    steps = args.steps
    lr = args.lr
    scenario = args.scenario

    torch.set_float32_matmul_precision("high")
    torch.manual_seed(seed)

    dir_subtitle = "wave3d_utah_forge_"+scenario+"_scenario"
    save_dir = os.path.join("results", dir_subtitle, dir_name, "D_NeuSA", f"seed_{seed}")
    os.makedirs(save_dir,exist_ok=True)

    if scenario == 'first':
        domain_t = [0,3]
        center = (0,0,0)
        normalizing_factor = 10
    elif scenario == 'second':
        domain_t = [0,1.2]
        center = (-3,0,0)
        normalizing_factor = 2

    # domain
    domain_x = [-4,4]
    domain_y = [-2.5,2.5]
    domain_z = [-2.5,2.5]

    # resolution
    n_x = 65
    n_y = 41
    n_z = 41
    n_t = 301

    # basis and spatial grid
    basis = CosineBasis3D(n_x,n_y,n_z,domain_x,domain_y,domain_z,device=device)
    X,Y,Z = torch.meshgrid(basis.x,basis.y,basis.z, indexing='ij')
    grid = torch.stack((X,Y,Z),dim=-1)
    xs = grid[:,:,:,0:1]
    ys = grid[:,:,:,1:2]
    zs = grid[:,:,:,2:3]

    # time discretization
    ts = torch.linspace(*domain_t,n_t)

    # initial condition
    sigma = 0.15
    u0 = 10*torch.exp((-(xs-center[0])**2-(ys-center[1])**2-(zs-center[2])**2)/(2*sigma**2)).squeeze(-1)
    hu0 = basis.dbt(u0)
    hv0 = torch.zeros_like(hu0)
    huv0 = torch.cat([hu0,hv0],dim=-1)

    # load velocity field
    data_path = "data/utah_forge/Seismic_Velocity_Model_FORGE_Site"

    c_as_array = []

    with open(data_path,'r') as f:
        next(f)
        for line in f:
            cols = line.split()
            c_as_array.append(float(cols[5])/normalizing_factor)     


    c_numpy = np.array(c_as_array).reshape(65,41,41)
    c = torch.tensor(c_numpy,dtype=torch.float32)
    c2 = c*c


    # initial vector field (Laplacian in the frequency domain as Hadamard product)
    omegas = basis.D2
    F0 = HadamardLayer(omegas)

    # time domain decomposition
    n_slices = 60
    assert (n_t - 1) % n_slices == 0
    offset = (n_t-1) // n_slices
    time_slices = [offset * i for i in range(1,n_slices+1)]

    # neural networks and NODEs
    Fs = []
    NODEs = []
    num_hidden_layers = 0

    for _ in range(n_slices):
        F_neural = MLPWithTranspose3D(M=n_x,N=n_y,P=n_z)
        F = WaveLikeVectorField(F_neural,F0,model_weight=10.0)
        NODE = NeuralODE(F,solver='rk4',sensitivity='autograd',return_t_eval=False)

        F = F.to(device)
        NODE = NODE.to(device)

        Fs.append(F)
        NODEs.append(NODE)
    
    # training
    huvs = []

    huv0 = huv0.to(device)
    c2 = c2.to(device)
    ts = ts.to(device)
    basis = basis.to(device)
    F0 = F0.to(device)

    start_time = time.time()

    trainer = AdamTrainer(NODEs[0].parameters(),
                            adam_lr = lr,
                            adam_steps = steps,
                            learning_rate_scheduler=True,
                            loss_function= lambda: loss_function(NODEs[0],Fs[0],F0,c2,huv0,ts[0:time_slices[0]+1],basis)
                        )
    
    
    trainer.train_neusa()
    huvs.append(NODEs[0](huv0,ts[0:time_slices[0]+1]).detach())

    for i in range(1,n_slices):
        huv_init = huvs[i-1][-1].detach()
        huvs[i-1] = huvs[i-1][:-1,:,:]

        trainer = AdamTrainer(NODEs[i].parameters(),
                            adam_lr = lr,
                            adam_steps = steps,
                            learning_rate_scheduler=True,
                            loss_function= lambda: loss_function(NODEs[i],Fs[i],F0,c2,huv_init,ts[time_slices[i-1]:time_slices[i]+1],basis)
                        )
    
        trainer.train_neusa()

        huvs.append(NODEs[i](huv_init,ts[time_slices[i-1]:time_slices[i]+1]).detach())

    end_time = time.time()
    training_time = end_time-start_time

    # retrieve model solution
    huv = torch.cat(huvs,dim=0)
    hu,_ = huv.chunk(2,dim=-1)
    u_pred = basis.idbt(hu).detach().cpu().numpy()

    # load ground-truth solution (via pseudo-spectral method)
    u_gt = wave_groundtruth_generator(c,u0,basis,ts,device)

    # compute and save metrics
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
        "training_time": float(training_time),
    }

    save_dir_metrics = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir_metrics,exist_ok=True)

    with open(os.path.join(save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

     #### PLOTS ####
    
    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    # slice at z = 0.5
    u_pred_slice = u_pred[:,:,:,24]
    u_gt_slice = u_gt[:,:,:,24]

    save_path = os.path.join(figs_dir,"plots.png")
    plot_prediction_vs_groundtruth(ts=ts,u_pred=u_pred_slice,u_gt=u_gt_slice,extent=[*domain_x,*domain_y],save_path=save_path,cmap="seismic")
    