import argparse
import os
import time
import json
import numpy as np
import torch
from torchdyn.core import NeuralODE

from basis.cosine2d import CosineBasis2D
from models.hadamard_layer import HadamardLayer
from models.vector_fields import WaveLikeVectorField
from models.mlps import MLPWithTranspose2D
from utils.trainers import AdamTrainer
from utils.metrics import Metrics
from utils.plots import plot_prediction_vs_groundtruth, plot_error
from pseudo_spectral_methods.wave import wave_groundtruth_generator

def loss_function(NODE,basis,huv0,ts,F,F0,c2):
    huv = NODE(huv0,ts)
    Fhuv = F(huv)
    hu,_ = huv.chunk(2,dim=-1)
    _,Fhv = Fhuv.chunk(2,dim=-1)

    # u_tt = v_t = c^2D^2u
    right_side = c2 * basis.idbt(F0(hu))
    v_t = basis.idbt(Fhv)

    # we train solely on a neighborhood of the internal domain
    dd = 30
    v_t = v_t[:,100-dd:301+dd,100-dd:301+dd]
    right_side = right_side[:,100-dd:301+dd,100-dd:301+dd]

    loss = torch.mean((v_t-right_side)**2)
    
    return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Wave 2D Marmousi D-NeuSA')
    parser.add_argument('--dir', type=str, default='free_experiment')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    dir_name = args.dir
    seed = args.seed
    device = args.device
    steps = args.steps
    lr = args.lr

    torch.set_float32_matmul_precision('high')
    torch.manual_seed(seed)

    save_dir = os.path.join('results','wave2d_marmousi',dir_name,'D_NeuSA',f"seed_{seed}")
    os.makedirs(save_dir, exist_ok=True)

    # domain and resolution
    domain_x = [-2.0,2.0]
    domain_y = [-2.0,2.0]
    domain_t = [0.0,2.0]

    n_x = 201
    n_y = 201
    time_steps = 201
    
    # time discretization
    ts = torch.linspace(*domain_t,time_steps)

    # extended domain and grid for the training
    domain_x_ext = [-4.0,4.0]
    domain_y_ext = [-4.0,4.0]
    M = 401
    N = 401

    # basis and 2 dimensional spatial grid
    basis = CosineBasis2D(M,N,domain_x_ext,domain_y_ext,device=device)
    X,Y = torch.meshgrid(basis.x,basis.y,indexing='ij')
    
    grid = torch.stack((X,Y),dim=-1)    # shape = (M,N,2)
    xs = grid[:,:,0:1]
    ys = grid[:,:,1:2]
    
    # initial condition
    sigma = 0.1
    u0 = torch.exp(-((xs**2)+(ys+0.5)**2)/(2*sigma**2)).squeeze(-1)
    v0 = torch.zeros_like(u0)
    hu0 = basis.dbt(u0)     # shape = (M,N)
    hv0 = torch.zeros_like(hu0)
    huv0 = torch.cat([hu0,hv0],dim=-1) # shape = (M,2N)
    
    # load Marmousi velocity field
    velocity_profile = np.loadtxt("data/wave2d_marmousi/velocity_profile.txt",dtype=np.float32)
    c = torch.tensor(velocity_profile)
    c2 = c * c

    # initial vector field (Laplacian in the frequency domain as Hadamard product)
    omegas = basis.D2
    F0 = HadamardLayer(omegas)

    # time domain decomposition
    n_slices = 40
    assert (time_steps - 1) % n_slices == 0
    offset = (time_steps-1) // n_slices
    time_slices = [offset * i for i in range(1,n_slices+1)]

    # neural networks and NODEs
    Fs = []
    NODEs = []
    num_hidden_layers=0

    for _ in range(n_slices):
        F_neural = MLPWithTranspose2D(M=M,N=N,num_hidden_layers=num_hidden_layers)
        F = WaveLikeVectorField(F_neural,F0,model_weight=2.0)
        NODE = NeuralODE(F,solver='rk4',sensitivity='autograd',return_t_eval=False)

        F = F.to(device)
        NODE = NODE.to(device)

        Fs.append(F)
        NODEs.append(NODE)
    


    # training
    huvs = []

    basis = basis.to(device)
    huv0 = huv0.to(device)
    ts = ts.to(device)
    F0 = F0.to(device)
    c2 = c2.to(device)

    start_time = time.time()

    trainer = AdamTrainer(NODEs[0].parameters(),
                            adam_lr = lr,
                            adam_steps = 5*steps,
                            learning_rate_scheduler=True,
                            loss_function=lambda: loss_function(NODEs[0],basis,huv0,ts[0:time_slices[0]+1],Fs[0],F0,c2)
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
                                loss_function=lambda: loss_function(NODEs[i],basis,huv_init,ts[time_slices[i-1]:time_slices[i]+1],Fs[i],F0,c2)
                            )

        
        trainer.train_neusa()
            
        huvs.append(NODEs[i](huv_init,ts[time_slices[i-1]:time_slices[i]+1]).detach())

    end_time = time.time()
    training_time = end_time-start_time

    # retrieve model solution
    huv = torch.cat(huvs,dim=0)
    hu,_ = huv.chunk(2,dim=-1)
    u = basis.idbt(hu).detach().cpu().numpy()
    
    u_pred = u[:,100:301,100:301]
    

    # ground-truth solution with pseudo-spectral method
    # notice that we need more time steps in order to have convergence of the method
    gt_time_steps = 401
    ts = torch.linspace(*domain_t,gt_time_steps)
    u_gt_ps = wave_groundtruth_generator(c,u0,basis,ts,device)
    
    u_gt = u_gt_ps[:,100:301,100:301]
    time_idx = range(0,gt_time_steps,2)
    u_gt = u_gt[time_idx,:,:]
    ts = ts[time_idx]
    
    # compute and save metrics
    metrics = Metrics(u_pred,u_gt)
    relative_l2_error_value = metrics.relative_l2_error()
    relative_l1_error_value = metrics.relative_l1_error()

    metrics_dict = {
            "relative_l2_error": float(relative_l2_error_value),
            "relative_l1_error": float(relative_l1_error_value),
            "training_time": float(training_time)
        }

    save_dir_solutions = os.path.join(save_dir, "solutions")
    os.makedirs(save_dir_solutions,exist_ok=True)

    np.save(os.path.join(save_dir_solutions, "u_pred.npy"),u_pred)
    np.save(os.path.join(save_dir_solutions, "u_gt.npy"),u_gt)


    save_dir_metrics = os.path.join(save_dir, "metrics")
    os.makedirs(save_dir_metrics,exist_ok=True)

    with open(os.path.join(save_dir_metrics, "metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

    #### PLOTS ####

    figs_dir = os.path.join(save_dir,"figures")
    os.makedirs(figs_dir, exist_ok=True)

    save_path = os.path.join(figs_dir,"plots.png")
    plot_prediction_vs_groundtruth(ts=ts,u_pred=u_pred,u_gt=u_gt,extent=[*domain_x,*domain_y],save_path=save_path,cmap="seismic")
    
    save_path = os.path.join(figs_dir,"error.png")
    plot_error(ts=ts,error=u_pred-u_gt,extent=[*domain_x,*domain_y],save_path=save_path,cmap="seismic")
