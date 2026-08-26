import numpy as np
import matplotlib.pyplot as plt
import torch
from basis.fourier1d import FourierBasis1D

plt.rcParams.update({'font.size': 12})


def plot_allen_cahn_snapshots(xs,ts,u_ref,u_dneusa):

    offset = len(ts)//3
    idxs = [0, offset, 2*offset, 3*offset, -1]

    titles = [f"t={ts[idxs[i]]:.2f}" for i in range(4)]

    fig, axs = plt.subplots(1,4,figsize=(16,2.5), sharey=True)


    for i, ax in enumerate(axs):
        ax.plot(xs,u_ref[idxs[i]],label="Reference",color='blue')
        ax.plot(xs,u_dneusa[idxs[i]],label="D-NeuSA",color='red',linestyle='--')
        ax.set_title(titles[i])
        ax.grid(False)
        ax.set_xlabel(r"$x$")

        
    
    axs[0].set_ylabel(r"$u(x,t)$")
    

    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, -0.27),
        fontsize=14,
    )
    
    plt.savefig("paper_figs/allen_cahn_1d_snapshots.png",dpi=300,bbox_inches='tight')
    

# domain, grids and basis
domain_x = [-1.0,1.0]
n_x = 101

basis = FourierBasis1D(n_x,domain_x)
xs = basis.x.numpy()

domain_t = [0.0,1.0]
n_t = 1001
ts = torch.linspace(*domain_t,n_t)

u_ref_path = "results/allen_cahn_1d/baseline/D_NeuSA/seed_11/solutions/u_pred.npy"
u_dneusa_path = "results/allen_cahn_1d/baseline/D_NeuSA/seed_11/solutions/u_ref.npy"

u_ref = np.load(u_ref_path)
u_dneusa = np.load(u_dneusa_path)

plot_allen_cahn_snapshots(xs,ts,u_ref,u_dneusa)