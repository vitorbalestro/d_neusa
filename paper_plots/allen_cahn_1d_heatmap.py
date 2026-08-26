import numpy as np
import matplotlib.pyplot as plt

u_ref_path = "results/allen_cahn_1d/baseline/D_NeuSA/seed_11/solutions/u_ref.npy"
u_dneusa_path = "results/allen_cahn_1d/baseline/D_NeuSA/seed_11/solutions/u_pred.npy"
u_neusa_path = "results/allen_cahn_1d/baseline/NeuSA/seed_11/solutions/u_pred.npy"

u_ref = np.load(u_ref_path)
u_dneusa = np.load(u_dneusa_path)
u_neusa = np.load(u_neusa_path)

def plot_allen_cahn_heatmap(u_dneusa,u_neusa,u_ref,extent=[0,1,-1,1],save_path=None,cmap="seismic"):
    
    fig,axes = plt.subplots(1,3,figsize=(15,4),sharey=True)

    vmin = u_ref.min()
    vmax = u_ref.max()

    im0 = axes[0].imshow(u_neusa.T,aspect='auto',origin='lower',vmin=vmin,vmax=vmax,extent=extent,cmap=cmap)
    axes[0].set_title('NeuSA')
    axes[0].set_ylabel(r"$x$")
    axes[0].set_xlabel(r"$t$")
    im1 = axes[1].imshow(u_dneusa.T,aspect='auto',origin='lower',vmin=vmin,vmax=vmax,extent=extent,cmap=cmap)
    axes[1].set_title('D-NeuSA',fontweight='bold')
    axes[1].set_xlabel(r"$t$")
    im2 = axes[2].imshow(u_ref.T,aspect='auto',origin='lower',vmin=vmin,vmax=vmax,extent=extent,cmap=cmap)
    axes[2].set_title('Reference')
    axes[2].set_xlabel(r"$t$")

    cbar = fig.colorbar(im2,ax=axes,shrink=0.85)

    plt.savefig('paper_figs/allen_cahn_1d_heatmap.png',dpi=300,bbox_inches='tight')

plot_allen_cahn_heatmap(u_dneusa,u_neusa,u_ref)