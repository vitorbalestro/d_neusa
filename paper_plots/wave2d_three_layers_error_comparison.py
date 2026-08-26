import matplotlib.pyplot as plt 
import numpy as np 
import torch

u_dneusa = np.load("results/wave2d_three_layers/baseline/D_NeuSA/seed_11/solutions/u_pred.npy")
u_ref = np.load("results/wave2d_three_layers/baseline/D_NeuSA/seed_11/solutions/u_ref.npy")
u_neusa = np.load("results/wave2d_three_layers/baseline/NeuSA/seed_11/solutions/u_pred.npy")

# domain
domain_x = [-2,2]
domain_y = [-2,2]
domain_z = [-2,2]
domain_t = [0,2]

n_t = 201
ts = torch.linspace(*domain_t,n_t)

extent=[*domain_x,*domain_y]
cmap='seismic'
interpolation='bicubic'

num_frames = 5
idxs = np.linspace(0, len(ts)-1, num_frames, dtype=int)
selected_ts = [ts[i] for i in idxs]

fig, axs = plt.subplots(2, num_frames, figsize=(12,4))

u_ref_slice = u_ref[:,:,:]
u_dneusa_slice = u_dneusa[:,:,:]     
u_neusa_slice = u_neusa[:,:,:]

vmax = 0.2*np.abs(u_ref_slice[idxs[2],:,:]).max()
vmin = -vmax
for j, i in enumerate(idxs):
    
    u_dneusa_plot = u_dneusa_slice[i].T - u_ref_slice[i].T
    u_neusa_plot = u_neusa_slice[i].T - u_ref_slice[i].T
    
    im0 = axs[0, j].imshow(u_dneusa_plot, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax,interpolation=interpolation)
    axs[0, j].set_title(f"t = {selected_ts[j]:.2f}")
    axs[0, j].set_xticks([])
    axs[0, j].set_yticks([])
    axs[0, j].tick_params(labelbottom=False, labelleft=False)

    im1 = axs[1, j].imshow(u_neusa_plot, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax,interpolation=interpolation)
    axs[1, j].set_xticks([])
    axs[1, j].set_yticks([])
    axs[1, j].tick_params(labelbottom=False, labelleft=False)

    
axs[0, 0].set_ylabel("D-NeuSA", fontsize=12, fontweight="bold")
axs[1, 0].set_ylabel("NeuSA", fontsize=12)

cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # left, bottom, width, height
fig.colorbar(im1, cax=cbar_ax).set_label("Error")

plt.subplots_adjust(right=0.85, wspace=0.1, hspace=0.2)

plt.savefig("paper_figs/wave2d_three_layers_error_comparison.png", dpi=300, bbox_inches='tight')

plt.close()