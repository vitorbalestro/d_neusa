import matplotlib.pyplot as plt
import numpy as np
import os

def experiment_plot_1d(xs, ts, u_ref, u_pred, figs_dir, fig_name):
    
    n_t = len(ts)
    offset = n_t // 3
    u_ref_plot = [u_ref[0,:], u_ref[offset,:], u_ref[2 * offset,:], u_ref[n_t-1,:]]
    u_pred_plot = [u_pred[0,:], u_pred[offset,:], u_pred[2 * offset,:], u_pred[n_t-1,:]]

    titles = [f"t={ts[0].item()}", f"t={ts[offset].item()}", f"t={ts[2 * offset].item()}", f"t={ts[n_t-1].item()}"]

    
    plt.figure()
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    for i, ax in enumerate(axs):
        ax.plot(xs, u_ref_plot[i], label="Reference", color='blue')
        ax.plot(xs, u_pred_plot[i], label="Predicted", color='red')
        ax.set_title(titles[i])
        ax.legend()
        ax.grid(False)

    plt.tight_layout()
    fig.savefig(os.path.join(figs_dir, fig_name))
    plt.close(fig)


def experiment_plot_2d(
        ts,
        u_pred,
        u_ref,
        figs_dir,
        fig_name,
        extent=[0, 1, 0, 1],
        cmap="seismic",
        transpose=True,
        interpolation='none'
    ):
    
    num_frames = 5
    idxs = np.linspace(0, len(ts)-1, num_frames, dtype=int)
    selected_ts = [ts[i] for i in idxs]

    fig, axs = plt.subplots(2, num_frames, figsize=(num_frames * 3, 6))
    

    vmax_gt = np.abs(u_ref[idxs[1],:,:]).max()
    vmin_gt = -vmax_gt
    for j, i in enumerate(idxs):
        if transpose == True:
            u_pred_plot = u_pred[i].T 
            u_gt_plot = u_ref[i].T
        else:
            u_pred_plot = u_pred[i]
            u_gt_plot = u_ref[i]
        im1 = axs[0, j].imshow(u_pred_plot, cmap=cmap, extent=extent, vmin=vmin_gt, vmax=vmax_gt,interpolation=interpolation)
        axs[0, j].set_title(f"t = {selected_ts[j]:.2f}")
        axs[0, j].set_xticks([])
        axs[0, j].set_yticks([])
        axs[0, j].tick_params(labelbottom=False, labelleft=False)

        im2 = axs[1, j].imshow(u_gt_plot, cmap=cmap, extent=extent, vmin=vmin_gt, vmax=vmax_gt,interpolation=interpolation)
        axs[1, j].set_xticks([])
        axs[1, j].set_yticks([])
        axs[1, j].tick_params(labelbottom=False, labelleft=False)

    axs[0, 0].set_ylabel("Prediction", fontsize=12)
    axs[1, 0].set_ylabel("Reference", fontsize=12)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # left, bottom, width, height
    fig.colorbar(im1, cax=cbar_ax).set_label("Amplitude")

    plt.subplots_adjust(right=0.9, wspace=0.1, hspace=0.2)


    
    fig.savefig(os.path.join(figs_dir, fig_name))
    plt.close(fig)