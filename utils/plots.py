import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

def create_graphs_grid(x_gt, u_gt, u_pred, titles, figs_dir):
    plt.figure()
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    for i, ax in enumerate(axs):
        ax.plot(x_gt, u_gt[i], label="GT", color='blue')
        ax.plot(x_gt, u_pred[i], label="Predicted", color='red')
        ax.set_title(titles[i])
        ax.legend()
        ax.grid(False)

    plt.tight_layout()
    fig.savefig(os.path.join(figs_dir, "t_row.png"))
    plt.close(fig)

def create_time_evolution_gif(x_gt,t_gt,u_gt,u_pred,figs_dir, gif_name="evolution.gif", duration=50):
    
    ymin, ymax = np.min(u_gt), np.max(u_gt)
    
    frames = []
    for i in range(len(t_gt)):
        fig, ax = plt.subplots()
        ax.plot(x_gt,u_gt[i,:],color='blue', label="GT")
        ax.plot(x_gt,u_pred[i,:],color='red', label="Predicted")
        ax.set_title(f"t={t_gt[i].item()}")
        ax.set_ylim(ymin,ymax+0.05)
        ax.grid(False)
    
        fname = f"_frame_{i:03d}.png"
        fig.savefig(fname)
        plt.close(fig)
        frames.append(Image.open(fname))
    
    frames[0].save(os.path.join(figs_dir,gif_name),save_all=True,append_images=frames[1:],duration=duration,loop=0)

    for frame in frames:
        frame.close()
    for i in range(len(t_gt)):
        os.remove(f"_frame_{i:03d}.png")

def plot_prediction_vs_groundtruth(ts,u_pred,u_gt,extent=[0, 1, 0, 1],save_path=None,cmap="seismic",transpose=True):
    
    num_frames = 5
    idxs = np.linspace(0, len(ts)-1, num_frames, dtype=int)
    selected_ts = [ts[i] for i in idxs]

    fig, axs = plt.subplots(2, num_frames, figsize=(num_frames * 3, 6))
    

    vmax_gt = np.abs(u_gt[idxs[1],:,:]).max()
    vmin_gt = -vmax_gt
    for j, i in enumerate(idxs):
        if transpose == True:
            u_pred_plot = u_pred[i].T 
            u_gt_plot = u_gt[i].T
        else:
            u_pred_plot = u_pred[i]
            u_gt_plot = u_gt[i]
        im1 = axs[0, j].imshow(u_pred_plot, cmap=cmap, extent=extent, vmin=vmin_gt, vmax=vmax_gt)
        axs[0, j].set_title(f"t = {selected_ts[j]:.2f}")
        axs[0, j].set_xticks([])
        axs[0, j].set_yticks([])
        axs[0, j].tick_params(labelbottom=False, labelleft=False)

        im2 = axs[1, j].imshow(u_gt_plot, cmap=cmap, extent=extent, vmin=vmin_gt, vmax=vmax_gt)
        axs[1, j].set_xticks([])
        axs[1, j].set_yticks([])
        axs[1, j].tick_params(labelbottom=False, labelleft=False)

    axs[0, 0].set_ylabel("Prediction", fontsize=12)
    axs[1, 0].set_ylabel("Ground-Truth", fontsize=12)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # left, bottom, width, height
    fig.colorbar(im1, cax=cbar_ax).set_label("Amplitude")

    plt.subplots_adjust(right=0.9, wspace=0.1, hspace=0.2)


    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_error(ts, error, extent=[0, 1, 0, 1], save_path=None, cmap="seismic"):
    num_frames = 5
    idxs = np.linspace(0, len(ts)-1, num_frames, dtype=int)
    selected_ts = [ts[i] for i in idxs]

    fig, axs = plt.subplots(1, num_frames, figsize=(num_frames * 3, 6))

    if num_frames == 1:
        axs = [axs]

    vmin = error.min()
    vmax = error.max()

    for j, i in enumerate(idxs):
        im = axs[j].imshow(error[i].T, cmap=cmap,
                           extent=extent, vmin=vmin, vmax=vmax)
        axs[j].set_title(f"t = {selected_ts[j]:.2f}")
        axs[j].set_xticks([])
        axs[j].set_yticks([])
        axs[j].tick_params(labelbottom=False, labelleft=False)

    axs[0].set_ylabel("Error", fontsize=12)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  
    fig.colorbar(im, cax=cbar_ax).set_label("Amplitude")

    plt.subplots_adjust(right=0.9, wspace=0.1)

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

