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
