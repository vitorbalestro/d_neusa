import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update({'font.size': 12})

def create_graphs_grid(xs, u_neusa, u_dneusa, u_pinn, u_fls, u_qres,
                       titles=["t=0.0", "t=0.33", "t=0.66", "t=0.99"]):

    fig, axs = plt.subplots(1, 4, figsize=(15, 4), sharey=False)

    axs[0].set_ylabel("Error")
    for i, ax in enumerate(axs):
        ax.plot(xs, u_neusa[i], label="NeuSA", color='green')
        ax.plot(xs, u_dneusa[i], label="D-NeuSA", color='red',linewidth=2.5,linestyle='--')
        #ax.plot(xs, u_pinn[i], label="PINN", color="#E69F00")
        #ax.plot(xs, u_fls[i], label="FLS", color="#7B3294")
        ax.plot(xs, u_qres[i], label="QRes", color="#8C564B")

        ax.set_title(titles[i])
        ax.grid(False)

    # legenda única
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=6,
        frameon=True,
        fontsize=12
    )

    plt.savefig("paper_figs/burgers1d_error_by_model.png", dpi=300, bbox_inches="tight")
    plt.close(fig)



domain_x = [-1.0,1.0]
n_x = 101

xs = torch.linspace(*domain_x,n_x).numpy()

domain_t = [0.0,1.0]
n_t = 201
ts = torch.linspace(*domain_t,n_t).numpy()

u_ref = np.load("results/burgers1d/baseline/D_NeuSA/seed_11/solutions/u_ref.npy")[::5,::5]

u_neusa = np.load("results/burgers1d/baseline/NeuSA/seed_11/solutions/u_pred.npy")[::5,::5]
u_dneusa = np.load("results/burgers1d/baseline/D_NeuSA/seed_11/solutions/u_pred.npy")[::5,::5]
u_pinn = np.load("results/burgers1d/baseline/PINN/seed_11/solutions/u_pred.npy")[:,::2]
u_fls = np.load("results/burgers1d/baseline/FLS/seed_11/solutions/u_pred.npy")[:,::2]
u_qres = np.load("results/burgers1d/baseline/QRes/seed_11/solutions/u_pred.npy")[:,::2]

offset = n_t // 4
u_ref_plot = [u_ref[offset,:]-u_ref[offset,:], u_ref[2*offset,:]-u_ref[2*offset,:], u_ref[3 * offset,:]-u_ref[3*offset,:], u_ref[-1,:]-u_ref[-1,:]]
u_neusa_plot = [u_neusa[offset,:]-u_ref[offset,:], u_neusa[2*offset,:]-u_ref[2*offset,:], u_neusa[3 * offset,:]-u_ref[3*offset,:], u_neusa[-1,:]-u_ref[-1,:]]
u_dneusa_plot = [u_dneusa[offset,:]-u_ref[offset,:], u_dneusa[2*offset,:]-u_ref[2*offset,:], u_dneusa[3 * offset,:]-u_ref[3*offset,:], u_dneusa[-1,:]-u_ref[-1,:]]
u_pinn_plot = [u_pinn[offset,:]-u_ref[offset,:], u_pinn[2*offset,:]-u_ref[2*offset,:], u_pinn[3 * offset,:]-u_ref[3*offset,:], u_pinn[-1,:]-u_ref[-1,:]]
u_fls_plot = [u_fls[offset,:]-u_ref[offset,:], u_fls[2*offset,:]-u_ref[2*offset,:], u_fls[3 * offset,:]-u_ref[3*offset,:], u_fls[-1,:]-u_ref[-1,:]]
u_qres_plot = [u_qres[offset,:]-u_ref[offset,:], u_qres[2*offset,:]-u_ref[2*offset,:], u_qres[3 * offset,:]-u_ref[3*offset,:], u_qres[-1,:]-u_ref[-1,:]]

x_plot = xs
titles = [f"t={ts[offset].item()}", f"t={ts[2*offset].item()}", f"t={ts[3 * offset].item()}", f"t={ts[-1].item()}"]

create_graphs_grid(x_plot,u_neusa_plot,u_dneusa_plot,u_pinn_plot,u_fls_plot,u_qres_plot,titles=titles)