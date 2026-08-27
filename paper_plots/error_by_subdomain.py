import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

equations = [
    "allen_cahn_1d",
    "burgers1d",
    "wave2d_three_layers",
    "wave2d_marmousi",
    "wave3d_utah_forge",
]

titles = [
    "1D Allen-Cahn",
    "Burgers",
    "Three layers",
    "Marmousi",
    "Utah FORGE",
]

ns_t = [1001, 1001, 201, 201, 301]
ns_slices = [100, 100,40, 50, 60]
seeds = [11,12,14,15,17]

fig = plt.figure(figsize=(15, 8))

gs = GridSpec(2, 6, figure=fig)

axes = [
    fig.add_subplot(gs[0, 0:2]),
    fig.add_subplot(gs[0, 2:4]),
    fig.add_subplot(gs[0, 4:6]),
    fig.add_subplot(gs[1, 1:3]),
    fig.add_subplot(gs[1, 3:5]),
]

for ax, equation, title, n_t, n_slices in zip(
    axes, equations, titles, ns_t, ns_slices
):
    rrmse_dneusa = []
    rrmse_neusa = []

    rrmse_dneusa_dict = {}
    rrmse_neusa_dict = {}

    for seed in seeds:

        rrmse_dneusa_dict[seed] = [0.0 for _ in range(n_slices)]
        rrmse_neusa_dict[seed] = [0.0 for _ in range(n_slices)]

        u_ref = np.load(
            f"results/{equation}/baseline/D_NeuSA/seed_{seed}/solutions/u_ref.npy"
        )

        u_dneusa = np.load(
            f"results/{equation}/baseline/D_NeuSA/seed_{seed}/solutions/u_pred.npy"
        )

        u_neusa = np.load(
            f"results/{equation}/baseline/NeuSA/seed_{seed}/solutions/u_pred.npy"
        )

        offset = (n_t - 1) // n_slices

        

        for k in range(n_slices):

            sl = slice(offset * k, offset * (k + 1))

            ref = u_ref[sl]
            dneusa = u_dneusa[sl]
            neusa = u_neusa[sl]

            rrmse_dneusa_dict[seed][k] += np.sqrt(np.sum((ref - dneusa) ** 2) / np.sum(ref ** 2))
            rrmse_neusa_dict[seed][k] += np.sqrt(np.sum((ref - neusa) ** 2) / np.sum(ref ** 2))

    for k in range(n_slices):
        kth_rrmse_dneusa_acc = 0.0
        kth_rrmse_neusa_acc = 0.0
        for seed in seeds:
            kth_rrmse_dneusa_acc += rrmse_dneusa_dict[seed][k] 
            kth_rrmse_neusa_acc += rrmse_neusa_dict[seed][k]

        rrmse_dneusa.append(kth_rrmse_dneusa_acc / len(seeds))
        rrmse_neusa.append(kth_rrmse_neusa_acc / len(seeds)
                           )
    xs = np.arange(n_slices)

    ax.plot(
        xs,
        rrmse_dneusa,
        marker="o",
        markersize=4,
        linewidth=2.5,
        label="D-NeuSA",
    )

    ax.plot(
        xs,
        rrmse_neusa,
        marker="o",
        markersize=4,
        linewidth=2.5,
        label="NeuSA",
    )

    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Temporal subdomain index", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)

    if ax in [axes[0], axes[3]]:
        ax.set_ylabel("Relative RMSE", fontsize=13)

axes[0].legend(fontsize=12)

plt.tight_layout()

plt.savefig(
    "paper_figs/error_by_subdomain.png",
    dpi=300,
    bbox_inches="tight",
)