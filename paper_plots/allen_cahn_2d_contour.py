import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})


def equivalent_radius(u, xs, ys):
    """
    Equivalent radius of the region u > 0:
        R = sqrt(A / pi)
    """
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]

    # area for every time instant
    area = np.sum(u > 0, axis=(1, 2)) * dx * dy

    return np.sqrt(area / np.pi)


def plot_interface_snapshots(
    xs, ys, ts, u_ref, u_dneusa, u_cqres,
    times=(0.0, 3.0, 6.0, 9.0),
    save_path="paper_figs/allen_cahn_2d_contour.png"
):
    # Find only the indices corresponding to the requested times
    time_indices = [
        np.argmin(np.abs(ts - t))
        for t in times
    ]

    X, Y = np.meshgrid(xs, ys, indexing="ij")

    fig, axs = plt.subplots(
        1, len(time_indices),
        figsize=(3.2 * len(time_indices), 3.5),
        sharex=True,
        sharey=True
    )

    for i, (ax, idx) in enumerate(zip(axs, time_indices)):

        # Read only the required snapshots
        u_ref_t = u_ref[idx]
        u_dneusa_t = u_dneusa[idx]
        u_cqres_t = u_cqres[idx]

        # Reference
        ax.contour(
            X, Y, u_ref_t,
            levels=[0],
            colors="blue",
            linewidths=2
        )

        # D-NeuSA
        ax.contour(
            X, Y, u_dneusa_t,
            levels=[0],
            colors="red",
            linewidths=2.5,
            linestyles="--"
        )

        # C-QRes
        ax.contour(
            X, Y, u_cqres_t,
            levels=[0],
            colors="green",
            linewidths=2.5,
            linestyles="-."
        )

        ax.set_title(f"t={ts[idx]:.1f}")
        ax.set_xlabel(r"$x$")

        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])

        ax.set_aspect("equal")

    axs[0].set_ylabel(r"$y$")

    # Legend handles
    line_ref, = axs[0].plot(
        [], [],
        color="blue",
        linewidth=2,
        label="Reference"
    )

    line_dneusa, = axs[0].plot(
        [], [],
        color="red",
        linewidth=2.5,
        linestyle="--",
        label=r"$\mathbf{D\mathrm{-}NeuSA}$"
    )

    line_cqres, = axs[0].plot(
        [], [],
        color="green",
        linewidth=2.5,
        linestyle="-.",
        label=r"$\mathbf{C\mathrm{-}QRes}$"
    )

    fig.legend(
        handles=[line_ref, line_dneusa, line_cqres],
        loc="lower center",
        ncol=3,
        frameon=True,
        bbox_to_anchor=(0.5, -0.10),
        fontsize=13
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plot_radius(
    xs, ys, ts, u_ref, u_dneusa, u_cqres,
    save_path="paper_figs/allen_cahn_2d_radius.png"
):
    r_ref = equivalent_radius(u_ref, xs, ys)
    r_dneusa = equivalent_radius(u_dneusa, xs, ys)
    r_cqres = equivalent_radius(u_cqres, xs, ys)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        ts, r_ref,
        color="blue",
        linewidth=2,
        label="Reference"
    )

    ax.plot(
        ts, r_dneusa,
        color="red",
        linestyle="--",
        linewidth=2.5,
        label=r"$\mathbf{D\mathrm{-}NeuSA}$"
    )

    ax.plot(
        ts, r_cqres,
        color="green",
        linestyle="-.",
        linewidth=2.5,
        label=r"$\mathbf{C\mathrm{-}QRes}$"
    )

    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"Equivalent radius $R(t)$")

    ax.grid(False)
    ax.legend(frameon=True)

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return r_ref, r_dneusa, r_cqres


# ============================================================
# Load solutions
# ============================================================

u_ref = np.load(
    "results/allen_cahn_2d/baseline/D_NeuSA/seed_11/solutions/u_ref.npy"
)[::100,:,:]

u_dneusa = np.load(
    "results/allen_cahn_2d/baseline/D_NeuSA/seed_11/solutions/u_pred.npy"
)[::100,:,:]

u_cqres = np.load(
    "results/allen_cahn_2d/baseline/Causal_QRes/seed_11/solutions/u_pred.npy"
)


# ============================================================
# Grid
# ============================================================

xs = np.linspace(0, 1, u_ref.shape[1])
ys = np.linspace(0, 1, u_ref.shape[2])
ts = np.linspace(0, 10, u_ref.shape[0])


# ============================================================
# Plots
# ============================================================

plot_interface_snapshots(
    xs, ys, ts,
    u_ref, u_dneusa, u_cqres
)

r_ref, r_dneusa, r_cqres = plot_radius(
    xs, ys, ts,
    u_ref, u_dneusa, u_cqres
)