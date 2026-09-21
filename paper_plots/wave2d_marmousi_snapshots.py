import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 12})

# ---------------------------------------------------------
# Load solutions
# ---------------------------------------------------------

u_ref = np.load(
    "results/wave2d_marmousi/baseline/D_NeuSA/"
    "seed_11/solutions/u_ref.npy",
    mmap_mode="r"
)

u_neusa = np.load(
    "results/wave2d_marmousi/baseline/NeuSA/"
    "seed_11/solutions/u_pred.npy",
    mmap_mode="r"
)

u_dneusa = np.load(
    "results/wave2d_marmousi/baseline/D_NeuSA/"
    "seed_11/solutions/u_pred.npy",
    mmap_mode="r"
)

u_cqres = np.load(
    "results/wave2d_marmousi/baseline/Causal_QRes/"
    "seed_11/solutions/u_pred.npy",
    mmap_mode="r"
)

# ---------------------------------------------------------
# Domain
# ---------------------------------------------------------

domain_x = [-2, 2]
domain_y = [-2, 2]
domain_t = [0, 2]

n_t, n_x, n_y = u_ref.shape

xs = np.linspace(*domain_x, n_x)
ys = np.linspace(*domain_y, n_y)
ts = np.linspace(*domain_t, n_t)

extent = [*domain_x, *domain_y]

# ---------------------------------------------------------
# Times
# ---------------------------------------------------------

times_to_plot = [0.4, 0.8, 1.2, 1.6]

idxs = [
    np.argmin(np.abs(ts - t))
    for t in times_to_plot
]

# ---------------------------------------------------------
# Interfaces and receivers
# ---------------------------------------------------------

interfaces = [0.5, 1.0]

receivers = [
    (-1.0, 0.75),
    (0.0, 0.75),
    (1.0, 0.75),
]

# ---------------------------------------------------------
# Common color scale
# ---------------------------------------------------------

vmax = max(
    np.abs(u_ref[idx]).max()
    for idx in idxs
)
vmin = -vmax

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

# Visual order from TOP to BOTTOM:
#
# C-QRes
# NeuSA
# D-NeuSA
# Reference

solutions = [
    ("C-QRes", u_cqres),
    ("NeuSA", u_neusa),
    ("D-NeuSA", u_dneusa),
    ("Reference", u_ref),
]

n_rows = len(solutions)
n_cols = len(idxs)

fig, axs = plt.subplots(
    n_rows,
    n_cols,
    figsize=(12, 9.5),
    sharex=True,
    sharey=True
)

for row, (model_name, u) in enumerate(solutions):

    for col, idx in enumerate(idxs):

        ax = axs[row, col]

        im = ax.imshow(
            u[idx].T,
            cmap="seismic",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic"
        )


        # -------------------------------------------------
        # Receivers -- reference only
        # -------------------------------------------------

        if model_name == "Reference":
            for xr, yr in receivers:
                ax.scatter(
                    xr,
                    yr,
                    s=14,
                    facecolor="white",
                    edgecolor="black",
                    linewidth=0.8,
                    zorder=5
                )

        # -------------------------------------------------
        # Axes
        # -------------------------------------------------

        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])

        # Invert y axis
        ax.invert_yaxis()

        # Titles only on top row
        if row == 0:
            ax.set_title(f"t={ts[idx]:.1f}")

        # x labels only on bottom row
        if row == n_rows - 1:
            ax.set_xlabel(r"$x$", labelpad=3)

    # -----------------------------------------------------
    # y label -- close to axis
    # -----------------------------------------------------

    axs[row, 0].set_ylabel(
        r"$y$",
        rotation=90,
        labelpad=-1
    )

    # -----------------------------------------------------
    # Model name -- substantially farther from y
    # -----------------------------------------------------

    axs[row, 0].text(
        -0.43,
        0.5,
        model_name,
        transform=axs[row, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold" if model_name == "D-NeuSA" else "normal"
    )

# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------

# More room on left for model names
fig.subplots_adjust(
    left=0.15,
    right=0.88,
    bottom=0.07,
    top=0.95,
    wspace=0.08,
    hspace=0.12
)

# ---------------------------------------------------------
# Colorbar
# ---------------------------------------------------------

cbar_ax = fig.add_axes([
    0.90,
    0.14,
    0.015,
    0.72
])

cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Amplitude")

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

plt.savefig(
    "paper_figs/wave2d_marmousi_snapshots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)