import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# =========================================================
# Load solutions
# =========================================================

u_ref = np.load(
    "results/allen_cahn_2d/baseline/D_NeuSA/"
    "seed_11/solutions/u_ref.npy"
)[::100,:,:]

u_dneusa = np.load(
    "results/allen_cahn_2d/baseline/D_NeuSA/"
    "seed_11/solutions/u_pred.npy"
)[::100,:,:]

u_cqres = np.load(
    "results/allen_cahn_2d/baseline/Causal_QRes/"
    "seed_11/solutions/u_pred.npy"
)


# =========================================================
# Grid
# =========================================================

nt, nx, ny = u_ref.shape

xs = np.linspace(0, 1, nx)
ys = np.linspace(0, 1, ny)
ts = np.linspace(0, 10, nt)


# =========================================================
# Output
# =========================================================

output_dir = "supplementary_material"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(
    output_dir,
    "allen_cahn_2d_animation.gif"
)


# =========================================================
# Common color scale
# =========================================================
#
# Allen--Cahn is naturally centered around zero.
# Using a symmetric scale makes comparison easier.
# =========================================================

vmax = max(
    np.max(np.abs(u_ref)),
    np.max(np.abs(u_cqres)),
    np.max(np.abs(u_dneusa))
)

vmin = -vmax


# =========================================================
# Figure
# =========================================================

fig, axs = plt.subplots(
    1,
    3,
    figsize=(10.5, 3.4),
    constrained_layout=True
)

titles = [
    "C-QRes",
    "D-NeuSA",
    "Reference"
]

solutions = [
    u_cqres,
    u_dneusa,
    u_ref
]


# ---------------------------------------------------------
# Initial images
# ---------------------------------------------------------

images = []

for ax, title, u in zip(
    axs,
    titles,
    solutions
):

    im = ax.imshow(
        u[0].T,
        extent=[
            xs[0],
            xs[-1],
            ys[0],
            ys[-1]
        ],
        origin="lower",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        aspect="equal"
    )

    images.append(im)

    ax.set_title(
        title,
        fontsize=13
    )

    ax.set_xlabel(
        r"$x$"
    )

    ax.set_ylabel(
        r"$y$"
    )


# =========================================================
# Shared colorbar
# =========================================================

cbar = fig.colorbar(
    images[0],
    ax=axs,
    shrink=0.85,
    pad=0.02
)

cbar.set_label(
    r"$u(x,y,t)$"
)


# =========================================================
# Time label
# =========================================================

time_text = fig.suptitle(
    rf"$t={ts[0]:.2f}$",
    fontsize=15
)


# =========================================================
# Animation update
# =========================================================

def update(frame):

    for im, u in zip(
        images,
        solutions
    ):
        im.set_data(
            u[frame].T
        )

    time_text.set_text(
        rf"$t={ts[frame]:.2f}$"
    )

    return images + [time_text]


# =========================================================
# Create animation
# =========================================================

animation = FuncAnimation(
    fig,
    update,
    frames=nt,
    interval=80,
    blit=False,
    repeat=True
)


# =========================================================
# Save GIF
# =========================================================

writer = PillowWriter(
    fps=12
)

animation.save(
    output_path,
    writer=writer,
    dpi=120
)

plt.close(fig)

print(
    f"GIF saved to: {output_path}"
)