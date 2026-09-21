import os
import numpy as np
import torch
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_dir = (
    "results/transport1d/transport_experiment/PINN/"
    "seed_42/checkpoints"
)

neusa_path = (
    "results/transport1d/transport_experiment/NeuSA/"
    "seed_42/solutions/u_pred.npy"
)

dneusa_path = (
    "results/transport1d/transport_experiment/D_NeuSA/"
    "seed_42/solutions/u_pred.npy"
)

snapshot_steps = [
    1_000,
    10_000,
    100_000,
    5_000_000,
]

all_steps = [
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
    100_000,
    200_000,
    500_000,
    1_000_000,
    2_000_000,
    3_000_000,
    4_000_000,
    5_000_000,
]

t_plot = 0.5

nx = 201
nt = 201

xs = np.linspace(0.0, 1.0, nx)
ts = np.linspace(0.0, 1.0, nt)

it = np.argmin(np.abs(ts - t_plot))
t_actual = ts[it]


# ============================================================
# Exact solution
# ============================================================

def exact_solution(x, t):
    return (
        np.sin(2 * np.pi * (x - t))
        + 0.3 * np.sin(20 * np.pi * (x - t))
    )


def low_frequency_solution(x, t):
    return np.sin(2 * np.pi * (x - t))


# ============================================================
# PINN checkpoint prediction
# ============================================================

def load_prediction(step):

    from models.PINN import PINN

    model = PINN(
        in_dim=2,
        out_dim=1,
        hidden_dim=128,
        num_layer=4,
    ).to(device)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"model_step_{step}.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    x_tensor = torch.tensor(
        xs,
        dtype=torch.float32,
        device=device
    ).reshape(-1, 1)

    t_tensor = torch.full_like(
        x_tensor,
        t_actual
    )

    with torch.no_grad():
        pred = model(
            t_tensor,
            x_tensor
        )

    return pred.detach().cpu().numpy().squeeze()


# ============================================================
# Fourier amplitude
# ============================================================

def fourier_amplitude(u, k):

    # x=0 and x=1 are the same periodic point.
    u_periodic = u[:-1]

    coeff = (
        np.fft.fft(u_periodic)
        / len(u_periodic)
    )

    return 2.0 * np.abs(coeff[k])


# ============================================================
# Load PINN checkpoints
# ============================================================

predictions = {}

for step in all_steps:
    predictions[step] = load_prediction(step)


# ============================================================
# Fourier amplitudes
# ============================================================

amp_k1 = []
amp_k10 = []

for step in all_steps:

    u = predictions[step]

    amp_k1.append(
        fourier_amplitude(u, 1)
    )

    amp_k10.append(
        fourier_amplitude(u, 10) / 0.3
    )

amp_k1 = np.asarray(amp_k1)
amp_k10 = np.asarray(amp_k10)


# ============================================================
# Load NeuSA and D-NeuSA
# ============================================================

u_neusa = np.load(neusa_path)
u_dneusa = np.load(dneusa_path)


assert u_neusa.shape == (nt, nx)
assert u_dneusa.shape == (nt, nx)


# ============================================================
# Curves at t = t_plot
# ============================================================

u_ref = exact_solution(
    xs,
    t_actual
)

u_low = low_frequency_solution(
    xs,
    t_actual
)

u_pinn_final = predictions[
    all_steps[-1]
]

u_neusa_plot = u_neusa[it]
u_dneusa_plot = u_dneusa[it]


# ============================================================
# Figure
#
# Explicit positioning is intentional here.  The figure has
# two geometrically different rows:
#
#   (a) four relatively narrow panels;
#   (b,c) two larger panels.
#
# Using add_axes gives independent control over both rows and
# avoids GridSpec compressing panel (c).
# ============================================================

fig = plt.figure(
    figsize=(15.5, 7.4)
)


# ============================================================
# Geometry
# ============================================================

# ---------- top row ----------
top_left = 0.070
top_right = 0.985

top_bottom = 0.565
top_height = 0.350

top_gap = 0.060

top_width = (
    (top_right - top_left)
    - 3 * top_gap
) / 4


# ---------- bottom row ----------
bottom_left = 0.070
bottom_right = 0.985

bottom_bottom = 0.085
bottom_height = 0.300

bottom_gap = 0.105

bottom_width = (
    (bottom_right - bottom_left)
    - bottom_gap
) / 2


# ============================================================
# (a) PINN training evolution
# ============================================================

axs_top = []

for i, step in enumerate(snapshot_steps):

    left = (
        top_left
        + i * (top_width + top_gap)
    )

    ax = fig.add_axes([
        left,
        top_bottom,
        top_width,
        top_height
    ])

    axs_top.append(ax)

    ax.plot(
        xs,
        u_ref,
        linewidth=2.0,
        label="Reference"
    )

    ax.plot(
        xs,
        predictions[step],
        "--",
        linewidth=2.0,
        label="PINN"
    )

    ax.plot(
        xs,
        u_low,
        ":",
        linewidth=2.0,
        label="Low-frequency component"
    )

    ax.set_title(
        f"{step:,} steps",
        fontsize=12,
        pad=7
    )

    ax.set_xlim(
        0.0,
        1.0
    )

    ax.set_ylim(
        -1.45,
        1.45
    )

    ax.set_xticks(
        np.linspace(0, 1, 6)
    )

    ax.set_xlabel(
        r"$x$",
        fontsize=11
    )

    ax.tick_params(
        axis="both",
        labelsize=10
    )

    if i == 0:
        ax.set_ylabel(
            rf"$u(x,{t_actual:.1f})$",
            fontsize=11
        )


# ============================================================
# Label (a)
# ============================================================

fig.text(
    0.025,
    0.955,
    "(a)",
    fontsize=14,
    fontweight="bold",
    ha="left",
    va="top"
)


# ============================================================
# Legend for (a)
# ============================================================

handles, labels = (
    axs_top[0].get_legend_handles_labels()
)

fig.legend(
    handles,
    labels,
    loc="center",
    bbox_to_anchor=(0.5, 0.475),
    ncol=3,
    frameon=True,
    fontsize=10,
    handlelength=2.8,
    columnspacing=2.0
)


# ============================================================
# (b) Fourier amplitudes
# ============================================================

ax_fourier = fig.add_axes([
    bottom_left,
    bottom_bottom,
    bottom_width,
    bottom_height
])

ax_fourier.plot(
    all_steps,
    amp_k1,
    "-o",
    markersize=4,
    linewidth=2,
    label=r"$k=1$"
)

ax_fourier.plot(
    all_steps,
    amp_k10,
    "-s",
    markersize=4,
    linewidth=2,
    label=r"$k=10$"
)

ax_fourier.axhline(
    1.0,
    linestyle="--",
    linewidth=1.2,
    label="Reference amplitude"
)

ax_fourier.set_xscale(
    "log"
)

ax_fourier.set_xlabel(
    "Training steps",
    fontsize=11
)

ax_fourier.set_ylabel(
    "Normalized Fourier amplitude",
    fontsize=11
)

ax_fourier.set_ylim(
    0.0,
    1.1
)

ax_fourier.tick_params(
    axis="both",
    labelsize=10
)

ax_fourier.legend(
    frameon=True,
    fontsize=9,
    loc="center"
)

ax_fourier.grid(
    alpha=0.25
)


# ============================================================
# Label (b)
# ============================================================

fig.text(
    bottom_left - 0.050,
    bottom_bottom + bottom_height + 0.040,
    "(b)",
    fontsize=14,
    fontweight="bold",
    ha="left",
    va="center"
)


# ============================================================
# (c) Final comparison
# ============================================================

# Make panel (c) narrower while keeping the same height.
# This gives it a more natural aspect ratio for a solution snapshot.
compare_width = 0.29

# Center panel (c) inside the horizontal region previously
# allocated to it.
compare_region_left = (
    bottom_left
    + bottom_width
    + bottom_gap
)

compare_left = (
    compare_region_left
    + (bottom_width - compare_width) / 2
)

ax_compare = fig.add_axes([
    compare_left,
    bottom_bottom,
    compare_width,
    bottom_height
])

ax_compare.plot(
    xs,
    u_ref,
    linewidth=2.2,
    label="Reference",
    zorder=2
)

ax_compare.plot(
    xs,
    u_pinn_final,
    "-.",
    linewidth=1.8,
    label="PINN",
    zorder=3
)

ax_compare.plot(
    xs,
    u_neusa_plot,
    color="tab:purple",
    linewidth=1.8,
    label="NeuSA",
    zorder=1
)

ax_compare.plot(
    xs,
    u_dneusa_plot,
    "--",
    color="tab:red",
    linewidth=2.5,
    label="D-NeuSA",
    zorder=4
)

ax_compare.set_xlim(
    0.0,
    1.0
)

ax_compare.set_ylim(
    -1.45,
    1.45
)

ax_compare.set_xticks(
    np.linspace(0, 1, 6)
)

ax_compare.set_xlabel(
    r"$x$",
    fontsize=11
)

ax_compare.set_ylabel(
    rf"$u(x,{t_actual:.1f})$",
    fontsize=11
)

ax_compare.tick_params(
    axis="both",
    labelsize=10
)

ax_compare.legend(
    frameon=True,
    fontsize=9,
    ncol=2,
    loc="upper left"
)

ax_compare.grid(
    alpha=0.15
)


# ============================================================
# Label (c)
# ============================================================

fig.text(
    compare_left - 0.050,
    bottom_bottom + bottom_height + 0.040,
    "(c)",
    fontsize=14,
    fontweight="bold",
    ha="left",
    va="center"
)

# ============================================================
# Save
# ============================================================

os.makedirs(
    "paper_figs",
    exist_ok=True
)

plt.savefig(
    "paper_figs/transport1d_analysis.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.08
)

plt.show()