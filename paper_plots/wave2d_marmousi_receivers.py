import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Load solutions
# ---------------------------------------------------------

u_ref = np.load(
    "results/wave2d_marmousi/baseline/D_NeuSA/"
    "seed_11/solutions/u_ref.npy"
)

u_neusa = np.load(
    "results/wave2d_marmousi/baseline/NeuSA/"
    "seed_11/solutions/u_pred.npy"
)

u_dneusa = np.load(
    "results/wave2d_marmousi/baseline/D_NeuSA/"
    "seed_11/solutions/u_pred.npy"
)

u_cqres = np.load(
    "results/wave2d_marmousi/baseline/Causal_QRes/"
    "seed_11/solutions/u_pred.npy"
)


# ---------------------------------------------------------
# Grid
# ---------------------------------------------------------

xs = np.linspace(-2, 2, u_ref.shape[1])
ys = np.linspace(-2, 2, u_ref.shape[2])
ts = np.linspace(0, 2, u_ref.shape[0])


# ---------------------------------------------------------
# Receivers
# ---------------------------------------------------------

# R1 = (-1, 0.75)
# R2 = ( 0, 0.75)
# R3 = ( 1, 0.75)

receivers = [
    (-1.0, 0.75),
    ( 0.0, 0.75),
    ( 1.0, 0.75),
]

receiver_names = [
    r"$R_1$",
    r"$R_2$",
    r"$R_3$"
]


# ---------------------------------------------------------
# Figure
#
# Row 1: receiver signals
# Row 2: absolute error at each receiver
# ---------------------------------------------------------

fig, axs = plt.subplots(
    2,
    3,
    figsize=(12, 5.2),
    sharex="col"
)


for j, ((xr, yr), receiver_name) in enumerate(
    zip(receivers, receiver_names)
):

    ix = np.argmin(np.abs(xs - xr))
    iy = np.argmin(np.abs(ys - yr))

    # -----------------------------------------------------
    # Extract receiver signals
    # -----------------------------------------------------

    ref = u_ref[:, ix, iy]

    cqres = u_cqres[:, ix, iy]
    neusa = u_neusa[:, ix, iy]
    dneusa = u_dneusa[:, ix, iy]


    # =====================================================
    # TOP ROW: receiver signals
    # =====================================================

    ax = axs[0, j]

    # Reference
    ax.plot(
        ts,
        ref,
        label="Reference",
        color="blue",
        linewidth=1.5
    )

    # C-QRes
    ax.plot(
        ts,
        cqres,
        label="C-QRes",
        color="purple",
        linewidth=1.5
    )

    # NeuSA
    ax.plot(
        ts,
        neusa,
        label="NeuSA",
        color="green",
        linewidth=1.5
    )

    # D-NeuSA
    ax.plot(
        ts,
        dneusa,
        label=r"$\mathbf{D\mathrm{-}NeuSA}$",
        color="red",
        linestyle="--",
        linewidth=2
    )

    ax.set_title(
        receiver_name,
        fontsize=13
    )

    ax.set_xlim(
        ts[0],
        ts[-1]
    )


    # =====================================================
    # BOTTOM ROW: absolute receiver errors
    # =====================================================

    ax_err = axs[1, j]

    error_cqres = np.abs(
        cqres - ref
    )

    error_neusa = np.abs(
        neusa - ref
    )

    error_dneusa = np.abs(
        dneusa - ref
    )


    # C-QRes
    ax_err.plot(
        ts,
        error_cqres,
        label="C-QRes",
        color="purple",
        linewidth=1.5
    )

    # NeuSA
    ax_err.plot(
        ts,
        error_neusa,
        label="NeuSA",
        color="green",
        linewidth=1.5
    )

    # D-NeuSA
    ax_err.plot(
        ts,
        error_dneusa,
        label=r"$\mathbf{D\mathrm{-}NeuSA}$",
        color="red",
        linestyle="--",
        linewidth=2
    )

    ax_err.set_xlabel(
        r"$t$"
    )

    ax_err.set_xlim(
        ts[0],
        ts[-1]
    )

    ax_err.set_ylim(
        bottom=0
    )


# ---------------------------------------------------------
# Y labels
# ---------------------------------------------------------

axs[0, 0].set_ylabel(
    r"$u(R_i,t)$"
)

axs[1, 0].set_ylabel(
    "Absolute error"
)


# ---------------------------------------------------------
# Shared legend
#
# Use handles from top row so Reference is also included.
# ---------------------------------------------------------

handles, labels = axs[0, 0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    frameon=True,
    bbox_to_anchor=(0.5, -0.02),
    fontsize=12
)


# ---------------------------------------------------------
# Spacing
# ---------------------------------------------------------

fig.subplots_adjust(
    bottom=0.16,
    hspace=0.18,
    wspace=0.25
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

plt.savefig(
    "paper_figs/wave2d_marmousi_receivers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)