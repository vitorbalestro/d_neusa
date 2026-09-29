import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pathlib import Path

plt.rcParams.update({
    "font.size": 13
})

diffusions = [0.1, 0.01, 0.01/pi, 0.001]

labels = [
    r"$\nu = 0.1$",
    r"$\nu = 0.01$",
    r"$\nu = 0.01/\pi$",
    r"$\nu = 0.001$",
]

seed = 11


# ============================================================
# Load solutions
# ============================================================

def load_solution(d):

    if d != 0.01/pi:
        path = Path(
            f"results/burgers1d/"
            f"ablation_diffusion_coefficient/"
            f"diffusion_{d}/D_NeuSA/seed_{seed}/solutions"
        )
    else:
        path = Path(
            f"results/burgers1d/"
            f"baseline/D_NeuSA/seed_{seed}/solutions"
        )

    # Arrays have shape (Nt, Nx).
    # The last temporal index corresponds to t = 1.

    u_pred = np.load(path / "u_pred.npy")[-1, :]
    u_ref = np.load(path / "u_ref.npy")[-1, :]

    assert u_pred.shape == u_ref.shape

    x = np.linspace(-1, 1, len(u_ref))

    return x, u_ref, u_pred


results = [load_solution(d) for d in diffusions]


# ============================================================
# Plot
# ============================================================

fig, axs = plt.subplots(
    1, 4,
    figsize=(16, 3.8),
    sharex=True,
    sharey=True,
)

for i, (label, data) in enumerate(zip(labels, results)):

    x, u_ref, u_pred = data

    axs[i].plot(
        x,
        u_ref,
        label="Reference",
        color="blue",
        linewidth=1.8,
    )

    axs[i].plot(
        x,
        u_pred,
        label="D-NeuSA",
        color="red",
        linestyle="--",
        linewidth=2.0,
    )

    axs[i].set_title(label, pad=10)

    axs[i].set_xlim(-1, 1)
    axs[i].set_xticks([-1, -0.5, 0, 0.5, 1])
    axs[i].set_xlabel(r"$x$")
    axs[i].grid(False)

axs[0].set_ylabel(r"$u(x,1)$")


# ============================================================
# Shared legend below the plots
# ============================================================

handles, legend_labels = axs[0].get_legend_handles_labels()

fig.legend(
    handles,
    legend_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=2,
    frameon=False,
    fontsize=13,
)

fig.subplots_adjust(
    left=0.06,
    right=0.99,
    top=0.88,
    bottom=0.28,
    wspace=0.12,
)


# ============================================================
# Save
# ============================================================

Path("paper_figs").mkdir(parents=True, exist_ok=True)

fig.savefig(
    "paper_figs/burgers1d_shock.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()
plt.close(fig)