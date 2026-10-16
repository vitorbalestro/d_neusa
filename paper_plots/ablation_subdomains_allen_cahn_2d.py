
import matplotlib.pyplot as plt
import json
import numpy as np

plt.rcParams.update({
    "font.size": 15
})

# --------------------------------------------------------
# Number of temporal subdomains
# --------------------------------------------------------

p = [100, 200, 500, 1000, 2000, 5000]
x_plot = np.array(p)

seeds = [12, 13, 14, 15, 16]

# --------------------------------------------------------
# Load metrics
# --------------------------------------------------------

rrmse = []
time = []

for n in p:
    rrmse_mean = 0.0
    time_mean = 0.0

    for seed in seeds:
        with open(
            f"results/allen_cahn_2d/ablation_subdomains/subdomains_{n}/D_NeuSA/seed_{seed}/metrics/metrics.json",
            "r"
        ) as f:
            metrics = json.load(f)

        rrmse_mean += metrics["relative_l2_error"]
        time_mean += metrics["training_time"]

    rrmse_mean /= len(seeds)
    time_mean /= len(seeds)

    rrmse.append(rrmse_mean)
    time.append(time_mean)

# --------------------------------------------------------
# Figure
# --------------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(10, 5),
    sharex=True
)

# ========================================================
# Allen-Cahn 2D - Error
# ========================================================

ax = axes[0]

ax.plot(
    x_plot,
    rrmse,
    "-o",
    color="tab:blue",
    linewidth=2,
    markersize=7,
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Number of temporal subdomains")
ax.set_ylabel("rRMSE")
ax.set_title("(a) 2D Allen-Cahn")

ax.grid(False)

# ========================================================
# Allen-Cahn 2D - Training time
# ========================================================

ax = axes[1]

ax.plot(
    x_plot,
    time,
    "-o",
    color="tab:red",
    linewidth=2,
    markersize=7,
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Number of temporal subdomains")
ax.set_ylabel("Training time (s)")
ax.set_title("(b) 2D Allen-Cahn")

ax.grid(False)

# --------------------------------------------------------
# Common axis formatting
# --------------------------------------------------------

for ax in axes:
    ax.set_xticks(p)
    ax.set_xticklabels([str(x) for x in p])
    ax.set_xlim(90, 5500)

plt.tight_layout()

# --------------------------------------------------------
# Save figure
# --------------------------------------------------------

plt.savefig(
    "paper_figs/ablation_subdomains_allen_cahn_2d.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
