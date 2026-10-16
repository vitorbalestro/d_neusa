import matplotlib.pyplot as plt
import json
import numpy as np

plt.rcParams.update({
    "font.size": 15
})

# --------------------------------------------------------
# Number of temporal subdomains
# --------------------------------------------------------

p = [10, 20, 50, 100, 200, 500]
x_plot = np.arange(len(p))

seeds = [11,12,13,14,15]

# --------------------------------------------------------
# Fill with your values
# --------------------------------------------------------

rrmse_allen = []
time_allen = []

rrmse_burgers = []
time_burgers = []

for n in p:
    rrmse_allen_mean = 0.0
    time_allen_mean = 0.0
    for seed in seeds:
        with open(
            f"results/allen_cahn_1d/ablation_subdomains/subdomains_{n}/D_NeuSA/seed_{seed}/metrics/metrics.json",
            "r"
        ) as f:
            metrics = json.load(f)

        rrmse_allen_mean = rrmse_allen_mean + metrics["relative_l2_error"]
        time_allen_mean = time_allen_mean + metrics["training_time"]

    rrmse_allen_mean = rrmse_allen_mean / len(seeds)
    time_allen_mean = time_allen_mean / len(seeds)
    rrmse_allen.append(rrmse_allen_mean)
    time_allen.append(time_allen_mean)

    rrmse_burgers_mean = 0.0
    time_burgers_mean = 0.0
    for seed in seeds:
        with open(
            f"results/burgers1d/ablation_subdomains/subdomains_{n}/D_NeuSA/seed_{seed}/metrics/metrics.json",
            "r"
        ) as f:
            metrics = json.load(f)
        rrmse_burgers_mean = rrmse_burgers_mean + metrics["relative_l2_error"]
        time_burgers_mean = time_burgers_mean + metrics["training_time"]
    
    rrmse_burgers_mean = rrmse_burgers_mean / len(seeds)
    time_burgers_mean = time_burgers_mean / len(seeds)
    rrmse_burgers.append(rrmse_burgers_mean)
    time_burgers.append(time_burgers_mean)

# --------------------------------------------------------
# Figure
# --------------------------------------------------------

fig, axes = plt.subplots(
    2, 2,
    figsize=(10, 8),
    sharex=True
)

# Use the actual number of subdomains
x_plot = np.array(p)

# ========================================================
# Allen-Cahn - Error
# ========================================================

ax = axes[0, 0]

ax.plot(
    x_plot,
    rrmse_allen,
    "-o",
    color="tab:blue",
    linewidth=2,
    markersize=7,
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_ylabel("rRMSE")
ax.set_title("(a) 1D Allen-Cahn")
ax.grid(False)

# ========================================================
# Allen-Cahn - Time
# ========================================================

ax = axes[0, 1]

ax.plot(
    x_plot,
    time_allen,
    "-o",
    color="tab:red",
    linewidth=2,
    markersize=7,
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_ylabel("Training time (s)")
ax.set_title("(b) 1D Allen-Cahn")
ax.grid(False)

# ========================================================
# Burgers - Error
# ========================================================

ax = axes[1, 0]

ax.plot(
    x_plot,
    rrmse_burgers,
    "-o",
    color="tab:blue",
    linewidth=2,
    markersize=7,
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Number of temporal subdomains")
ax.set_ylabel("rRMSE")
ax.set_title("(c) Burgers")
ax.grid(False)

# ========================================================
# Burgers - Time
# ========================================================

ax = axes[1, 1]

ax.plot(
    x_plot,
    time_burgers,
    "-o",
    color="tab:red",
    linewidth=2,
    markersize=7,
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Number of temporal subdomains")
ax.set_ylabel("Training time (s)")
ax.set_title("(d) Burgers")
ax.grid(False)

# --------------------------------------------------------
# Common axis formatting
# --------------------------------------------------------

for ax in axes.flat:
    ax.set_xticks(p)
    ax.set_xticklabels([str(x) for x in p])
    ax.set_xlim(9, 550)

plt.tight_layout()

plt.savefig(
    "paper_figs/ablation_subdomains_allen_cahn_burgers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
