import matplotlib.pyplot as plt
import json
import numpy as np

plt.rcParams.update({
    "font.size": 15
})

# --------------------------------------------------------
# Number of temporal subdomains
# --------------------------------------------------------

p = [10, 20, 40, 50, 100]
x_plot = np.arange(len(p))

seeds = [10,11,13,14,15]

# --------------------------------------------------------
# Fill with your values
# --------------------------------------------------------

rrmse_three = []
time_three = []

rrmse_marmousi = []
time_marmousi = []

for n in p:
    rrmse_three_mean = 0.0
    time_three_mean = 0.0
    for seed in seeds:
        with open(
            f"results/wave2d_three_layers/ablation_subdomains/subdomains_{n}/D_NeuSA/seed_{seed}/metrics/metrics.json",
            "r"
        ) as f:
            metrics = json.load(f)

        rrmse_three_mean = rrmse_three_mean + metrics["relative_l2_error"]
        time_three_mean = time_three_mean + metrics["training_time"]

    rrmse_three_mean = rrmse_three_mean / len(seeds)
    time_three_mean = time_three_mean / len(seeds)

    rrmse_three.append(rrmse_three_mean)
    time_three.append(time_three_mean)
    
    rrmse_marmousi_mean = 0.0
    time_marmousi_mean = 0.0
    for seed in seeds:
        with open(
            f"results/wave2d_marmousi/ablation_subdomains/subdomains_{n}/D_NeuSA/seed_{seed}/metrics/metrics.json",
            "r"
        ) as f:
            metrics = json.load(f)
        rrmse_marmousi_mean = rrmse_marmousi_mean + metrics["relative_l2_error"]
        time_marmousi_mean = time_marmousi_mean + metrics["training_time"]
    
    rrmse_marmousi_mean = rrmse_marmousi_mean / len(seeds)
    time_marmousi_mean = time_marmousi_mean / len(seeds)

    rrmse_marmousi.append(rrmse_marmousi_mean)
    time_marmousi.append(time_marmousi_mean)
# --------------------------------------------------------
# Figure
# --------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(10,8),
    sharex=True
)

# ========================================================
# Three Layers - Error
# ========================================================

ax = axes[0,0]

ax.plot(
    x_plot,
    rrmse_three,
    "-o",
    color="tab:blue",
    linewidth=2,
    markersize=7,
)

#ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xticks(x_plot)
ax.set_xticklabels([str(x) for x in p])

ax.set_ylabel(r"rRMSE (log scale)")
ax.set_title("(a) Three layers")

ax.grid(False)

# ========================================================
# Three layers - Time
# ========================================================

ax = axes[0,1]

ax.plot(
    x_plot,
    time_three,
    "-o",
    color="tab:red",
    linewidth=2,
    markersize=7,
)

#ax.set_xscale("log")

ax.set_xticks(x_plot)
ax.set_xticklabels([str(x) for x in p])

ax.set_ylabel("Training time (s)")
ax.set_title("(b) Three layers")

ax.grid(False)

# ========================================================
# Marmousi - Error
# ========================================================

ax = axes[1,0]

ax.plot(
    x_plot,
    rrmse_marmousi,
    "-o",
    color="tab:blue",
    linewidth=2,
    markersize=7,
)

#ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xticks(x_plot)
ax.set_xticklabels([str(x) for x in p])

ax.set_xlabel("Number of temporal subdomains")
ax.set_ylabel(r"rRMSE (log scale)")
ax.set_title("(c) Marmousi")

ax.grid(False)

# ========================================================
# Marmousi - Time
# ========================================================

ax = axes[1,1]

ax.plot(
    x_plot,
    time_marmousi,
    "-o",
    color="tab:red",
    linewidth=2,
    markersize=7,
)

#ax.set_xscale("log")

ax.set_xticks(x_plot)
ax.set_xticklabels([str(x) for x in p])

ax.set_xlabel("Number of temporal subdomains")
ax.set_ylabel("Training time (s)")
ax.set_title("(d) Marmousi")

ax.grid(False)

# --------------------------------------------------------

plt.tight_layout()

plt.savefig(
    "paper_figs/ablation_subdomains_wave2d.png",
    dpi=300,
    bbox_inches="tight"
)

