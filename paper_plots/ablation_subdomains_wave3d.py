import matplotlib.pyplot as plt
import json
import numpy as np

plt.rcParams.update({
    "font.size": 15
})


p = [5, 10, 20, 50, 60, 150]
x_plot = np.arange(len(p))

seeds = [11,12,13,14,15]

rrmse_three = []
time_three = []

for n in p:
    rrmse_mean = 0.0
    time_mean = 0.0
    for seed in seeds:
        with open(
            f"results/wave3d_utah_forge/ablation_subdomains/subdomains_{n}/D_NeuSA/seed_{seed}/metrics/metrics.json",
            "r"
        ) as f:
            metrics = json.load(f)

        rrmse_mean = rrmse_mean + metrics["relative_l2_error"]
        time_mean = time_mean + metrics["training_time"]

    rrmse_mean = rrmse_mean / len(seeds)
    time_mean = time_mean / len(seeds)

    rrmse_three.append(rrmse_mean)
    time_three.append(time_mean)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(10,5),
    sharex=True
)

ax = axes[0]

ax.plot(
    x_plot,
    rrmse_three,
    "-o",
    color="tab:blue",
    linewidth=2,
    markersize=7,
)

ax.set_yscale("log")

ax.set_xticks(x_plot)
ax.set_xticklabels([str(x) for x in p])

ax.set_xlabel("Number of temporal subdomains")

ax.set_ylabel(r"rRMSE (log scale)")
ax.set_title("(a) Utah FORGE")

ax.grid(False)


ax = axes[1]

ax.plot(
    x_plot,
    time_three,
    "-o",
    color="tab:red",
    linewidth=2,
    markersize=7,
)

ax.set_xticks(x_plot)
ax.set_xticklabels([str(x) for x in p])

ax.set_xlabel("Number of temporal subdomains")

ax.set_ylabel("Training time (s)")
ax.set_title("(b) Utah FORGE")

ax.grid(False)


plt.tight_layout()

plt.savefig(
    "paper_figs/ablation_subdomains_wave3d.png",
    dpi=300,
    bbox_inches="tight"
)

