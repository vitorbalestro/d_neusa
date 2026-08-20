import matplotlib.pyplot as plt
import numpy as np
import torch
import json

plt.rcParams.update({
    "font.size": 13
})

diffusions = [0.1,0.01,0.001,0.0001,0.00001,0.000001]
seeds_dneusa = [11, 12, 13, 14, 15]
seeds_neusa = [11, 12, 13, 14, 15]
rrmse_list_dneusa = []
rrmse_list_neusa = []

for d in diffusions:
    rrmse_mean_dneusa = 0.0

    for s in seeds_dneusa:
        path = f"results/allen_cahn_1d/ablation_diffusion_coefficient/diffusion_{d}/D_NeuSA/seed_{s}/metrics/metrics.json"
        with open(path,"r") as f:
            metrics = json.load(f)
        rrmse = metrics["relative_l2_error"]
        rrmse_mean_dneusa = rrmse_mean_dneusa + rrmse


    rrmse_mean_dneusa = rrmse_mean_dneusa / len(seeds_dneusa)

    rrmse_list_dneusa.append(rrmse_mean_dneusa)

    rrmse_mean_neusa = 0.0

    for s in seeds_neusa:
        path = f"results/allen_cahn_1d/ablation_diffusion_coefficient/diffusion_{d}/NeuSA/seed_{s}/metrics/metrics.json"
        with open(path,"r") as f:
            metrics = json.load(f)
        rrmse = metrics["relative_l2_error"]
        rrmse_mean_neusa = rrmse_mean_neusa + rrmse


    rrmse_mean_neusa = rrmse_mean_neusa / len(seeds_neusa)
    rrmse_list_neusa.append(rrmse_mean_neusa)


plt.figure(figsize=(6,4))


plt.plot(
    diffusions,
    rrmse_list_dneusa,
    marker="s",
    linewidth=2,
    label="D-NeuSA"
)

plt.plot(
    diffusions,
    rrmse_list_neusa,
    marker="o",
    linewidth=2,
    label="NeuSA"
)

plt.xscale("log")

plt.gca().invert_xaxis()

plt.xlabel("Diffusion coefficient (log scale)")
plt.ylabel("rRMSE (log scale)")
plt.yscale("log")

plt.legend()

plt.tight_layout()

plt.savefig(
    "paper_figs/allen_cahn_1d_diffusion_coefficient_ablation.png",
    dpi=300,
    bbox_inches="tight"
)