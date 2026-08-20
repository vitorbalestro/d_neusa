import matplotlib.pyplot as plt
import numpy as np
import torch
import json
from math import pi

plt.rcParams.update({
    "font.size": 13
})

diffusions = [0.1,0.01,0.01/pi,0.001]
seeds_dneusa = [11,12,13,14,15]
rrmse_list_dneusa = []

for d in diffusions:
    rrmse_mean_dneusa = 0.0
    if d == 0.01/pi:
        seeds_dneusa = [11,12,14,15,17]
    else:
        seeds_dneusa = [11,12,13,14,15]

    for s in seeds_dneusa:
        
        if d != 0.01/pi:
            path = f"results/burgers1d/ablation_diffusion_coefficient/diffusion_{d}/D_NeuSA/seed_{s}/metrics/metrics.json"
        else:
            path = f"results/burgers1d/baseline/D_NeuSA/seed_{s}/metrics/metrics.json"

        with open(path,"r") as f:
            metrics = json.load(f)
        rrmse = metrics["relative_l2_error"]
        rrmse_mean_dneusa = rrmse_mean_dneusa + rrmse


    rrmse_mean_dneusa = rrmse_mean_dneusa / len(seeds_dneusa)

    rrmse_list_dneusa.append(rrmse_mean_dneusa)


plt.figure(figsize=(6,4))


plt.plot(
    diffusions,
    rrmse_list_dneusa,
    marker="s",
    linewidth=2,
    label="D-NeuSA"
)

plt.xscale("log")

plt.gca().invert_xaxis()

plt.xlabel("Diffusion coefficient (log scale)")
plt.ylabel("rRMSE (log scale)")
plt.yscale("log")
plt.xticks([0.1,0.01,0.01/pi,0.001],[r"$0.1$",r"$0.01$",r"$0.01/\pi$",r"$0.001$"])
plt.legend()

plt.tight_layout()

plt.savefig(
    "paper_figs/burgers1d_diffusion_coefficient_ablation.png",
    dpi=300,
    bbox_inches="tight"
)