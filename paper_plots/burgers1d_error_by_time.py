import numpy as np
import matplotlib.pyplot as plt
from math import pi

diffusions = [0.1,0.01,0.01/pi,0.001]

plt.figure(figsize=(6,4))

for d in diffusions:

    if d == 0.01/pi:
        ref_path = "results/burgers1d/baseline/D_NeuSA/seed_11/solutions/u_ref.npy"
        dneusa_path = "results/burgers1d/baseline/D_NeuSA/seed_11/solutions/u_pred.npy"
        offset = 1
    
    elif d == 0.1 or d == 0.001:
        ref_path = f"results/burgers1d/ablation_diffusion_coefficient/diffusion_{d}/D_NeuSA/seed_11/solutions/u_ref.npy"
        dneusa_path = f"results/burgers1d/ablation_diffusion_coefficient/diffusion_{d}/D_NeuSA/seed_11/solutions/u_pred.npy"
        offset = 30
    else:
        ref_path = f"results/burgers1d/ablation_diffusion_coefficient/diffusion_{d}/D_NeuSA/seed_11/solutions/u_ref.npy"
        dneusa_path = f"results/burgers1d/ablation_diffusion_coefficient/diffusion_{d}/D_NeuSA/seed_11/solutions/u_pred.npy"
        offset = 5

    u_gt = np.load(ref_path)
    u_dneusa = np.load(dneusa_path)

    ts = np.linspace(0,1,1001)
    err = []
    for t in range(1001):
        err_t = np.sqrt(np.sum((u_dneusa[offset*t,:]-u_gt[offset*t,:])**2) / np.sum(u_gt[offset*t,:]**2))
        err.append(err_t) 

    if d == 0.01/pi:
        label = fr"$\nu=0.01/\pi$"
    else:
        label = fr"$\nu={d}$"

    plt.plot(ts,err,lw=1,label=label)

plt.xlabel("Time")
plt.ylabel("Instantaneous rRMSE")
plt.legend()

plt.tight_layout()
plt.savefig("paper_figs/burgers1d_error_by_time.png",dpi=300,bbox_inches="tight")