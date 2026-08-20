import numpy as np
import matplotlib.pyplot as plt

domain_t = [0.0,10.0]
n_t = 20001
ts = np.linspace(*domain_t,n_t)
n_slices = 1000

u_ref_path = "results/allen_cahn_2d/baseline/D_NeuSA/seed_11/solutions/u_ref.npy"
u_dneusa_path = "results/allen_cahn_2d/baseline/D_NeuSA/seed_11/solutions/u_pred.npy"

u_ref = np.load(u_ref_path)
u_dneusa = np.load(u_dneusa_path)

offset = (n_t - 1) // n_slices
time_slices = [offset*i for i in range(1,n_slices+1)]

rrmses_per_subdomain_dneusa = []

for i in range(n_slices):
    u_gt_subdomain = u_ref[offset*i:offset*(i+1),:]
    u_dneusa_subdomain = u_dneusa[offset*i:offset*(i+1),:]


    rrmse_dneusa = np.sqrt(np.sum((u_gt_subdomain-u_dneusa_subdomain)**2) /
        np.sum((u_gt_subdomain)**2))
    
    rrmses_per_subdomain_dneusa.append(rrmse_dneusa)


xs_plot = [i for i in range(n_slices)]

plt.plot(
    xs_plot,
    rrmses_per_subdomain_dneusa,
    marker="o",
    markersize=3,
    linewidth=2,
    label="D-NeuSA"
)


plt.title("2D Allen-Cahn", fontsize=15)
plt.xlabel("Temporal subdomain index",fontsize=13)
plt.ylabel("Relative RMSE",fontsize=13)
plt.legend(fontsize=12)

plt.tight_layout()

plt.savefig(
    "paper_figs/allen_cahn_2d_error_by_subdomain.png",
    dpi=300,
    bbox_inches="tight"
)


