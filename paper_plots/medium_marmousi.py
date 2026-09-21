import numpy as np
import matplotlib.pyplot as plt
import torch

plt.rcParams.update({'font.size': 15})

velocity_profile = np.loadtxt("data/wave2d_marmousi/velocity_profile.txt",dtype=np.float32)
c = torch.tensor(velocity_profile[:, ::-1].copy())
c = c[100:301,100:301]

fig, axes = plt.subplots()
im = axes.imshow(c.T,extent=[-2,2,-2,2])
axes.yaxis.set_inverted(True)
axes.yaxis.set_ticks([-2, -1, 0, 1, 2])
axes.xaxis.set_ticks([-2, -1, 0, 1, 2])
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(im, label="c(x,y)")
plt.savefig("paper_figs/medium_marmousi.png",dpi=300, bbox_inches="tight")