import numpy as np
import matplotlib.pyplot as plt
import torch

plt.rcParams.update({'font.size': 15})

domain_x = [-2.0,2.0]
domain_y = [-2.0,2.0]
n_x = 201
n_y = 201

x = torch.linspace(*domain_x,n_x)
y = torch.linspace(*domain_y,n_y)

X,Y = torch.meshgrid(x,y,indexing='ij')

grid = torch.stack((X,Y),dim=-1)    # shape = (M,N,2)
xs = grid[:,:,0:1]
ys = grid[:,:,1:2]  
ys = ys.flip(1)  # we reverse the direction of y as usually done in geophysics

c = (1.0 + 0.25 * (torch.sigmoid(1000 * (ys - 0.5)) + torch.sigmoid(1000 * (ys - 1.0)))).squeeze(-1)

fig, axes = plt.subplots()
im = axes.imshow(c.T,extent=[*domain_x,*domain_y])
axes.yaxis.set_inverted(True)
axes.yaxis.set_ticks([-2, -1, 0, 1, 2])
axes.xaxis.set_ticks([-2, -1, 0, 1, 2])
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar(im, label="c(x,y)")
plt.savefig("paper_figs/medium_three_layers.png",dpi=300, bbox_inches="tight")