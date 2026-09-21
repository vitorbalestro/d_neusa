import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from scipy import interpolate as spinterp
import torch

plt.rcParams.update({'font.size': 15})


class InterpolatorWrapper:
    def __init__(self, points, data):
        self.interpolator = spinterp.RegularGridInterpolator(points, data)

    def __call__(self, x):
        # points is a (..., 3) array. We need to change it to (N, 4), pass to the interpolator, then reshape
        shape = x.shape[:-1]
        x_flat = x.reshape(-1, 3)
        interpolated_flat = self.interpolator(x_flat)
        return interpolated_flat.reshape(shape)


def plot_3d_domain(x, y, z, vel_profile, cmap="jet", elev=30, azim=-135, ax=None, refine=False):
    """
    Plots slices at x=x0, y=y0, z=z0 on cube faces, amounting to an isometric view of the domain from the outside.
    """
    cmap = cm.get_cmap(cmap)
    Nx, Ny, Nz = len(x), len(y), len(z)

    x0, y0, z0 = x[0].item(), y[0].item(), z[0].item()
    x1, y1, z1 = x[-1].item(), y[-1].item(), z[-1].item()
    if refine:  # Evaluate the velocity on an finer grid for better plots
        x_refined = np.linspace(x[0], x[-1], 4 * Nx)
        y_refined = np.linspace(y[0], y[-1], 4 * Ny)
        z_refined = np.linspace(z[0], z[-1], 4 * Nz)
        Y_yz, Z_yz = np.meshgrid(y_refined, z_refined, indexing="ij")
        X_xz, Z_xz = np.meshgrid(x_refined, z_refined, indexing="ij")
        X_xy, Y_xy = np.meshgrid(x_refined, y_refined, indexing="ij")

    else:
        # Coordinate grids for surfaces
        Y_yz, Z_yz = np.meshgrid(y, z, indexing="ij")
        X_xz, Z_xz = np.meshgrid(x, z, indexing="ij")
        X_xy, Y_xy = np.meshgrid(x, y, indexing="ij")

    X_yz = np.zeros_like(Y_yz) + x0
    Y_xz = np.zeros_like(X_xz) + y0
    Z_xy = np.zeros_like(X_xy) + z0
    plane_yz = np.stack((X_yz, Y_yz, Z_yz), axis=-1)
    slice_x0 = vel_profile(torch.tensor(plane_yz)).squeeze()
    plane_xz = np.stack((X_xz, Y_xz, Z_xz), axis=-1)
    slice_y0 = vel_profile(torch.tensor(plane_xz)).squeeze()
    plane_xy = np.stack((X_xy, Y_xy, Z_xy), axis=-1)
    slice_z0 = vel_profile(torch.tensor(plane_xy)).squeeze()

    # Normalize color range
    all_velocities = np.stack([slice_x0, slice_y0, slice_z0])
    vmax = np.max(all_velocities)
    vmin = np.min(all_velocities)

    def normed(arr):
        return (arr - vmin) / (vmax - vmin)

    # Create axis
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_axis_off()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    ax.zaxis.set_inverted(True)

    # Plot slice at x = x0 (left face)
    _ = ax.plot_surface(0 * Z_yz + x0, Y_yz, Z_yz, rstride=1, cstride=1, shade=False,
                        facecolors=cmap(normed(slice_x0)))
    # Plot slice at y = y0 (left face)
    _ = ax.plot_surface(X_xz, 0 * X_xz + y0, Z_xz, rstride=1, cstride=1, shade=False,
                        facecolors=cmap(normed(slice_y0)))
    # Plot slice at z = z0 (top face)
    _ = ax.plot_surface(X_xy, Y_xy, 0 * X_xy + z0, rstride=1, cstride=1, shade=False,
                        facecolors=cmap(normed(slice_z0)))

    # the one bad point that sits behind the plot. We don't want to plot any lines coming from there
    bad_point = (x1, y1, z1)
    # Lines parallel to x
    for y_point in (y0, y1):
        for z_point in (z0, z1):
            if bad_point in {(x0, y_point, z_point), ((x1, y_point, z_point))}:
                pass
            else:
                plt.plot([x0, x1], [y_point, y_point], [z_point, z_point], color="black", zorder=10)
    # Lines parallel to y
    for x_point in (x0, x1):
        for z_point in (z0, z1):
            if bad_point in {(x_point, y0, z_point), ((x_point, y1, z_point))}:
                pass
            else:
                plt.plot([x_point, x_point], [y0, y1], [z_point, z_point], color="black", zorder=10)
    # Lines parallel to z
    for x_point in (x0, x1):
        for y_point in (y0, y1):
            if bad_point in {(x_point, y_point, z0), (x_point, y_point, z1)}:
                pass
            else:
                plt.plot([x_point, x_point], [y_point, y_point], [z0, z1], color="black", zorder=10)

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5, aspect=10, pad=0.001,
                 location="right", orientation="vertical", label="c(x,y,z)")
    plt.tight_layout()

    return fig

# load velocity field
data_path = "data/utah_forge/Seismic_Velocity_Model_FORGE_Site"

normalizing_factor = 10
c_as_array = []

with open(data_path,'r') as f:
    next(f)
    for line in f:
        cols = line.split()
        c_as_array.append(float(cols[5])/normalizing_factor)     


c_numpy = np.array(c_as_array).reshape(65,41,41)
c_numpy = c_numpy[0:41,:,:]
c_numpy = np.transpose(c_numpy,(2,1,0))  # put in (x,y,z) order
c = torch.tensor(c_numpy,dtype=torch.float32)
c2 = c*c

vel_profile = InterpolatorWrapper((np.linspace(-2,2,41), np.linspace(-2,2,41), np.linspace(-2,2,41)), c_numpy)

fig_3d = plot_3d_domain(np.linspace(-2,2,41), np.linspace(-2,2,41), np.linspace(-2,2,41), vel_profile,
                        cmap="viridis", elev=30, azim=-135, refine=False)
fig_3d.savefig("paper_figs/medium_utah_forge.png", dpi=300, bbox_inches="tight", pad_inches=0.2)

