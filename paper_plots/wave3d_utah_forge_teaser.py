import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors
import numpy as np
import scipy.interpolate as spinterp
import torch
import argparse
import os


def plot_orthogonal_slices(x, y, z, u, cmap="seismic", vmax=None, elev=30, azim=-135,
                           ax=None, refine=False, center=(0., 0., 0.)):
    """
    u: 3D numpy array with shape (Nx, Ny, Nz)
    Plots slices at x=x0, y=y0, z=z0 on cube faces.
    """
    cmap = cm.get_cmap(cmap)
    Nx, Ny, Nz = u.shape
    assert len(x) == Nx and len(y) == Ny and len(z) == Nz, "Dimensions of x, y, z must match dimensions of u"

    x0, y0, z0 = center
    if refine:  # Interpolate to a finer grid for better visualization
        interpolator = spinterp.RegularGridInterpolator((x, y, z), u, method="cubic")
        x_refined = np.linspace(x[0], x[-1], 8 * Nx)
        y_refined = np.linspace(y[0], y[-1], 8 * Ny)
        z_refined = np.linspace(z[0], z[-1], 8 * Nz)
        Y_yz, Z_yz = np.meshgrid(y_refined, z_refined, indexing="ij")
        X_xz, Z_xz = np.meshgrid(x_refined, z_refined, indexing="ij")
        X_xy, Y_xy = np.meshgrid(x_refined, y_refined, indexing="ij")
        X_yz = np.zeros_like(Y_yz) + x0
        Y_xz = np.zeros_like(X_xz) + y0
        Z_xy = np.zeros_like(X_xy) + z0

        xi_yz = np.stack((X_yz, Y_yz, Z_yz), axis=-1)
        slice_x0 = interpolator(xi_yz)
        xi_xz = np.stack((X_xz, Y_xz, Z_xz), axis=-1)
        slice_y0 = interpolator(xi_xz)
        xi_xy = np.stack((X_xy, Y_xy, Z_xy), axis=-1)
        slice_z0 = interpolator(xi_xy)
    else:
        # Coordinate grids for surfaces
        Y_yz, Z_yz = np.meshgrid(y, z, indexing="ij")
        X_xz, Z_xz = np.meshgrid(x, z, indexing="ij")
        X_xy, Y_xy = np.meshgrid(x, y, indexing="ij")
        ind_x0 = np.argmin(np.abs(x - x0))
        ind_y0 = np.argmin(np.abs(y - y0))
        ind_z0 = np.argmin(np.abs(z - z0))
        # Extract the three orthogonal slices
        slice_x0 = u[ind_x0, :, :]     # (Ny, Nz)
        slice_y0 = u[:, ind_y0, :]     # (Nx, Nz)
        slice_z0 = u[:, :, ind_z0]     # (Nx, Ny)

    # Normalize color range
    if vmax is None:
        # vmax = np.quantile(np.abs(u), 0.999)
        vmax = 0.8 * np.abs(u.numpy()).max()
    vmin = -vmax

    def normed(arr):
        if np.isclose(vmax, vmin):
            return 0.5 + 0 * arr
        else:
            return (arr - vmin) / (vmax - vmin)

    # Create axis
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
    else:
        fig = ax.get_figure()
    ax.view_init(elev=elev, azim=azim)
    ax.zaxis.set_inverted(True)

    # Plot slice at x = 0 (left face)
    _ = ax.plot_surface(0 * Z_yz + x[0].item(), Y_yz, Z_yz, rstride=1, cstride=1, shade=False,
                        facecolors=cmap(normed(slice_x0)))
    # Plot slice at y = 0 (right face)
    _ = ax.plot_surface(X_xz, 0 * X_xz + y[0].item(), Z_xz, rstride=1, cstride=1, shade=False,
                        facecolors=cmap(normed(slice_y0)))
    # Plot slice at z = 0 (top face)
    _ = ax.plot_surface(X_xy, Y_xy, 0 * X_xy + z[0].item(), rstride=1, cstride=1, shade=False,
                        facecolors=cmap(normed(slice_z0)))

    x0, y0, z0 = x[0].item(), y[0].item(), z[0].item()
    x1, y1, z1 = x[-1].item(), y[-1].item(), z[-1].item()
    # the one bad point that sits behind the plot. We don't want to plot any lines coming from there
    bad_point = (x1, y1, z1)
    linewidth = 1.0
    # Lines parallel to x
    for y_point in (y0, y1):
        for z_point in (z0, z1):
            if bad_point in {(x0, y_point, z_point), ((x1, y_point, z_point))}:
                pass
            else:
                ax.plot([x0, x1], [y_point, y_point], [z_point, z_point], color="black", zorder=10, linewidth=linewidth)
    # Lines parallel to y
    for x_point in (x0, x1):
        for z_point in (z0, z1):
            if bad_point in {(x_point, y0, z_point), ((x_point, y1, z_point))}:
                pass
            else:
                ax.plot([x_point, x_point], [y0, y1], [z_point, z_point], color="black", zorder=10, linewidth=linewidth)
    # Lines parallel to z
    for x_point in (x0, x1):
        for y_point in (y0, y1):
            if bad_point in {(x_point, y_point, z0), (x_point, y_point, z1)}:
                pass
            else:
                ax.plot([x_point, x_point], [y_point, y_point], [z0, z1], color="black", zorder=10, linewidth=linewidth)   
    # norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.5, aspect=10, pad=0.001,
    #              location="bottom", orientation="horizontal")
    # plt.tight_layout()

    return fig

def plot_prediction_vs_groundtruth(ts,u_dneusa,u_neusa,u_gt,extent=[0, 1, 0, 1, 0, 1],save_path=None,cmap="seismic", refine=False):
    
    elev = 30
    azim = -135
    center = (0., 0., -1.)

    num_frames = 5
    idxs = np.linspace(60, len(ts)-1, num_frames, dtype=int)
    selected_ts = [ts[i] for i in idxs]
    vmax_gt = 0.5 * np.abs(u_gt[idxs[1]]).max()
    # vmax_gt = 0.5
    vmin_gt = -vmax_gt

    fig, axs = plt.subplots(3, num_frames, figsize=(12,6), subplot_kw={"projection": "3d"})
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # left, bottom, width, height
    norm = mcolors.Normalize(vmin=vmin_gt, vmax=vmax_gt)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable=mappable, cax=cbar_ax).set_label("Amplitude")
    plt.subplots_adjust(left = 0.11, right=0.85, top=0.92, bottom=0.08, wspace=0.02, hspace=0.02)
    



    x = np.linspace(extent[0], extent[1], u_neusa.shape[1])
    y = np.linspace(extent[2], extent[3], u_neusa.shape[2])
    z = np.linspace(extent[4], extent[5], u_neusa.shape[3])


    for j, i in enumerate(idxs):
        for k in range(3):
            axs[k, j].set_axis_off()
        # if transpose == True:
        #     u_dneusa_plot = u_dneusa[i].T 
        #     u_neusa_plot = u_neusa[i].T
        #     u_gt_plot = u_gt[i].T
        # else:
        u_dneusa_plot = u_dneusa[i]
        u_gt_plot = u_gt[i]
        u_neusa_plot = u_neusa[i]
        # im1 = axs[0, j].imshow(u_neusa_plot, cmap=cmap, extent=extent, vmin=vmin_gt, vmax=vmax_gt,interpolation=interpolation)
        _ = plot_orthogonal_slices(x, y, z, u_neusa_plot, center=center,
                                   cmap=cmap, vmax=vmax_gt, elev=elev, azim=azim, ax=axs[0, j], refine=refine)

        

        axs[0, j].set_title(f"t = {selected_ts[j]:.2f}")
        # axs[0, j].set_xticks([])
        # axs[0, j].set_yticks([])
        # axs[0, j].tick_params(labelbottom=False, labelleft=False)

        rmse = np.sqrt(np.sum((u_neusa[i]-u_gt[i])**2)/np.sum((u_gt[i])**2))
        bbox = axs[0, j].get_position()
        # if j != 0:
        fig.text(x=bbox.x0 + 0.65*bbox.width,
                    y=bbox.y0 + 0.045,
                s=f"rMSE={rmse:.3f}",
                # transform=axs[0, j].transAxes,
                ha="left", va="top",
                fontsize=8,
                color="black"
                )

        # im2 = axs[1, j].imshow(u_dneusa_plot, cmap=cmap, extent=extent, vmin=vmin_gt, vmax=vmax_gt,interpolation=interpolation)
        _ = plot_orthogonal_slices(x, y, z, u_dneusa_plot, center=center,
                                   cmap=cmap, vmax=vmax_gt, elev=elev, azim=azim, ax=axs[1, j], refine=refine)
        # axs[1, j].set_xticks([])
        # axs[1, j].set_yticks([])
        # axs[1, j].tick_params(labelbottom=False, labelleft=False)
        rmse = np.sqrt(np.sum((u_dneusa[i]-u_gt[i])**2)/np.sum((u_gt[i])**2))
        bbox = axs[1, j].get_position()
        # if j != 0:
        fig.text(x=bbox.x0 + 0.65*bbox.width,
                    y=bbox.y0 + 0.045,
                s=f"rMSE={rmse:.3f}",
                # transform=axs[0, j].transAxes,
                ha="left", va="top",
                fontsize=8,
                color="black"
                )

        # im3 = axs[2, j].imshow(u_gt_plot, cmap=cmap, extent=extent, vmin=vmin_gt, vmax=vmax_gt,interpolation=interpolation)
        _ = plot_orthogonal_slices(x, y, z, u_gt_plot, center=center,
                                   cmap=cmap, vmax=vmax_gt, elev=elev, azim=azim, ax=axs[2, j], refine=refine)
        # axs[2, j].set_xticks([])
        # axs[2, j].set_yticks([])
        # axs[2, j].tick_params(labelbottom=False, labelleft=False)

    row_labels = ["NeuSA", "D-NeuSA", "Reference"]

    for i, label in enumerate(row_labels):
        ax = axs[i, 0]
        bbox = ax.get_position()

        fig.text(
            x=bbox.x0 - 0.015,
            y=bbox.y0 + bbox.height / 2,
            s=label,
            ha="right",
            va="center",
            rotation=90,
            fontsize=12,
            fontweight="bold" if label == "D-NeuSA" else "normal"
        )
        
    # axs[0, 0].set_ylabel("NeuSA", fontsize=12)
    # axs[1, 0].set_ylabel("D-NeuSA",fontsize=12, fontweight='bold')
    # axs[2, 0].set_ylabel("Reference", fontsize=12)




    # fig.tight_layout()


    if save_path:
        plt.savefig(save_path, dpi=300,bbox_inches='tight')

parser = argparse.ArgumentParser(description="Plotting script for Utah FORGE")
parser.add_argument("--scenario", default="first", choices=["first", "second"], help="Which scenario to plot")
parser.add_argument("--refine", action="store_true", help="Whether to refine the grid for smoother visualizations (can be slow)")
args = parser.parse_args()
scenario = args.scenario
refine = args.refine

dneusa_path = "results/wave3d_utah_forge/baseline/D_NeuSA/seed_11/solutions"
gt_path = "results/wave3d_utah_forge/baseline/D_NeuSA/seed_11/solutions"
neusa_path = "results/wave3d_utah_forge/baseline/NeuSA/seed_11/solutions"

u_dneusa = np.load(os.path.join(dneusa_path, "u_pred.npy"))
u_gt = np.load(os.path.join(dneusa_path, "u_ref.npy"))
u_neusa = np.load(os.path.join(neusa_path, "u_pred.npy"))

#Transposing the axes to get the correct order
u_dneusa = np.transpose(u_dneusa, (0, 3, 2, 1))
u_gt = np.transpose(u_gt, (0, 3, 2, 1))
u_neusa = np.transpose(u_neusa, (0, 3, 2, 1))


# domain
domain_x = [-2,2]
domain_y = [-2,2]
domain_z = [-2,2]
domain_t = [0,3]

n_t = 301
ts = torch.linspace(*domain_t,n_t)


save_path = f"paper_figs/wave3d_utah_forge_teaser.png"

plot_prediction_vs_groundtruth(ts=ts,u_dneusa=u_dneusa,u_neusa=u_neusa,u_gt=u_gt, refine=refine,
                               extent=[*domain_x,*domain_y, *domain_z],save_path=save_path,cmap="seismic")