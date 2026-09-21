import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})

# ---------------------------------------------------------
# Load solutions
# ---------------------------------------------------------

u_ref = np.load(
    "results/wave3d_utah_forge/baseline/D_NeuSA/"
    "seed_11/solutions/u_ref.npy",
    mmap_mode="r"
)

u_cqres = np.load(
    "results/wave3d_utah_forge/baseline/QRes/"
    "seed_11/solutions/u_pred.npy",
    mmap_mode="r"
)

u_neusa = np.load(
    "results/wave3d_utah_forge/baseline/NeuSA/"
    "seed_11/solutions/u_pred.npy",
    mmap_mode="r"
)

u_dneusa = np.load(
    "results/wave3d_utah_forge/baseline/D_NeuSA/"
    "seed_11/solutions/u_pred.npy",
    mmap_mode="r"
)

# ---------------------------------------------------------
# Domain
# ---------------------------------------------------------

domain_x = [-2, 2]
domain_y = [-2, 2]
domain_z = [-2, 2]
domain_t = [0, 3]

n_t, n_x, n_y, n_z = u_ref.shape

xs = np.linspace(*domain_x, n_x)
ys = np.linspace(*domain_y, n_y)
zs = np.linspace(*domain_z, n_z)
ts = np.linspace(*domain_t, n_t)

# ---------------------------------------------------------
# Time
# ---------------------------------------------------------

t_plot = 3.0
it = np.argmin(np.abs(ts - t_plot))

# ---------------------------------------------------------
# Slice positions
# ---------------------------------------------------------

x_slice = -1.0
y_slice = 0.0
z_slice = 0.0

ix = np.argmin(np.abs(xs - x_slice))
iy = np.argmin(np.abs(ys - y_slice))
iz = np.argmin(np.abs(zs - z_slice))

# ---------------------------------------------------------
# Extract slices
# ---------------------------------------------------------

def get_slices(u):
    """
    Returns:
        x = -1 -> field in (y,z)
        y = 0  -> field in (x,z)
        z = 0  -> field in (x,y)
    """

    u_t = u[it]

    slice_x = u_t[ix, :, :]     # (y,z)
    slice_y = u_t[:, iy, :]     # (x,z)
    slice_z = u_t[:, :, iz]     # (x,y)

    return slice_x, slice_y, slice_z


ref_slices = get_slices(u_ref)
cqres_slices = get_slices(u_cqres)
neusa_slices = get_slices(u_neusa)
dneusa_slices = get_slices(u_dneusa)

# ---------------------------------------------------------
# Common color scale
# ---------------------------------------------------------

# Use the reference solution to define one common scale
# for all methods and all planes.
vmax = max(
    np.abs(s).max()
    for s in ref_slices
)
vmin = -vmax

# ---------------------------------------------------------
# Plot configuration
# ---------------------------------------------------------

solutions = [
    ("QRes", cqres_slices),
    ("NeuSA", neusa_slices),
    ("D-NeuSA", dneusa_slices),
    ("Reference", ref_slices),
]

plane_titles = [
    r"$yz$-plane ($x=-1$)",
    r"$xz$-plane ($y=0$)",
    r"$xy$-plane ($z=0$)",
]

plane_info = [
    # x = -1 -> yz plane
    {
        "extent": [*domain_y, *domain_z],
        "xlabel": r"$y$",
        "ylabel": r"$z$",
    },

    # y = 0 -> xz plane
    {
        "extent": [*domain_x, *domain_z],
        "xlabel": r"$x$",
        "ylabel": r"$z$",
    },

    # z = 0 -> xy plane
    {
        "extent": [*domain_x, *domain_y],
        "xlabel": r"$x$",
        "ylabel": r"$y$",
    },
]

# ---------------------------------------------------------
# Figure
# ---------------------------------------------------------

fig, axs = plt.subplots(
    4,
    3,
    figsize=(9.5, 10.5)
)

for row, (model_name, slices) in enumerate(solutions):

    for col, (slice_data, info) in enumerate(
        zip(slices, plane_info)
    ):

        ax = axs[row, col]

        im = ax.imshow(
            slice_data.T,
            cmap="seismic",
            extent=info["extent"],
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            interpolation="bicubic"
        )

        # Same orientation convention as previous figures
        ax.invert_yaxis()

        # ---------------------------------------------
        # Ticks
        # ---------------------------------------------

        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=10,
            pad=3
        )

        # ---------------------------------------------
        # Column titles
        # ---------------------------------------------

        if row == 0:
            ax.set_title(
                plane_titles[col],
                fontsize=12,
                pad=8
            )

        # ---------------------------------------------
        # Coordinate labels
        # ---------------------------------------------

        # x-coordinate label only on bottom row
        if row == 3:
            ax.set_xlabel(
                info["xlabel"],
                fontsize=12,
                labelpad=4
            )

        # Correct vertical coordinate for every column
        ax.set_ylabel(
            info["ylabel"],
            fontsize=12,
            labelpad=-1
        )

    # -----------------------------------------------------
    # Model name
    # -----------------------------------------------------

    axs[row, 0].text(
        -0.38,
        0.5,
        model_name,
        transform=axs[row, 0].transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=12,
        fontweight="bold"
        if model_name == "D-NeuSA"
        else "normal"
    )

# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------

fig.subplots_adjust(
    left=0.14,
    right=0.86,
    bottom=0.07,
    top=0.92,
    wspace=0.30,
    hspace=0.22
)

# ---------------------------------------------------------
# Colorbar
# ---------------------------------------------------------

cbar_ax = fig.add_axes([
    0.89,
    0.16,
    0.018,
    0.68
])

cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Amplitude", fontsize=12)
cbar.ax.tick_params(labelsize=10)

# ---------------------------------------------------------
# Time
# ---------------------------------------------------------

fig.suptitle(
    fr"$t={ts[it]:.1f}$",
    y=0.975,
    fontsize=13
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

plt.savefig(
    "paper_figs/wave3d_utah_forge_slices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)