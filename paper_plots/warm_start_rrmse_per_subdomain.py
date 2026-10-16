import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

seeds = [11, 12, 14, 15, 17]

problems = {
    "allen_cahn_1d": {
        "title": "1D Allen--Cahn",
        "n_t": 1001,
        "n_slices": 100,
    },
    "allen_cahn_2d": {
        "title": "2D Allen--Cahn",
        "n_t": 20001,
        "n_slices": 1000,
    },
    "burgers1d": {
        "title": "Burgers",
        "n_t": 1001,
        "n_slices": 100,
    },
    "wave2d_three_layers": {
        "title": "Three Layers",
        "n_t": 201,
        "n_slices": 40,
    },
    "wave2d_marmousi": {
        "title": "Marmousi",
        "n_t": 201,
        "n_slices": 50,
    },
    "wave3d_utah_forge": {
        "title": "Utah FORGE",
        "n_t": 301,
        "n_slices": 60,
    },
}


# ============================================================
# Load solutions
# ============================================================

def load_solutions(equation, experiment, seed):

    solutions_dir = os.path.join(
        "results",
        equation,
        experiment,
        "D_NeuSA",
        f"seed_{seed}",
        "solutions",
    )

    u_ref = np.load(
        os.path.join(solutions_dir, "u_ref.npy")
    )

    u_pred = np.load(
        os.path.join(solutions_dir, "u_pred.npy")
    )

    return u_ref, u_pred


# ============================================================
# Local rRMSE
# ============================================================

def local_rrmse(u_ref, u_pred, n_t, n_slices):

    assert u_ref.shape[0] == n_t, (
        f"Expected {n_t} time steps in u_ref, "
        f"got {u_ref.shape[0]}."
    )

    assert u_pred.shape[0] == n_t, (
        f"Expected {n_t} time steps in u_pred, "
        f"got {u_pred.shape[0]}."
    )

    assert (n_t - 1) % n_slices == 0

    offset = (n_t - 1) // n_slices

    # Sum over all spatial dimensions first.
    # This is equivalent to computing the norm after flattening
    # each temporal subdomain, but considerably faster.
    spatial_axes = tuple(range(1, u_ref.ndim))

    squared_error_t = np.sum(
        (u_pred - u_ref) ** 2,
        axis=spatial_axes,
    )

    squared_ref_t = np.sum(
        u_ref ** 2,
        axis=spatial_axes,
    )

    errors = np.empty(n_slices)

    for k in range(n_slices):

        start = k * offset
        end = (k + 1) * offset

        # Same convention used in the NeuSA vs. D-NeuSA figure:
        # [start:end], with no duplicated interface points.
        numerator = np.sqrt(
            np.sum(squared_error_t[start:end])
        )

        denominator = np.sqrt(
            np.sum(squared_ref_t[start:end])
        )

        if denominator == 0:
            errors[k] = np.nan
        else:
            errors[k] = numerator / denominator

    return errors


# ============================================================
# Mean local rRMSE over seeds
# ============================================================

def mean_local_rrmse(
    equation,
    experiment,
    n_t,
    n_slices,
    seeds,
):

    errors = []

    for seed in seeds:

        print(
            f"{equation} | {experiment} | seed {seed}"
        )

        u_ref, u_pred = load_solutions(
            equation,
            experiment,
            seed,
        )

        rrmse = local_rrmse(
            u_ref,
            u_pred,
            n_t,
            n_slices,
        )

        errors.append(rrmse)

    errors = np.stack(
        errors,
        axis=0,
    )

    # Mean across seeds while all runs remain finite.
    mean_errors = np.mean(
        errors,
        axis=0,
    )

    # Once any seed becomes non-finite, stop the mean curve.
    # This keeps every displayed point as an average over
    # exactly the same five seeds.
    invalid = np.any(
        ~np.isfinite(errors),
        axis=0,
    )

    if np.any(invalid):
        first_invalid = np.where(invalid)[0][0]
        mean_errors[first_invalid:] = np.nan

    return mean_errors


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(11, 6),
)

axes = axes.ravel()


for ax, (equation, config) in zip(
    axes,
    problems.items(),
):

    n_t = config["n_t"]
    n_slices = config["n_slices"]

    # --------------------------------------------------------
    # Independent initialization
    # --------------------------------------------------------

    rrmse_ind = mean_local_rrmse(
        equation,
        "baseline",
        n_t,
        n_slices,
        seeds,
    )

    # --------------------------------------------------------
    # Warm start
    # --------------------------------------------------------

    rrmse_warm = mean_local_rrmse(
        equation,
        "warm_start_ablation",
        n_t,
        n_slices,
        seeds,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    subdomains = np.arange(n_slices)

    ax.plot(
        subdomains,
        rrmse_ind,
        linewidth=1.5,
        label="Independent",
    )

    ax.plot(
        subdomains,
        rrmse_warm,
        linewidth=1.5,
        label="Warm start",
    )

    ax.set_title(
        config["title"]
    )

    ax.set_xlabel(
        "Temporal subdomain index"
    )

    ax.set_yscale("log")

    ax.grid(False)


# ============================================================
# Axis labels
# ============================================================

axes[0].set_ylabel("Local rRMSE")
axes[3].set_ylabel("Local rRMSE")


# ============================================================
# Common legend
# ============================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.5, 0.01),
)

plt.tight_layout(
    rect=[0, 0.08, 1, 1]
)


# ============================================================
# Save
# ============================================================

os.makedirs(
    "paper_figs",
    exist_ok=True,
)

plt.savefig(
    "paper_figs/warm_start_rrmse_per_subdomain.png",
    dpi=300,
    bbox_inches="tight",
)

