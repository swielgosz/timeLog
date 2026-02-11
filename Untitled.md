import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from mldsml.wandb_utils import load_dataset

from neuralODE import constants
from neuralODE.neuralODE import load_model_wandb
from neuralODE.normalization import Normalization2BP
from neuralODE.visualizers.TrainingDataVisualizer2BP import _initial_orbital_elements
from neuralODE.visualizers.utils import prepare_save_path, save_pdf_and_png

# --- User-configurable experiment settings (no CLI parsing by request) ---
RUN_ID = "42d35h60"
PROJECT = "neuralODEs"
ENTITY = "mlds-lab"
OUTPUT_STEM = f"files/visualizations/jacobian_orbit_xy_heatmap_{RUN_ID}"
NUM_REFERENCE_TRAJS = 5  # Use 4 or 5; script will cap to available
SMA_QUANTILES = (0.1, 0.5, 0.9)
ECC_QUANTILES = (0.1, 0.9)
POINT_SIZE = 6
POINT_ALPHA = 0.9
COLORMAP = "magma"


def _select_reference_orbits(elements, *, num_reference, sma_q, ecc_q):
    """Pick orbits near target (sma, ecc) quantiles; returns indices."""
    if elements.size == 0:
        return []

    sma = elements[:, 0]
    ecc = elements[:, 1]

    sma_targets = np.quantile(sma, sma_q)
    ecc_targets = np.quantile(ecc, ecc_q)

    targets = [
        (sma_targets[0], ecc_targets[0]),
        (sma_targets[0], ecc_targets[-1]),
        (sma_targets[len(sma_targets) // 2], np.quantile(ecc, 0.5)),
        (sma_targets[-1], ecc_targets[0]),
        (sma_targets[-1], ecc_targets[-1]),
    ]

    targets = targets[: max(1, num_reference)]

    sma_range = max(np.ptp(sma), 1e-9)
    ecc_range = max(np.ptp(ecc), 1e-9)

    chosen = []
    for target_sma, target_ecc in targets:
        deltas = np.sqrt(
            ((sma - target_sma) / sma_range) ** 2
            + ((ecc - target_ecc) / ecc_range) ** 2,
        )
        for idx in np.argsort(deltas):
            if int(idx) not in chosen:
                chosen.append(int(idx))
                break

    return chosen


def _compute_jacobian_metrics(model, t_val, y_val):
    """Return (min_real, max_real) eigenvalue metrics for Jacobian at (t, y)."""

    def _eval(y_in):
        jac = jax.jacrev(lambda yy: model.func(t_val, yy))(y_in)
        eigvals = jnp.linalg.eigvals(jac)
        real_parts = jnp.real(eigvals)
        return jnp.min(real_parts), jnp.max(real_parts)

    min_real, max_real = _eval(y_val)
    return float(min_real), float(max_real)


def main():
    if RUN_ID == "REPLACE_WITH_WANDB_RUN_ID":
        raise ValueError("Set RUN_ID to the target wandb run id before executing.")

    # Load model + config from wandb
    config, model, run_obj = load_model_wandb(RUN_ID, project=PROJECT, entity=ENTITY)

    # Load dataset in physical units
    dataset_name = config.data.dataset_name
    data_dict = load_dataset(
        dataset_name,
        version="latest",
        project=PROJECT,
        entity=ENTITY,
    )

    orbit_keys = sorted(data_dict.keys())
    elements = _initial_orbital_elements(data_dict)
    if elements.shape[0] != len(orbit_keys):
        raise ValueError(
            "Mismatch between orbital element count and orbit keys; check dataset integrity.",
        )

    ref_indices = _select_reference_orbits(
        elements,
        num_reference=min(NUM_REFERENCE_TRAJS, len(orbit_keys)),
        sma_q=SMA_QUANTILES,
        ecc_q=ECC_QUANTILES,
    )
    if not ref_indices:
        raise ValueError("No reference orbits selected.")

    # Normalize the subset for model evaluation
    transform = Normalization2BP(
        l_char=constants.RADIUS_EARTH,
        mu=constants.MU_EARTH,
    )
    subset_dict = {orbit_keys[i]: data_dict[orbit_keys[i]] for i in ref_indices}
    subset_norm = transform.normalize_dataset(subset_dict)

    # Compute Jacobian metrics for each reference orbit
    orbits_payload = []
    for idx in ref_indices:
        orbit_key = orbit_keys[idx]
        phys_y = np.array(data_dict[orbit_key]["y"])
        norm_y = np.array(subset_norm[orbit_key]["y"])
        t_norm = np.array(subset_norm[orbit_key]["t"])

        if phys_y.shape[1] != norm_y.shape[1]:
            raise ValueError(f"Orbit length mismatch for {orbit_key}.")

        x = phys_y[0, :]
        y = phys_y[1, :]

        min_real_vals = []
        max_real_vals = []
        for k in range(norm_y.shape[1]):
            yk = jnp.asarray(norm_y[:, k])
            tk = float(t_norm[k])
            min_real, max_real = _compute_jacobian_metrics(model, tk, yk)
            min_real_vals.append(min_real)
            max_real_vals.append(max_real)

        sma_km, ecc = elements[idx, 0], elements[idx, 1]
        label = f"a={sma_km:.0f} km, e={ecc:.3f}"
        orbits_payload.append(
            {
                "label": label,
                "x": x,
                "y": y,
                "min_real": np.array(min_real_vals),
                "max_real": np.array(max_real_vals),
            },
        )

    # Compute global scales for color consistency
    all_min = np.concatenate([o["min_real"] for o in orbits_payload])
    all_max = np.concatenate([o["max_real"] for o in orbits_payload])
    min_vmin, min_vmax = float(all_min.min()), float(all_min.max())
    max_vmin, max_vmax = float(all_max.min()), float(all_max.max())

    # Plot
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(
        f"Jacobian Eigenvalue Heatmap in XY Plane ({run_obj.id})",
        fontsize=12,
    )

    ax_min, ax_max = axes

    for orbit in orbits_payload:
        sc_min = ax_min.scatter(
            orbit["x"],
            orbit["y"],
            c=orbit["min_real"],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            cmap=COLORMAP,
            vmin=min_vmin,
            vmax=min_vmax,
            label=orbit["label"],
        )
        sc_max = ax_max.scatter(
            orbit["x"],
            orbit["y"],
            c=orbit["max_real"],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            cmap=COLORMAP,
            vmin=max_vmin,
            vmax=max_vmax,
        )

    ax_min.set_title("min(Re(λ))")
    ax_max.set_title("max(Re(λ))")
    for ax in axes:
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.2)

    ax_min.legend(loc="upper right", fontsize=8, frameon=False)
    fig.colorbar(sc_min, ax=ax_min, shrink=0.9, pad=0.02)
    fig.colorbar(sc_max, ax=ax_max, shrink=0.9, pad=0.02)

    save_path, ext = prepare_save_path(OUTPUT_STEM, default_ext="pdf")
    pdf_path, png_path = save_pdf_and_png(
        fig,
        save_path,
        ext,
        transparent_png=True,
    )
    print(f"Saved visualization to {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
