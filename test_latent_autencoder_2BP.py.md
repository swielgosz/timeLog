``` python
"""
Test observation-space roundtrip behavior of a trained 2BP LatentODE autoencoder.

Functionality:
- Load a trained latent ODE checkpoint.
- Load normalized 2BP observations.
- Select one observed condition by orbit index and time-step index.
- Encode the selected orbit, decode it back, and compare true vs reconstructed state.
- Report where the decoded condition returns in observation space (nearest observed state).
- Save a JSON report and an XY trajectory plot.

Example usage:
    python scripts/training/test_latent_autoencoder_2BP.py
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

try:
    # Package import path (repo root on PYTHONPATH).
    from scripts.training.train_latent_2BP import load_2bp_dataset, load_latent_checkpoint
except ModuleNotFoundError:
    # Local fallback for direct absolute-path script execution.
    from train_latent_2BP import load_2bp_dataset, load_latent_checkpoint


def resolve_index(index: int, size: int, name: str) -> int:
    """
    Resolve positive/negative Python-style indices and validate bounds.

    Example usage:
        orbit_idx = resolve_index(index=-1, size=10, name="orbit_index")
    """
    if size <= 0:
        raise ValueError(f"{name} cannot be resolved because size={size}.")
    idx = int(index)
    if idx < 0:
        idx += int(size)
    if idx < 0 or idx >= int(size):
        raise ValueError(f"{name}={index} is out of range for size={size}.")
    return idx


def nearest_observation_index(
    query_state: np.ndarray,
    candidate_states: np.ndarray,
) -> tuple[int, float]:
    """
    Find the nearest observed state (Euclidean distance) to a query state.

    Example usage:
        idx, dist = nearest_observation_index(y_hat[0], ys[:, 0, :])
    """
    candidates = np.asarray(candidate_states, dtype=float)
    if candidates.ndim != 2:
        raise ValueError("candidate_states must have shape (N, D).")
    query = np.asarray(query_state, dtype=float).reshape(1, -1)
    distances = np.linalg.norm(candidates - query, axis=1)
    best_idx = int(np.argmin(distances))
    return best_idx, float(distances[best_idx])


def compute_roundtrip_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    condition_step_index: int,
) -> dict:
    """
    Compute compact reconstruction metrics for one orbit roundtrip.

    Example usage:
        metrics = compute_roundtrip_metrics(y_true, y_hat, condition_step_index=0)
    """
    y_true_np = np.asarray(y_true, dtype=float)
    y_hat_np = np.asarray(y_hat, dtype=float)
    if y_true_np.shape != y_hat_np.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true_np.shape} vs y_hat {y_hat_np.shape}.",
        )
    if y_true_np.ndim != 2 or y_true_np.shape[1] < 6:
        raise ValueError("Expected y arrays with shape (T, 6+).")

    cond_idx = resolve_index(
        index=condition_step_index,
        size=y_true_np.shape[0],
        name="condition_step_index",
    )
    err = y_hat_np - y_true_np
    return {
        "state_rmse": float(np.sqrt(np.mean(err**2))),
        "position_rmse": float(np.sqrt(np.mean(err[:, :3] ** 2))),
        "velocity_rmse": float(np.sqrt(np.mean(err[:, 3:6] ** 2))),
        "condition_state_error_norm": float(np.linalg.norm(err[cond_idx])),
        "condition_position_error_norm": float(np.linalg.norm(err[cond_idx, :3])),
        "condition_velocity_error_norm": float(np.linalg.norm(err[cond_idx, 3:6])),
        "final_state_error_norm": float(np.linalg.norm(err[-1])),
        "final_position_error_norm": float(np.linalg.norm(err[-1, :3])),
        "final_velocity_error_norm": float(np.linalg.norm(err[-1, 3:6])),
    }


def save_xy_plot(
    ts: np.ndarray,
    y_true: np.ndarray,
    y_hat: np.ndarray,
    orbit_index: int,
    condition_step_index: int,
    output_path: Path,
) -> None:
    """
    Save XY plot for true vs reconstructed orbit for quick visual inspection.

    Example usage:
        save_xy_plot(ts, y_true, y_hat, 0, 0, Path("files/figures/roundtrip.png"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    t_np = np.asarray(ts, dtype=float)
    yt = np.asarray(y_true, dtype=float)
    yh = np.asarray(y_hat, dtype=float)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(yt[:, 0], yt[:, 1], lw=2.0, label="true orbit")
    ax.plot(yh[:, 0], yh[:, 1], lw=2.0, ls="--", label="decoded orbit")
    ax.scatter(
        [yt[condition_step_index, 0]],
        [yt[condition_step_index, 1]],
        s=42,
        marker="o",
        label="selected observed condition",
    )
    ax.scatter(
        [yh[condition_step_index, 0]],
        [yh[condition_step_index, 1]],
        s=42,
        marker="x",
        label="decoded at selected step",
    )
    ax.set_xlabel("x (normalized)")
    ax.set_ylabel("y (normalized)")
    ax.set_title(
        f"Latent roundtrip orbit={orbit_index}, step={condition_step_index}, "
        f"T={len(t_np)}",
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """
    Run a single observation-space roundtrip check for a trained latent ODE.
    """
    config = {
        "checkpoint_path": "files/models/best_latent_2bp_example.eqx",
        "dataset": "complex_TBP_planar_10_train",
        "num_trajs": 10,
        "seed": 0,
        "orbit_index": 0,
        "condition_step_index": 0,
        "use_latent_mean": True,
        "output_json": "files/results/test_latent_autoencoder_2BP.json",
        "output_plot": "files/figures/test_latent_autoencoder_2BP/roundtrip_xy.png",
    }

    checkpoint_path = Path(config["checkpoint_path"])
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Update config['checkpoint_path'] to a trained latent ODE checkpoint.",
        )

    _checkpoint_cfg, model = load_latent_checkpoint(checkpoint_path)
    t_shared, ys, _transform = load_2bp_dataset(
        num_trajs=int(config["num_trajs"]),
        dataset_name=str(config["dataset"]),
        key=jr.PRNGKey(int(config["seed"])),
    )
    ys_np = np.asarray(ys, dtype=float)

    orbit_idx = resolve_index(
        index=int(config["orbit_index"]),
        size=ys_np.shape[0],
        name="orbit_index",
    )
    cond_idx = resolve_index(
        index=int(config["condition_step_index"]),
        size=ys_np.shape[1],
        name="condition_step_index",
    )

    key = jr.PRNGKey(int(config["seed"]))
    z_sample, z_mean, _logstd = model.encode(t_shared, ys[orbit_idx], key=key)
    z0 = z_mean if bool(config["use_latent_mean"]) else z_sample
    y_hat = np.asarray(model.decode(t_shared, z0), dtype=float)
    y_true = ys_np[orbit_idx]

    metrics = compute_roundtrip_metrics(y_true, y_hat, cond_idx)
    nearest_idx, nearest_dist = nearest_observation_index(
        query_state=y_hat[cond_idx],
        candidate_states=ys_np[:, cond_idx, :],
    )

    payload = {
        "config": config,
        "selection": {
            "orbit_index": int(orbit_idx),
            "condition_step_index": int(cond_idx),
            "selected_observed_state": y_true[cond_idx].tolist(),
            "decoded_state_at_selected_step": y_hat[cond_idx].tolist(),
        },
        "nearest_return_in_observation_space": {
            "time_step_index": int(cond_idx),
            "nearest_orbit_index": int(nearest_idx),
            "distance_norm": float(nearest_dist),
            "nearest_observed_state": ys_np[nearest_idx, cond_idx, :].tolist(),
        },
        "metrics": metrics,
    }

    output_json = Path(config["output_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))

    save_xy_plot(
        ts=np.asarray(t_shared, dtype=float),
        y_true=y_true,
        y_hat=y_hat,
        orbit_index=orbit_idx,
        condition_step_index=cond_idx,
        output_path=Path(config["output_plot"]),
    )

    print("Latent autoencoder roundtrip check complete.")
    print(f"selected orbit index: {orbit_idx}")
    print(f"selected condition step index: {cond_idx}")
    print(f"decoded nearest observed orbit index: {nearest_idx}")
    print(f"decoded nearest observed distance: {nearest_dist:.6e}")
    print(f"state RMSE: {metrics['state_rmse']:.6e}")
    print(f"position RMSE: {metrics['position_rmse']:.6e}")
    print(f"velocity RMSE: {metrics['velocity_rmse']:.6e}")
    print(f"saved report: {output_json}")
    print(f"saved plot: {config['output_plot']}")


if __name__ == "__main__":
    main()
```