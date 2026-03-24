``` python
"""
Train a modular residual UDE to learn planar J2 perturbations on top of known 2BP.

Functionality:
- Loads config from YAML (same pattern as scripts/training/train.py)
- Loads dataset artifact from Weights & Biases
- Applies deterministic 80/20 train/test orbit split
- Trains residual model: ydot = f_2bp(y) + f_theta(y, t)
- Logs metrics and saves model checkpoint + artifact

Example usage:
    python scripts/training/train_ude_zonal_j2.py
"""

# ruff: noqa: E402

import os
import sys
from pathlib import Path

# Force CPU platform (must be set before importing jax)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import jax.random as jr
import neuralODE
import wandb
from mldsml.config_utils import load_config, make_all_configs
from mldsml.wandb_utils import init_wandb, load_dataset

# Make local helper imports robust from repo root or scripts/training.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))

from neuralODE import constants
from neuralODE.normalization import Normalization2BP
from ude_zonal_j2j6_helpers import (
    UDEResidualDynamics,
    dataset_dict_to_arrays,
    precompute_base_trajectories,
    save_checkpoint,
    split_orbits_allow_single,
    train_loop,
)

neuralODE_path = neuralODE.__path__[0]
CONFIG_PATH = f"{neuralODE_path}/config/ude_j2.yaml"


def main() -> None:
    # Match train.py config flow (default+sweep -> first config).
    default_config = load_config(CONFIG_PATH)
    sweep_config = load_config(CONFIG_PATH)
    configs = make_all_configs(default_config, sweep_config)
    config = configs[0]
    params = config.parameters
    length_strategy = list(getattr(params, "length_strategy", [[0.0, 1.0]]))
    num_trajs = int(getattr(params, "num_trajs", -1))
    normalize_2bp = bool(getattr(params, "normalize_2bp", True))

    run = init_wandb(
        config=config,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
        mode="online",
        reinit=True,
    )

    data_dict = load_dataset(
        config.data.dataset_name,
        version=config.data.dataset_version,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )
    if normalize_2bp:
        normalizer = Normalization2BP(
            l_char=constants.RADIUS_EARTH,
            mu=constants.MU_EARTH,
        )
        data_dict = normalizer.normalize_dataset(data_dict)

    arrays = dataset_dict_to_arrays(data_dict)

    model_mu = 1.0 if normalize_2bp else float(params.mu)
    arrays = precompute_base_trajectories(
        arrays,
        mu=model_mu,
        rtol=float(params.rtol),
        atol=float(params.atol),
        max_steps=int(params.solver_max_steps),
    )

    if num_trajs > 0:
        n_keep = min(num_trajs, int(arrays["y"].shape[0]))
        arrays = {
            "orbit_keys": arrays["orbit_keys"][:n_keep],
            "t": arrays["t"][:n_keep],
            "y": arrays["y"][:n_keep],
            "mask": arrays["mask"][:n_keep],
            "base_y": arrays["base_y"][:n_keep],
        }

    train_data, test_data, train_idx, test_idx = split_orbits_allow_single(
        arrays,
        train_fraction=float(params.train_fraction),
        seed=int(params.seed),
    )

    model = UDEResidualDynamics(
        mu=model_mu,
        width=int(params.width),
        depth=int(params.depth),
        accel_scale=float(params.accel_scale),
        key=jr.PRNGKey(int(params.seed)),
    )

    run_config = {
        "dataset_name": str(config.data.dataset_name),
        "dataset_version": str(config.data.dataset_version),
        "seed": int(params.seed),
        "train_fraction": float(params.train_fraction),
        "num_trajs": num_trajs,
        "normalize_2bp": normalize_2bp,
        "model_width": int(params.width),
        "model_depth": int(params.depth),
        "model_accel_scale": float(params.accel_scale),
        "mu_earth_km3_s2": float(params.mu),
        "model_mu": float(model_mu),
        "learning_rate": float(params.lr),
        "train_steps": int(params.train_steps),
        "batch_size": int(params.batch_size),
        "l2_weight": float(params.l2_weight),
        "eval_every": int(params.eval_every),
        "solver_rtol": float(params.rtol),
        "solver_atol": float(params.atol),
        "solver_max_steps": int(params.solver_max_steps),
        "length_strategy": length_strategy,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "config_path": CONFIG_PATH,
    }
    wandb.config.update(run_config, allow_val_change=True)

    model, history = train_loop(
        model,
        train_data,
        test_data,
        lr=float(params.lr),
        steps=int(params.train_steps),
        batch_size=int(params.batch_size),
        l2_weight=float(params.l2_weight),
        rtol=float(params.rtol),
        atol=float(params.atol),
        max_steps=int(params.solver_max_steps),
        seed=int(params.seed),
        eval_every=int(params.eval_every),
        length_strategy=length_strategy,
        log_fn=wandb.log,
    )

    checkpoint_dir = Path(str(config.output.checkpoint_dir))
    checkpoint_path = checkpoint_dir / f"ude_j2_{run.id}.eqx"
    checkpoint_meta = {
        "run_id": run.id,
        "model_width": int(params.width),
        "model_depth": int(params.depth),
        "model_accel_scale": float(params.accel_scale),
        "mu": float(model_mu),
        "solver_rtol": float(params.rtol),
        "solver_atol": float(params.atol),
        "solver_max_steps": int(params.solver_max_steps),
        "length_strategy": length_strategy,
        "dataset_name": str(config.data.dataset_name),
        "dataset_version": str(config.data.dataset_version),
        "seed": int(params.seed),
        "train_fraction": float(params.train_fraction),
        "num_trajs": num_trajs,
        "normalize_2bp": normalize_2bp,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "history": history,
    }
    save_checkpoint(checkpoint_path, model, metadata=checkpoint_meta)

    artifact = wandb.Artifact(
        name=f"ude-j2-model-{run.id}",
        type="model",
        metadata=checkpoint_meta,
    )
    artifact.add_file(str(checkpoint_path), name=checkpoint_path.name)
    run.log_artifact(artifact)

    summary = {
        "num_orbits_total": int(arrays["y"].shape[0]),
        "num_orbits_train": int(train_data["y"].shape[0]),
        "num_orbits_test": int(test_data["y"].shape[0]),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_test_loss": float(history["test_loss"][-1]),
        "checkpoint_path": str(checkpoint_path),
    }
    wandb.log(summary)

    print("Training complete.")
    print(f"Run id: {run.id}")
    print(f"Checkpoint: {checkpoint_path}")
    print(
        "Split sizes: "
        f"train={summary['num_orbits_train']} test={summary['num_orbits_test']}",
    )

    wandb.finish()


if __name__ == "__main__":
    main()
```