``` python
"""
Train LatentODEv2 with orbital-element encoder features on 2BP planar orbits.

Identical pipeline to train_latent_v2.py but uses latent_2bp_oe.yaml, which:
  - Feeds (E, hz, ex, ey, cos_nu, sin_nu) to the encoder instead of raw Cartesian
  - Enables stochastic encoder + KL regularization to prevent memorization
  - Uses linear_residual dynamics

Why this helps generalization:
  E, hz, ex, ey are CONSERVED along each orbit — all segments of the same
  orbit share identical values for these features.  The encoder is forced to
  learn orbit-shape -> latent-shape mapping that generalizes to unseen (a,e).

Usage:
    python scripts/training/train_latent_2bp_oe.py
"""

import os
from pathlib import Path

import numpy as np

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"
)
os.environ["OMP_NUM_THREADS"] = "8"

import jax.random as jr
from mldsml.config_utils import load_config, make_all_configs
from mldsml.wandb_utils import init_wandb, load_dataset

import neuralODE
import wandb
from neuralODE import constants
from neuralODE.data import format_data, sample_data
from neuralODE.latentODEv2 import (
    augment_latent_v2_config,
    build_latent_model_v2,
    export_model_config,
    save_latent_v2_checkpoint,
    save_latent_v2_model,
    save_xy_reconstruction_plot,
    train_latent_v2_model,
)
from neuralODE.normalization import Normalization2BP

neuralODE_path = neuralODE.__path__[0]


def _denormalize_states(y_normalized, transform):
    y_physical = np.array(y_normalized, copy=True)
    y_physical[:, 0:3] *= transform.l_char
    y_physical[:, 3:6] *= transform.v_char
    return y_physical


def _log_reconstruction_plots(config, run, model, dataset, transform, *, key):
    orbit_indices = config.parameters.get("latent_plot_orbits", [0, 1])
    if isinstance(orbit_indices, int):
        orbit_indices = [int(orbit_indices)]

    output_dir = Path("files/figures/latent_2bp_oe")
    output_dir.mkdir(parents=True, exist_ok=True)

    use_mean = bool(config.parameters.latent_eval_use_mean)
    num_orbits = int(dataset["y"].shape[0])

    for orbit_idx in orbit_indices:
        if orbit_idx < 0 or orbit_idx >= num_orbits:
            continue
        key, subkey = jr.split(key)
        ts = dataset["t"][orbit_idx]
        y_true = dataset["y"][orbit_idx]
        z0, _, _ = model.encode(ts, y_true, key=subkey, deterministic=use_mean)
        y_hat = model.decode(ts, z0, y0_context=y_true[0])

        y_true_np = np.asarray(y_true)
        y_hat_np = np.asarray(y_hat)
        if transform is not None:
            y_true_np = _denormalize_states(y_true_np, transform)
            y_hat_np = _denormalize_states(y_hat_np, transform)

        plot_path = output_dir / f"latent_2bp_oe_{run.id}_orbit_{orbit_idx}.png"
        save_xy_reconstruction_plot(
            plot_path,
            y_true_np[:, :2],
            y_hat_np[:, :2],
            title=f"Latent ODE (orbital elements) reconstruction orbit {orbit_idx}",
        )
        wandb.log({f"reconstruction/orbit_{orbit_idx}": wandb.Image(str(plot_path))})


def main():
    cfg_path = f"{neuralODE_path}/config/latent_2bp_oe.yaml"
    default_config = load_config(cfg_path)
    sweep_config = load_config(cfg_path)
    config = make_all_configs(default_config, sweep_config)[0]
    config = augment_latent_v2_config(config)

    run = init_wandb(
        config=config,
        group=config.wandb.group,
        mode="online",
        reinit=True,
    )

    data_dict = load_dataset(
        config.data.dataset_name,
        version="latest",
        project="neuralODEs",
        entity="mlds-lab",
    )
    data_dict = sample_data(data_dict, config.parameters.num_trajs)

    transform = None
    if config.data.problem == "2BP":
        transform = Normalization2BP(
            l_char=constants.RADIUS_EARTH,
            mu=constants.MU_EARTH,
        )
        data_dict = transform.normalize_dataset(data_dict)

    dataset = format_data(data_dict)
    data_size = int(dataset["y"].shape[-1])
    seed = int(config.parameters.seed)
    model = build_latent_model_v2(config, data_size=data_size, key=jr.PRNGKey(seed))

    model, training_metadata = train_latent_v2_model(config, model, dataset, seed=seed)

    config_payload = export_model_config(config, data_size=data_size)
    checkpoint_root = Path(config.parameters.latent_checkpoint_dir)
    checkpoint_path = checkpoint_root / f"latent_2bp_oe_{run.id}.eqx"
    save_latent_v2_checkpoint(config_payload, model, checkpoint_path)
    save_latent_v2_model(config_payload, model, run.id, run=run)

    wandb.log(
        {
            "segment_ratio_final": (
                training_metadata.get("segment_ratio_schedule", [None])[-1]
                if training_metadata.get("segment_ratio_schedule")
                else None
            ),
            "num_total_orbits": training_metadata.get("num_total_orbits"),
            "latent/best_val_loss": training_metadata.get("best_val_loss"),
        },
    )

    plot_key = jr.PRNGKey(seed + 1)
    _log_reconstruction_plots(config, run, model, dataset, transform, key=plot_key)

    wandb.save(str(checkpoint_path))
    wandb.finish()


if __name__ == "__main__":
    main()
```