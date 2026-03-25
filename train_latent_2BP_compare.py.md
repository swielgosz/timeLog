``` python
"""
Train a Diffrax-based latent ODE on normalized 2BP data for comparison against train.py.

This script follows the same dataset/config/wandb flow as scripts/training/train.py,
but swaps the direct NeuralODE state propagator for a latent encoder-decoder model.
The high-level design is guided by the continuous-depth formulation in Chen et al.
(2018) and the accompanying educational notebook from llSourcell, while using the
repo's JAX/Diffrax stack instead of PyTorch.

References:
- Chen et al. (2018), Neural Ordinary Differential Equations:
  https://arxiv.org/abs/1806.07366
- llSourcell notebook:
  https://github.com/llSourcell/Neural_Differential_Equations/blob/master/Neural_Ordinary_Differential_Equations.ipynb

Example usage:
    python scripts/training/train_latent_2BP_compare.py
"""

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Iterator

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8"
)
os.environ["OMP_NUM_THREADS"] = "8"

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import wandb
from mldsml.config_utils import load_config, make_all_configs
from mldsml.wandb_utils import init_wandb, load_dataset
from tqdm import tqdm

import neuralODE
from neuralODE import constants
from neuralODE.data import format_data, sample_data, segment_data, split_data
from neuralODE.dynamics import get_dynamics_class
from neuralODE.latentODE import LatentODE
from neuralODE.normalization import Normalization2BP

neuralODE_path = neuralODE.__path__[0]


def augment_latent_config(config):
    """
    Add latent-model defaults on top of the train.py-style config.

    Example usage:
        config = augment_latent_config(config)
    """
    config = copy.deepcopy(config)
    params = config.parameters
    params.setdefault("latent_hidden_size", int(params.width))
    params.setdefault("latent_size", max(8, int(params.width) // 4))
    params.setdefault("latent_width", int(params.width))
    params.setdefault("latent_depth", int(params.depth))
    params.setdefault("latent_activation", "tanh")
    params.setdefault("latent_final_activation", "tanh")
    params.setdefault("latent_drift_output_activation", "tanh")
    params.setdefault("latent_kl_weight", 1e-4)
    params.setdefault("latent_eval_use_mean", True)
    params.setdefault("latent_solver_max_steps", 4096)
    if hasattr(config, "wandb"):
        config.wandb.group = f"{config.wandb.group}-latent"
    return config


def clone_model(model):
    """Deep copy an Equinox model via in-memory serialization."""
    import io

    buffer = io.BytesIO()
    eqx.tree_serialise_leaves(buffer, model)
    buffer.seek(0)
    return eqx.tree_deserialise_leaves(buffer, model)


def kl_div(mean, logstd):
    """KL divergence to a standard normal prior."""
    var = jnp.exp(2.0 * logstd)
    return 0.5 * jnp.sum(var + mean**2 - 1.0 - 2.0 * logstd)


def export_model_config(config, *, data_size: int) -> dict:
    """Build a flat latent-model config payload compatible with existing loaders."""
    params = config.parameters
    return {
        "data_size": int(data_size),
        "dataset": config.data.dataset_name,
        "problem": config.data.problem,
        "seed": int(params.seed),
        "hidden_size": int(params.latent_hidden_size),
        "latent_size": int(params.latent_size),
        "width_size": int(params.latent_width),
        "width": int(params.latent_width),
        "depth": int(params.latent_depth),
        "latent_activation": str(params.latent_activation),
        "latent_final_activation": str(params.latent_final_activation),
        "drift_output_activation": str(params.latent_drift_output_activation),
        "solver_rtol": float(params.rtol),
        "solver_atol": float(params.atol),
        "solver_max_steps": int(params.latent_solver_max_steps),
        "encoder_type": "gru",
        "decoder_type": "linear",
        "latent_eval_use_mean": bool(params.latent_eval_use_mean),
        "latent_kl_weight": float(params.latent_kl_weight),
        "segment_length_strategy": [int(v) for v in params.segment_length_strategy],
        "steps_strategy": [int(v) for v in params.steps_strategy],
        "lr_strategy": [float(v) for v in params.lr_strategy],
    }


def save_latent_checkpoint(config_payload: dict, model: LatentODE, path: Path) -> None:
    """Save a latent checkpoint with the existing JSON-header format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(json.dumps(config_payload).encode() + b"\n")
        eqx.tree_serialise_leaves(f, model)


def save_latent_model(
    config_payload: dict, model: LatentODE, run_id: str, run=None
) -> None:
    """Log the latent model as a compatible W&B artifact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.eqx"
        save_latent_checkpoint(config_payload, model, model_path)
        artifact = wandb.Artifact(
            name=f"latent-ode-{run_id}",
            type="latent-ode-model",
            metadata={"run_id": run_id},
        )
        artifact.add_file(str(model_path), name="model.eqx")
        active_run = run if run is not None else wandb.run
        if active_run is None:
            raise RuntimeError("No active wandb run available to log artifact.")
        active_run.log_artifact(artifact)


def latent_dataloader(data, batch_size: int, *, key) -> Iterator:
    """Yield minibatches from segmented latent-ODE training data."""
    ts = data["t"]
    ys = data["y"]
    n = ys.shape[0]
    idx = jr.permutation(key, n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield ts[batch_idx], ys[batch_idx]


def has_valid_segment_batch(data: dict) -> bool:
    """Return True when segmented data has at least one `(T,D)` sample."""
    if not isinstance(data, dict) or "t" not in data or "y" not in data:
        return False
    t = jnp.asarray(data["t"])
    y = jnp.asarray(data["y"])
    if t.ndim != 2 or y.ndim != 3:
        return False
    if t.shape[0] == 0 or y.shape[0] == 0:
        return False
    return t.shape[0] == y.shape[0] and t.shape[1] == y.shape[1]


def ensure_batched(t_data, y_data):
    """Normalize trajectory tensors to `(B, T)` and `(B, T, D)` shapes."""
    t_data = jnp.asarray(t_data)
    y_data = jnp.asarray(y_data)
    if t_data.ndim == 1:
        t_data = t_data[None, :]
    if y_data.ndim == 2:
        y_data = y_data[None, :, :]
    if t_data.ndim != 2 or y_data.ndim != 3:
        raise ValueError(
            f"Expected batched shapes (B,T)/(B,T,D), got {t_data.shape} and {y_data.shape}.",
        )
    if t_data.shape[0] != y_data.shape[0] or t_data.shape[1] != y_data.shape[1]:
        raise ValueError(
            f"Mismatched time/state shapes {t_data.shape} and {y_data.shape}.",
        )
    return t_data, y_data


def build_latent_model(config, data_size, *, key):
    """Construct the latent ODE model from the augmented config."""
    params = config.parameters
    return LatentODE(
        data_size=data_size,
        hidden_size=int(params.latent_hidden_size),
        latent_size=int(params.latent_size),
        width_size=int(params.latent_width),
        depth=int(params.latent_depth),
        latent_activation=str(params.latent_activation),
        latent_final_activation=str(params.latent_final_activation),
        drift_output_activation=str(params.latent_drift_output_activation),
        solver_rtol=float(params.rtol),
        solver_atol=float(params.atol),
        solver_max_steps=int(params.latent_solver_max_steps),
        key=key,
    )


def loss_fn(model, t_batch, y_batch, *, key, kl_weight: float, use_mean: bool):
    """Latent ODE loss = reconstruction + KL."""
    t_batch, y_batch = ensure_batched(t_batch, y_batch)
    batch_size = y_batch.shape[0]
    keys = jr.split(key, batch_size)

    def single_loss(ti, yi, ki):
        z_sample, mean, logstd = model.encode(ti, yi, key=ki)
        z0 = mean if use_mean else z_sample
        y_hat = model.decode(ti, z0)
        mpe = (
            jnp.linalg.norm(yi - y_hat, axis=-1)
            / jnp.maximum(
                jnp.linalg.norm(yi, axis=-1),
                1e-8,
            )
            * 100.0
        )
        mpe_mean = jnp.mean(mpe)
        mse = jnp.mean((yi - y_hat) ** 2)
        ref_power = jnp.mean(yi**2) + 1e-8
        nrmse = jnp.sqrt(mse / ref_power + 1e-8) * 100.0
        recon = mpe_mean + nrmse
        kl = kl_div(mean, logstd)
        total = recon + kl_weight * kl
        return total, (recon, mpe_mean, nrmse, kl)

    losses, comps = jax.vmap(single_loss)(t_batch, y_batch, keys)
    return (
        jnp.mean(losses),
        (
            jnp.mean(comps[0]),
            jnp.mean(comps[1]),
            jnp.mean(comps[2]),
            jnp.mean(comps[3]),
        ),
    )


@eqx.filter_jit
def train_step(
    model, opt_state, t_batch, y_batch, *, optimizer, key, kl_weight, use_mean
):
    """Single latent-ODE optimization step."""
    grad_fn = eqx.filter_value_and_grad(
        lambda m, tb, yb, k: loss_fn(
            m,
            tb,
            yb,
            key=k,
            kl_weight=kl_weight,
            use_mean=use_mean,
        )[0],
        has_aux=False,
    )
    loss, grads = grad_fn(model, t_batch, y_batch, key)
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def evaluate(model, data, *, key, kl_weight: float, use_mean: bool):
    """Evaluate latent ODE loss on a full segmented split."""
    t_eval, y_eval = ensure_batched(data["t"], data["y"])
    return loss_fn(
        model,
        t_eval,
        y_eval,
        key=key,
        kl_weight=kl_weight,
        use_mean=use_mean,
    )


def train_latent_model(config, model, dataset, *, seed: int):
    """Train a latent ODE using the repo's segmented-dataset curriculum style."""
    params = config.parameters
    train_orbits, val_orbits = split_data(dataset, config, shuffle=True)
    segment_lengths = [int(v) for v in params.segment_length_strategy]
    lr_strategy = [float(v) for v in params.lr_strategy]
    steps_strategy = [int(v) for v in params.steps_strategy]
    batch_size = int(params.batch_size)
    kl_weight = float(params.latent_kl_weight)
    use_mean = bool(params.latent_eval_use_mean)
    log_every = int(params.get("latent_log_every", 20))

    best_loss = np.inf
    best_model = model
    global_step = 0
    key = jr.PRNGKey(seed)

    total_steps = int(np.sum(steps_strategy))
    progress_bar = tqdm(total=total_steps, desc="Latent 2BP training", unit="step")

    for phase, (segment_length, lr_value, steps) in enumerate(
        zip(segment_lengths, lr_strategy, steps_strategy),
    ):
        train_segments = segment_data(train_orbits, segment_length)
        val_segments = segment_data(val_orbits, segment_length)
        if not has_valid_segment_batch(train_segments):
            raise ValueError(
                "No valid training segments produced for "
                f"segment_length={segment_length}. "
                "Reduce segment_length_strategy or check dataset masks.",
            )
        if not has_valid_segment_batch(val_segments):
            # Keep training moving even when val split or segmentation yields zero samples.
            val_segments = train_segments
        optimizer = optax.adam(lr_value)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        for step in range(int(steps)):
            key, loader_key, train_key = jr.split(key, 3)
            t_batch, y_batch = next(
                latent_dataloader(train_segments, batch_size, key=loader_key),
            )
            model, opt_state, _ = train_step(
                model,
                opt_state,
                t_batch,
                y_batch,
                optimizer=optimizer,
                key=train_key,
                kl_weight=kl_weight,
                use_mean=use_mean,
            )

            if step % log_every == 0 or step == int(steps) - 1:
                key, train_eval_key, val_eval_key = jr.split(key, 3)
                train_total, (train_recon, train_mpe, train_nrmse, train_kl) = evaluate(
                    model,
                    train_segments,
                    key=train_eval_key,
                    kl_weight=kl_weight,
                    use_mean=use_mean,
                )
                val_total, (val_recon, val_mpe, val_nrmse, val_kl) = evaluate(
                    model,
                    val_segments,
                    key=val_eval_key,
                    kl_weight=kl_weight,
                    use_mean=use_mean,
                )
                wandb.log(
                    {
                        "latent/train_loss": float(train_total),
                        "latent/train_recon": float(train_recon),
                        "latent/train_mpe": float(train_mpe),
                        "latent/train_nrmse": float(train_nrmse),
                        "latent/train_kl": float(train_kl),
                        "latent/val_loss": float(val_total),
                        "latent/val_recon": float(val_recon),
                        "latent/val_mpe": float(val_mpe),
                        "latent/val_nrmse": float(val_nrmse),
                        "latent/val_kl": float(val_kl),
                        "latent/phase": phase,
                        "latent/segment_length": int(segment_length),
                        "latent/log_every": int(log_every),
                        "global_step": global_step,
                    },
                    step=global_step,
                )
                print(
                    f"step {global_step:04d} | "
                    f"train loss {float(train_total):.4f} | "
                    f"train pe+nmse {float(train_recon):.4f} | "
                    f"train mpe {float(train_mpe):.4f} | "
                    f"train nrmse {float(train_nrmse):.4f} | "
                    f"train kl {float(train_kl):.4f} | "
                    f"val loss {float(val_total):.4f} | "
                    f"val pe+nmse {float(val_recon):.4f} | "
                    f"val mpe {float(val_mpe):.4f} | "
                    f"val nrmse {float(val_nrmse):.4f} | "
                    f"val kl {float(val_kl):.4f}",
                )
                if float(val_total) < best_loss:
                    best_loss = float(val_total)
                    best_model = clone_model(model)

            progress_bar.update(1)
            global_step += 1

    progress_bar.close()
    return best_model


def main():
    default_config = load_config(f"{neuralODE_path}/config/test_latent.yaml")
    sweep_config = load_config(f"{neuralODE_path}/config/test_latent.yaml")
    configs = make_all_configs(default_config, sweep_config)
    config = augment_latent_config(configs[0])

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

    if config.data.problem == "2BP":
        transform = Normalization2BP(
            l_char=constants.RADIUS_EARTH,
            mu=constants.MU_EARTH,
        )
        data_dict = transform.normalize_dataset(data_dict)

    dataset = format_data(data_dict)
    data_size = int(dataset["y"].shape[-1])
    config_payload = export_model_config(config, data_size=data_size)
    model = build_latent_model(
        config,
        data_size,
        key=jr.PRNGKey(int(config.parameters.seed)),
    )
    _ = get_dynamics_class(config.data.problem)

    model = train_latent_model(
        config,
        model,
        dataset,
        seed=int(config.parameters.seed),
    )

    artifact_dir = Path("files/models")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / f"latent_2bp_compare_{run.id}.eqx"
    save_latent_checkpoint(config_payload, model, checkpoint_path)
    save_latent_model(config_payload, model, run.id, run=run)
    wandb.save(str(checkpoint_path))
    wandb.finish()


if __name__ == "__main__":
    main()
```