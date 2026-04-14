``` python
"""
Train a 2BP GRU-encoder Neural ODE with Jacobian linearity regularization.

Functionality:
- Encoder: GRU over reversed (t, y) -> latent posterior (mean/logstd)
- Decoder: nonlinear hidden-state Neural ODE (diffrax) -> trajectory reconstruction
- Regularizer: mean ||J_f(h_t) - A||_F^2 over sampled hidden states

Example usage:
    python scripts/training/train_latent_2BP_gru_jacobian_reg.py
"""

import os
import json
from pathlib import Path

# Keep CPU-only and limit threads to avoid GPU/cublas init failures.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from tqdm import tqdm

from neuralODE.data import segment_data
from neuralODE.dynamics import get_dynamics_class
from neuralODE.latentODE import JacobianRegularizedGRULatentODE

try:
    # Package import path (repo root on PYTHONPATH).
    from scripts.training.train_latent_2BP import (
        accel_percent_error,
        batch_accel_percent_error,
        dataloader,
        kl_div,
        load_2bp_dataset,
        normalize_length_strategy,
        pe_plus_nmse,
        save_latent_model,
    )
except ModuleNotFoundError:
    # Local fallback for direct absolute-path script execution.
    from train_latent_2BP import (
        accel_percent_error,
        batch_accel_percent_error,
        dataloader,
        kl_div,
        load_2bp_dataset,
        normalize_length_strategy,
        pe_plus_nmse,
        save_latent_model,
    )


def loss_fn(
    model: JacobianRegularizedGRULatentODE,
    t_batch,
    y_batch,
    key,
    *,
    kl_weight=1e-3,
    jacobian_weight=1e-3,
    jacobian_samples=8,
):
    """
    Compute batch loss = reconstruction + KL + Jacobian linearity regularizer.
    """
    batch_size = y_batch.shape[0]
    keys = jr.split(key, batch_size)

    def single_loss(ti, yi, ki):
        z0, mean, logstd = model.encode(ti, yi, key=ki)
        y_hat = model.decode(ti, z0)
        pe_rmse, mpe, nrmse = pe_plus_nmse(yi, y_hat)
        kl = kl_div(mean, logstd)
        jac_pen = model.jacobian_linearity_penalty(
            ti,
            z0,
            num_samples=jacobian_samples,
        )
        total = pe_rmse + kl_weight * kl + jacobian_weight * jac_pen
        return total, (pe_rmse, mpe, nrmse, kl, jac_pen)

    losses, comps = jax.vmap(single_loss)(t_batch, y_batch, keys)
    total = jnp.mean(losses)
    pe_rmse = jnp.mean(comps[0])
    mpe = jnp.mean(comps[1])
    nrmse = jnp.mean(comps[2])
    kl = jnp.mean(comps[3])
    jac_pen = jnp.mean(comps[4])
    return total, (pe_rmse, mpe, nrmse, kl, jac_pen)


@eqx.filter_jit
def train_step(
    model,
    opt_state,
    t_batch,
    y_batch,
    key,
    optimizer,
    *,
    kl_weight,
    jacobian_weight,
    jacobian_samples,
):
    grad_fn = eqx.filter_value_and_grad(
        lambda m, tb, yb, k: loss_fn(
            m,
            tb,
            yb,
            k,
            kl_weight=kl_weight,
            jacobian_weight=jacobian_weight,
            jacobian_samples=jacobian_samples,
        )[0],
        has_aux=False,
    )
    loss, grads = grad_fn(model, t_batch, y_batch, key)
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def build_model(
    data_size,
    hidden_size,
    latent_size,
    width_size,
    depth,
    key,
    *,
    target_init_scale=1e-2,
    latent_activation="softplus",
    latent_final_activation="tanh",
    drift_output_activation="tanh",
):
    """
    Build GRU-based Neural ODE with Jacobian regularization target matrix.
    """
    return JacobianRegularizedGRULatentODE(
        data_size=data_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        width_size=width_size,
        depth=depth,
        target_init_scale=target_init_scale,
        latent_activation=latent_activation,
        latent_final_activation=latent_final_activation,
        drift_output_activation=drift_output_activation,
        key=key,
    )


def save_gru_jacobian_checkpoint(
    config: dict,
    model: JacobianRegularizedGRULatentODE,
    path: Path,
) -> None:
    """
    Save GRU Jacobian-regularized model checkpoint with JSON header.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(json.dumps(config).encode() + b"\n")
        eqx.tree_serialise_leaves(f, model)


def load_gru_jacobian_checkpoint(
    path: Path,
) -> tuple[dict, JacobianRegularizedGRULatentODE]:
    """
    Load GRU Jacobian-regularized model checkpoint.
    """
    with path.open("rb") as f:
        config = json.loads(f.readline().decode())
        data_size = int(config.get("data_size", 6))
        hidden_size = int(config.get("hidden_size", 32))
        latent_size = int(config.get("latent_size", 4))
        width_size = int(config.get("width_size", config.get("width", 32)))
        depth = int(config.get("depth", 2))
        model = JacobianRegularizedGRULatentODE(
            data_size=data_size,
            hidden_size=hidden_size,
            latent_size=latent_size,
            width_size=width_size,
            depth=depth,
            target_init_scale=float(config.get("target_init_scale", 1e-2)),
            latent_activation=str(config.get("latent_activation", "softplus")),
            latent_final_activation=str(config.get("latent_final_activation", "tanh")),
            drift_output_activation=str(config.get("drift_output_activation", "tanh")),
            key=jr.PRNGKey(0),
        )
        model = eqx.tree_deserialise_leaves(f, model)
    return config, model


def main():
    """
    Train and log GRU Jacobian-regularized Neural ODE on normalized 2BP data.
    """
    config = {
        "dataset": "complex_TBP_planar_10_train",
        "segment_length_strategy": [16, 64, 360],
        "length_strategy": [[0.0, 1.0]] * 3,
        "steps_strategy": [200, 200, 500],
        "lr_strategy": [1e-3, 1e-3, 1e-3],
        # "lr_strategy": [1e-3, 1e-3, 1e-3],
        "save_best": True,
        "latent_size": 4,
        "hidden_size": 32,
        "width_size": 32,
        "depth": 2,
        "latent_activation": "tanh",
        "latent_final_activation": "tanh",
        "drift_output_activation": "tanh",
        "target_init_scale": 1e-2,
        "jacobian_weight": 1e-3,
        "jacobian_samples": 8,
        "lr": 1e-3,
        "num_steps": 1500,
        "batch_size": 32,
        "kl_weight": 1e-4,
        "kl_weight_strategy": [1e-5, 1e-4, 1e-3],
        "num_trajs": 10,
        "seed": 0,
        "encoder_type": "gru",
        "linearity_regularization": "jacobian_target_matrix",
    }
    wandb_run = wandb.init(
        project="neuralODEs",
        entity="mlds-lab",
        group="latent-ode-gru-jacobian-reg-2bp",
        config=config,
    )

    key = jr.PRNGKey(config["seed"])
    key, data_key, model_key = jr.split(key, 3)

    t_shared, ys, transform = load_2bp_dataset(
        config["num_trajs"],
        config["dataset"],
        key=data_key,
    )
    batch_size = min(config["batch_size"], ys.shape[0])
    data_size = ys.shape[-1]
    full_dataset = {
        "t": jnp.repeat(t_shared[None, :], ys.shape[0], axis=0),
        "y": ys,
        "mask": jnp.ones((ys.shape[0], ys.shape[1]), dtype=bool),
    }
    total_length = int(ys.shape[1])
    length_strategy, segment_strategy, steps_strategy, lr_strategy, kl_strategy = (
        normalize_length_strategy(config, total_length=total_length)
    )
    segment_ratio_schedule = [
        100.0 * float(length) / float(total_length) for length in segment_strategy
    ]
    segment_cache = {}
    best_loss = float("inf")
    best_path = Path("files/models") / f"best_latent_2bp_gru_jac_{wandb_run.id}.eqx"

    model = build_model(
        data_size=data_size,
        hidden_size=config["hidden_size"],
        latent_size=config["latent_size"],
        width_size=config["width_size"],
        depth=config["depth"],
        key=model_key,
        target_init_scale=config["target_init_scale"],
        latent_activation=config["latent_activation"],
        latent_final_activation=config["latent_final_activation"],
        drift_output_activation=config["drift_output_activation"],
    )
    dynamics = get_dynamics_class("2BP")

    total_steps = int(sum(steps_strategy))
    progress_bar = tqdm(
        total=total_steps,
        desc="Latent 2BP GRU Jacobian-reg training",
        unit="step",
    )
    global_step = 0

    for phase, (length_fracs, segment_length, steps, lr_value, kl_weight) in enumerate(
        zip(length_strategy, segment_strategy, steps_strategy, lr_strategy, kl_strategy),
    ):
        optimizer = optax.adam(lr_value)
        params = eqx.filter(model, eqx.is_inexact_array)
        opt_state = optimizer.init(params)
        if segment_length not in segment_cache:
            segment_cache[segment_length] = segment_data(full_dataset, segment_length)
        train_stage = segment_cache[segment_length]

        ts = train_stage["t"]
        ys_stage = train_stage["y"]

        length_size = int(ys_stage.shape[1])
        start_idx = int(np.floor(length_size * length_fracs[0]))
        end_idx = int(np.ceil(length_size * length_fracs[1]))
        start_idx = max(0, start_idx)
        end_idx = min(length_size, end_idx)
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, length_size)

        _ts = ts[:, start_idx:end_idx]
        _ys = ys_stage[:, start_idx:end_idx, :]

        for step in range(int(steps)):
            key, loader_key, subkey = jr.split(key, 3)
            for t_batch, y_batch in dataloader((_ts, _ys), batch_size, loader_key):
                model, opt_state, _loss = train_step(
                    model,
                    opt_state,
                    t_batch,
                    y_batch,
                    subkey,
                    optimizer,
                    kl_weight=kl_weight,
                    jacobian_weight=config["jacobian_weight"],
                    jacobian_samples=config["jacobian_samples"],
                )
                break

            if step % 20 == 0 or step == int(steps) - 1:
                key, eval_key = jr.split(key)
                total, (pe_rmse, mpe, nrmse, kl, jac_pen) = loss_fn(
                    model,
                    t_batch,
                    y_batch,
                    eval_key,
                    kl_weight=kl_weight,
                    jacobian_weight=config["jacobian_weight"],
                    jacobian_samples=config["jacobian_samples"],
                )
                acc_err_step = float(
                    batch_accel_percent_error(
                        model,
                        dynamics,
                        t_batch,
                        y_batch,
                        eval_key,
                    ),
                )
                wandb.log(
                    {
                        "train/loss": float(total),
                        "train/pe_plus_nmse": float(pe_rmse),
                        "train/mpe": float(mpe),
                        "train/nrmse": float(nrmse),
                        "train/kl": float(kl),
                        "train/jacobian_penalty": float(jac_pen),
                        "train/accel_percent_error": acc_err_step,
                        "curriculum/phase": phase,
                        "curriculum/segment_length": int(segment_length),
                        "curriculum/segment_ratio": float(
                            segment_ratio_schedule[phase],
                        ),
                        "curriculum/kl_weight": kl_weight,
                        "global_step": global_step,
                    },
                    step=global_step,
                )
                if config["save_best"] and float(total) < best_loss:
                    best_loss = float(total)
                    save_gru_jacobian_checkpoint(config, model, best_path)
                print(
                    f"step {global_step:04d} | loss {float(total):.4f} | pe+nmse {float(pe_rmse):.4f} | mpe {float(mpe):.4f} | nrmse {float(nrmse):.4f} | kl {float(kl):.4f} | jac {float(jac_pen):.4f} | accel_err {acc_err_step:.2f}%",
                )

            progress_bar.update(1)
            global_step += 1

    progress_bar.close()

    output_dir = Path("files/figures/latent_2bp_gru_jacobian_reg")
    output_dir.mkdir(parents=True, exist_ok=True)
    for orbit_idx in [0, 1]:
        key, plot_key = jr.split(key)
        z0, _, _ = model.encode(t_shared, ys[orbit_idx], key=plot_key)
        y_hat = model.decode(t_shared, z0)
        y_true_np = np.asarray(ys[orbit_idx])
        y_hat_np = np.asarray(y_hat)
        y_true_phys = y_true_np.copy()
        y_hat_phys = y_hat_np.copy()
        y_true_phys[:, 0:3] *= transform.l_char
        y_true_phys[:, 3:6] *= transform.v_char
        y_hat_phys[:, 0:3] *= transform.l_char
        y_hat_phys[:, 3:6] *= transform.v_char
        acc_err_final = float(accel_percent_error(dynamics, y_true_np, y_hat_np))

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(y_true_phys[:, 0], y_true_phys[:, 1], label="true XY")
        ax.plot(y_hat_phys[:, 0], y_hat_phys[:, 1], "--", label="gru-jac recon XY")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(
            f"GRU Jacobian-reg latent ODE reconstruction (XY, orbit {orbit_idx})",
        )
        ax.legend()
        fig.tight_layout()
        local_plot_path = (
            output_dir
            / f"latent_gru_jacobian_reconstruction_{wandb_run.id}_{orbit_idx}.png"
        )
        fig.savefig(local_plot_path, dpi=300)
        wandb.log(
            {
                f"reconstruction/orbit_{orbit_idx}": wandb.Image(fig),
                f"metrics/accel_percent_error_orbit_{orbit_idx}": acc_err_final,
                "step": config["num_steps"],
            },
        )
        plt.close(fig)

    config["data_size"] = data_size
    model_to_save = model
    if config.get("save_best", False) and best_path.exists():
        try:
            _, model_to_save = load_gru_jacobian_checkpoint(best_path)
            print(f"Loaded best checkpoint for artifact logging: {best_path}")
        except Exception as exc:
            print(
                "Warning: failed to load best checkpoint; "
                f"falling back to final model. Error: {exc}",
            )
    save_latent_model(
        config=config,
        model=model_to_save,
        run_id=wandb_run.id,
        run=wandb_run,
        metadata={
            "encoder_type": "gru",
            "linearity_regularization": "jacobian_target_matrix",
        },
    )
    wandb_run.finish()


if __name__ == "__main__":
    main()
```