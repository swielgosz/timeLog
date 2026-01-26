# Latent training scripts
These scripts learn latent dynamics, *not* perturbations
## `train_latent.py`
``` python
"""
Latent ODE demo for 2BP data, modeled after
https://docs.kidger.site/diffrax/examples/latent_ode/

- Encoder: GRU over (t, y) reversed -> hidden_to_latent (mean/logstd)
- Latent dynamics: scale * tanh(MLP(hidden))
- Decoder: hidden_to_data linear

Loads `complex_TBP_planar_10_train`, normalizes (2BP), trains, logs metrics +
XY plot + accel error to WandB.
"""

import os
from typing import Iterator, Tuple

# Keep CPU-only and limit threads
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import json
import tempfile
from pathlib import Path

import diffrax
import diffrax as dfx
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
from mldsml.config_utils import DotDict
from mldsml.wandb_utils import load_dataset

import wandb
from neuralODE import constants
from neuralODE.data import format_data, sample_data
from neuralODE.dynamics import get_dynamics_class
from neuralODE.normalization import Normalization2BP


class LatentFunc(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __call__(self, t, y, args):
        return self.scale * jnn.tanh(self.mlp(y))


class LatentODE(eqx.Module):
    func: LatentFunc
    rnn_cell: eqx.nn.GRUCell
    hidden_to_latent: eqx.nn.Linear  # hidden -> mean+logstd
    latent_to_hidden: eqx.nn.MLP  # z -> hidden init for ODE
    hidden_to_data: eqx.nn.Linear  # hidden -> data

    hidden_size: int
    latent_size: int

    def __init__(
        self,
        *,
        data_size,
        hidden_size,
        latent_size,
        width_size,
        depth,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        mkey, gkey, hlkey, lhkey, hdkey = jr.split(key, 5)
        scale = jnp.ones(())
        mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=mkey,
        )
        self.func = LatentFunc(scale, mlp)
        self.rnn_cell = eqx.nn.GRUCell(data_size + 1, hidden_size, key=gkey)
        self.hidden_to_latent = eqx.nn.Linear(
            hidden_size,
            2 * latent_size,
            key=hlkey,
        )
        self.latent_to_hidden = eqx.nn.MLP(
            latent_size,
            hidden_size,
            width_size=width_size,
            depth=depth,
            key=lhkey,
        )
        self.hidden_to_data = eqx.nn.Linear(hidden_size, data_size, key=hdkey)
        self.hidden_size = hidden_size
        self.latent_size = latent_size

    def encode(self, ts, ys, *, key):
        # ts: (T,), ys: (T, D)
        data = jnp.concatenate([ts[:, None], ys], axis=1)
        h = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            h = self.rnn_cell(data_i, h)
        stats = self.hidden_to_latent(h)
        mean, logstd = stats[: self.latent_size], stats[self.latent_size :]
        logstd = jnp.clip(logstd, -5.0, 2.0)
        std = jnp.exp(logstd)
        z = mean + jr.normal(key, (self.latent_size,)) * std
        return z, mean, logstd

    def decode(self, ts, z0):
        h0 = self.latent_to_hidden(z0)
        term = dfx.ODETerm(self.func)
        solver = dfx.Tsit5()
        dt0 = jnp.maximum(jnp.asarray(ts[1] - ts[0]), jnp.asarray(1e-6))
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=h0,
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=4096,
        )
        ys_hat = jax.vmap(self.hidden_to_data)(sol.ys)
        return ys_hat


def kl_div(mean, logstd):
    var = jnp.exp(2.0 * logstd)
    return 0.5 * jnp.sum(var + mean**2 - 1.0 - 2.0 * logstd)


def pe_plus_nmse(y_true, y_pred, eps=1e-8):
    norm_true = jnp.linalg.norm(y_true, axis=-1)
    denom = jnp.maximum(norm_true, eps)
    mpe = jnp.linalg.norm(y_true - y_pred, axis=-1) / denom * 100.0
    mpe_mean = jnp.mean(mpe)
    mse = jnp.mean((y_true - y_pred) ** 2)
    ref_power = jnp.mean(y_true**2) + eps
    nrmse = jnp.sqrt(mse / ref_power + eps) * 100.0
    return mpe_mean + nrmse, mpe_mean, nrmse


def loss_fn(model: LatentODE, t_batch, y_batch, key, kl_weight=1e-3):
    B = y_batch.shape[0]
    keys = jr.split(key, B)

    def single_loss(ti, yi, ki):
        z0, mean, logstd = model.encode(ti, yi, key=ki)
        y_hat = model.decode(ti, z0)
        pe_rmse, mpe, nrmse = pe_plus_nmse(yi, y_hat)
        kl = kl_div(mean, logstd)
        return pe_rmse + kl_weight * kl, (pe_rmse, mpe, nrmse, kl)

    losses, comps = jax.vmap(single_loss)(t_batch, y_batch, keys)
    total = jnp.mean(losses)
    pe_rmse = jnp.mean(comps[0])
    mpe = jnp.mean(comps[1])
    nrmse = jnp.mean(comps[2])
    kl = jnp.mean(comps[3])
    return total, (pe_rmse, mpe, nrmse, kl)


@eqx.filter_jit
def train_step(model, opt_state, t_batch, y_batch, key, optimizer, kl_weight):
    grad_fn = eqx.filter_value_and_grad(
        lambda m, tb, yb, k: loss_fn(m, tb, yb, k, kl_weight)[0],
        has_aux=False,
    )
    loss, grads = grad_fn(model, t_batch, y_batch, key)
    params = eqx.filter(model, eqx.is_inexact_array)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def accel_percent_error(dynamics, y_true_seq, y_hat_seq, eps=1e-8):
    acc_true = jax.vmap(lambda s: dynamics(0.0, s)[3:])(y_true_seq)
    acc_pred = jax.vmap(lambda s: dynamics(0.0, s)[3:])(y_hat_seq)
    denom = jnp.maximum(jnp.linalg.norm(acc_true, axis=-1), eps)
    err = jnp.linalg.norm(acc_pred - acc_true, axis=-1) / denom
    return jnp.mean(err) * 100.0


def batch_accel_percent_error(model, dynamics, t_batch, y_batch, key, eps=1e-8):
    B = y_batch.shape[0]
    keys = jr.split(key, B)

    def single_accel(ti, yi, ki):
        z0, _, _ = model.encode(ti, yi, key=ki)
        y_hat = model.decode(ti, z0)
        return accel_percent_error(dynamics, yi, y_hat, eps=eps)

    return jnp.mean(jax.vmap(single_accel)(t_batch, y_batch, keys))


def build_model(data_size, hidden_size, latent_size, width_size, depth, key):
    return LatentODE(
        data_size=data_size,
        hidden_size=hidden_size,
        latent_size=latent_size,
        width_size=width_size,
        depth=depth,
        key=key,
    )


def load_2bp_dataset(num_trajs: int, key=jr.PRNGKey(0)):
    data_dict = load_dataset(
        "complex_TBP_planar_10_train",
        version="latest",
        project="neuralODEs",
        entity="mlds-lab",
    )
    data_dict = sample_data(data_dict, num_trajs)
    transform = Normalization2BP(
        l_char=constants.RADIUS_EARTH,
        mu=constants.MU_EARTH,
    )
    data_norm = transform.normalize_dataset(data_dict)
    data_jnp = format_data(data_norm)
    ts = data_jnp["t"]
    ys = data_jnp["y"]
    mask = data_jnp["mask"].astype(bool)
    valid_mask = jnp.all(mask, axis=0)
    ts = ts[:, valid_mask]
    ys = ys[:, valid_mask, :]
    t_shared = ts[0]
    return t_shared, ys, transform


def dataloader(data: Tuple[jnp.ndarray, jnp.ndarray], batch_size: int, key) -> Iterator:
    t_shared, ys = data
    n = ys.shape[0]
    idx = jr.permutation(key, n)
    for start in range(0, n, batch_size):
        batch_idx = idx[start : start + batch_size]
        yield (
            jnp.repeat(t_shared[None, :], len(batch_idx), axis=0),
            ys[batch_idx],
        )


def load_latent_model_wandb(
    filename,
    project="neuralODEs",
    entity="mlds-lab",
    version="latest",
) -> tuple[DotDict, LatentODE, wandb.apis.public.Run]:
    """
    Load a LatentODE model artifact saved by train_latent.py.

    Accepts a run ID (with or without the `latent-ode-` prefix) or an
    artifact spec like `latent-ode-<run_id>:v0`.
    """
    api = wandb.Api()

    model_name = filename.split("/")[-1].split(".")[0]

    if ":" in model_name:
        version = model_name.split(":")[-1]
        model_name = model_name.split(":")[0]

    # Retrieve the artifact
    artifact_name = f"{entity}/{project}/{model_name}:{version}"
    artifact = api.artifact(artifact_name, type="model")
    artifact_dir = artifact.download()

    # Get the run associated with the artifact
    artifact_run = artifact.logged_by()

    # Load the model from the artifact
    model_file_path = f"{artifact_dir}/{model_name}.eqx"
    with open(model_file_path, "rb") as f:
        config = json.loads(f.readline().decode())
        config = DotDict(config)  # Convert to DotDict for easier access

        # Copy parameters to top level
        # HACK
        for key, value in config.parameters.items():
            config[key] = value

        data_shape = 6
        model = LatentODE(
            data_size=data_shape,
            hidden_size=config.hidden_size,
            latent_size=config.latent_size,
            width_size=config.width,
            depth=config.depth,
            config=config,
            key=jr.PRNGKey(0),
        )
        model = eqx.tree_deserialise_leaves(f, model)

    return config, model, artifact_run


def save_latent_model(
    config: dict,
    model: LatentODE,
    run_id: str,
    project: str = "neuralODEs",
    entity: str = "mlds-lab",
    metadata: dict | None = None,
):
    """
    Serialize and log a latent ODE model as a wandb artifact.
    """
    meta = {"run_id": run_id}
    if metadata:
        meta.update(metadata)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        model_path = tmpdir_path / "model.eqx"
        with open(model_path, "wb") as f:
            f.write(json.dumps(config).encode() + b"\n")
            eqx.tree_serialise_leaves(f, model)
        artifact = wandb.Artifact(
            name=f"latent-ode-{run_id}",
            type="latent-ode-model",
            metadata=meta,
        )
        artifact.add_file(str(model_path), name="model.eqx")
        wandb.log_artifact(artifact, project=project, entity=entity)


def main():
    # config = {
    #     "dataset": "complex_TBP_planar_10_train",
    #     "latent_size": 16,
    #     "hidden_size": 16,
    #     "width_size": 64,
    #     "depth": 2,
    #     "lr": 1e-3,
    #     "num_steps": 5000,
    #     "batch_size": 64,
    #     "kl_weight": 1e-3,
    #     "num_trajs": 10,
    #     "seed": 0,
    # }
    config = {
        "dataset": "complex_TBP_planar_10_train",
        "latent_size": 4,
        "hidden_size": 4,
        "width_size": 4,
        "depth": 2,
        "lr": 1e-3,
        "num_steps": 500,
        "batch_size": 64,
        "kl_weight": 1e-3,
        "num_trajs": 10,
        "seed": 0,
    }
    wandb_run = wandb.init(
        project="neuralODEs",
        entity="mlds-lab",
        group="latent-ode",
        config=config,
    )

    key = jr.PRNGKey(config["seed"])
    key, data_key, model_key = jr.split(key, 3)

    t_shared, ys, transform = load_2bp_dataset(config["num_trajs"], key=data_key)
    batch_size = min(config["batch_size"], ys.shape[0])
    data_size = ys.shape[-1]

    model = build_model(
        data_size=data_size,
        hidden_size=config["hidden_size"],
        latent_size=config["latent_size"],
        width_size=config["width_size"],
        depth=config["depth"],
        key=model_key,
    )
    optimizer = optax.adam(config["lr"])
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    dynamics = get_dynamics_class("2BP")

    for step in range(config["num_steps"]):
        key, loader_key, subkey = jr.split(key, 3)
        for t_batch, y_batch in dataloader((t_shared, ys), batch_size, loader_key):
            model, opt_state, loss = train_step(
                model,
                opt_state,
                t_batch,
                y_batch,
                subkey,
                optimizer,
                config["kl_weight"],
            )
            break  # one minibatch per step

        if step % 20 == 0 or step == config["num_steps"] - 1:
            key, eval_key = jr.split(key)
            total, (pe_rmse, mpe, nrmse, kl) = loss_fn(
                model,
                t_batch,
                y_batch,
                eval_key,
                config["kl_weight"],
            )
            acc_err_step = float(
                batch_accel_percent_error(model, dynamics, t_batch, y_batch, eval_key),
            )
            wandb.log(
                {
                    "train/loss": float(total),
                    "train/pe_plus_nmse": float(pe_rmse),
                    "train/mpe": float(mpe),
                    "train/nrmse": float(nrmse),
                    "train/kl": float(kl),
                    "train/accel_percent_error": acc_err_step,
                    "step": step,
                },
            )
            print(
                f"step {step:04d} | loss {float(total):.4f} | pe+nmse {float(pe_rmse):.4f} | mpe {float(mpe):.4f} | nrmse {float(nrmse):.4f} | kl {float(kl):.4f} | accel_err {acc_err_step:.2f}%",
            )

    # Final reconstruction on first orbit (denormalized, XY plot)
    key, plot_key = jr.split(key)
    z0, _, _ = model.encode(t_shared, ys[0], key=plot_key)
    y_hat = model.decode(t_shared, z0)
    t_np = np.asarray(t_shared) * transform.t_char
    y_true_np = np.asarray(ys[0])
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
    ax.plot(y_hat_phys[:, 0], y_hat_phys[:, 1], "--", label="recon XY")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Latent ODE reconstruction (XY, first orbit)")
    ax.legend()
    fig.tight_layout()
    wandb.log(
        {
            "reconstruction": wandb.Image(fig),
            "metrics/accel_percent_error": acc_err_final,
            "step": config["num_steps"],
        },
    )
    plt.close(fig)

    # Save model as a wandb artifact (eqx serialization with JSON header)
    config["data_size"] = data_size
    save_latent_model(
        config=config,
        model=model,
        run_id=wandb_run.id,
        project="neuralODEs",
        entity="mlds-lab",
    )

    wandb_run.finish()


if __name__ == "__main__":
    main()
```

 # 
# `train_latent_UDE_forced_oscillator`