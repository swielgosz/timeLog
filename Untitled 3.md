``` python
"""
Latent ODE demo for 2BP data (inspired by https://docs.kidger.site/diffrax/examples/latent_ode/).

Loads `complex_TBP_planar_10_train` from WandB, normalizes it, trains a simple
latent ODE, logs metrics/plot to WandB, and saves a true vs. reconstructed orbit.
"""

import os
from typing import Iterator, Tuple

# Keep things CPU-only and avoid oversubscribing threads
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS",
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax
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
        return self.scale * jnp.tanh(self.mlp(y))


class LatentODE(eqx.Module):
    encoder: eqx.nn.MLP
    func: LatentFunc
    decoder: eqx.nn.MLP
    latent_dim: int

    def encode(self, y0):
        stats = self.encoder(y0)
        mean, logstd = jnp.split(stats, 2, axis=-1)
        logstd = jnp.clip(logstd, -5.0, 2.0)
        return mean, logstd

    def decode(self, z):
        return self.decoder(z)


def kl_div(mean, logstd):
    var = jnp.exp(2.0 * logstd)
    return 0.5 * jnp.sum(var + mean**2 - 1.0 - 2.0 * logstd)


def solve_trajectory(model: LatentODE, t_obs, y_obs, key):
    mean, logstd = model.encode(y_obs[0])
    eps = jr.normal(key, mean.shape)
    z0 = mean + jnp.exp(logstd) * eps

    term = dfx.ODETerm(model.func)
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(ts=t_obs)
    # Use observed step as initial dt0; ensure positive
    dt0 = jnp.maximum(jnp.asarray(t_obs[1] - t_obs[0]), jnp.asarray(1e-6))
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t_obs[0],
        t1=t_obs[-1],
        dt0=dt0,
        y0=z0,
        saveat=saveat,
        max_steps=4096,
    )
    z_traj = sol.ys
    y_hat = jax.vmap(model.decode)(z_traj)
    return y_hat, mean, logstd


def loss_fn(model: LatentODE, t_batch, y_batch, key, kl_weight=1e-3):
    batch_size = y_batch.shape[0]
    keys = jr.split(key, batch_size)

    def pe_plus_nmse(y_true, y_pred, eps=1e-8):
        norm_true = jnp.linalg.norm(y_true, axis=-1)
        denom = jnp.maximum(norm_true, eps)
        mpe = jnp.linalg.norm(y_true - y_pred, axis=-1) / denom * 100.0
        mpe_mean = jnp.mean(mpe)
        mse = jnp.mean((y_true - y_pred) ** 2)
        ref_power = jnp.mean(y_true**2) + eps
        normalized_rmse = jnp.sqrt(mse / ref_power + eps) * 100.0
        return mpe_mean + normalized_rmse, mpe_mean, normalized_rmse

    def single_loss(ti, yi, ki):
        y_hat, mean, logstd = solve_trajectory(model, ti, yi, ki)
        pe_rmse, mpe_mean, nrmse = pe_plus_nmse(yi, y_hat)
        kl = kl_div(mean, logstd)
        return pe_rmse + kl_weight * kl, (pe_rmse, mpe_mean, nrmse, kl)

    losses, comps = jax.vmap(single_loss)(t_batch, y_batch, keys)
    total = jnp.mean(losses)
    pe_rmse = jnp.mean(comps[0])
    mpe = jnp.mean(comps[1])
    nrmse = jnp.mean(comps[2])
    kl = jnp.mean(comps[3])
    return total, (pe_rmse, mpe, nrmse, kl)


def accel_percent_error(dynamics, y_true_seq, y_hat_seq, eps=1e-8):
    """
    Compute mean percent error of accelerations from a dynamics model.
    y_* shape: (T, 6)
    """
    acc_true = jax.vmap(lambda s: dynamics(0.0, s)[3:])(y_true_seq)
    acc_pred = jax.vmap(lambda s: dynamics(0.0, s)[3:])(y_hat_seq)
    denom = jnp.maximum(jnp.linalg.norm(acc_true, axis=-1), eps)
    err = jnp.linalg.norm(acc_pred - acc_true, axis=-1) / denom
    return jnp.mean(err) * 100.0


def batch_accel_percent_error(model, dynamics, t_batch, y_batch, key, eps=1e-8):
    """
    Compute mean accel percent error over a batch by integrating each example.
    Shapes: t_batch (B, T), y_batch (B, T, 6)
    """
    batch_size = y_batch.shape[0]
    keys = jr.split(key, batch_size)

    def single_accel(ti, yi, ki):
        y_hat, _, _ = solve_trajectory(model, ti, yi, ki)
        return accel_percent_error(dynamics, yi, y_hat, eps=eps)

    return jnp.mean(jax.vmap(single_accel)(t_batch, y_batch, keys))


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


def build_model(state_dim: int, latent_dim: int, width: int, key):
    k1, k2, k3 = jr.split(key, 3)
    encoder = eqx.nn.MLP(
        in_size=state_dim,
        out_size=2 * latent_dim,
        width_size=width,
        depth=2,
        # activation=jax.nn.tanh,
        key=k1,
    )
    func = LatentFunc(
        scale=jnp.array(0.1),
        mlp=eqx.nn.MLP(
            in_size=latent_dim,
            out_size=latent_dim,
            width_size=width,
            depth=2,
            activation=jax.nn.tanh,
            key=k2,
        ),
    )
    decoder = eqx.nn.MLP(
        in_size=latent_dim,
        out_size=state_dim,
        width_size=width,
        depth=2,
        activation=jax.nn.tanh,
        key=k3,
    )
    return LatentODE(encoder=encoder, func=func, decoder=decoder, latent_dim=latent_dim)


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
    # Use timesteps valid for all selected orbits
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


def main():
    cfg = {
        "dataset": "complex_TBP_planar_10_train",
        "latent_dim": 4,
        "width": 64,
        "lr": 1e-3,
        "num_steps": 1000,
        "batch_size": 64,
        "kl_weight": 1e-3,
        "num_trajs": 10,
        "seed": 0,
    }
    wandb_run = wandb.init(
        project="neuralODEs",
        entity="mlds-lab",
        group="latent-ode",
        config=cfg,
    )

    key = jr.PRNGKey(cfg["seed"])
    key, data_key, model_key, loader_key = jr.split(key, 4)

    t_shared, ys, transform = load_2bp_dataset(cfg["num_trajs"], key=data_key)
    batch_size = min(cfg["batch_size"], ys.shape[0])
    state_dim = ys.shape[-1]

    model = build_model(state_dim, cfg["latent_dim"], cfg["width"], model_key)
    optimizer = optax.adam(cfg["lr"])
    params = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    num_steps = cfg["num_steps"]
    kl_weight = cfg["kl_weight"]
    dynamics = get_dynamics_class("2BP")

    for step in range(num_steps):
        key, loader_key = jr.split(key)
        for t_batch, y_batch in dataloader((t_shared, ys), batch_size, loader_key):
            key, subkey = jr.split(key)
            model, opt_state, loss = train_step(
                model,
                opt_state,
                t_batch,
                y_batch,
                subkey,
                optimizer,
                kl_weight,
            )
            break  # one minibatch per step

        if step % 20 == 0 or step == num_steps - 1:
            key, eval_key = jr.split(key)
            total, (pe_rmse, mpe, nrmse, kl) = loss_fn(
                model,
                t_batch,
                y_batch,
                eval_key,
                kl_weight,
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

    # Plot true vs reconstructed for first orbit
    key, plot_key = jr.split(key)
    y_hat, _, _ = solve_trajectory(model, t_shared, ys[0], plot_key)
    t_np = np.asarray(t_shared)
    y_true_np = np.asarray(ys[0])
    y_hat_np = np.asarray(y_hat)
    dyn = get_dynamics_class("2BP")
    acc_err = float(accel_percent_error(dyn, y_true_np, y_hat_np))

    # Denormalize for plotting
    t_phys = t_np * transform.t_char
    y_true_phys = y_true_np.copy()
    y_hat_phys = y_hat_np.copy()
    y_true_phys[:, 0:3] *= transform.l_char
    y_true_phys[:, 3:6] *= transform.v_char
    y_hat_phys[:, 0:3] *= transform.l_char
    y_hat_phys[:, 3:6] *= transform.v_char

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(y_true_phys[:, 0], y_true_phys[:, 1], label="true XY")
    ax.plot(y_hat_phys[:, 0], y_hat_phys[:, 1], "--", label="recon XY")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Latent ODE reconstruction (XY projection, first orbit)")
    ax.legend()
    fig.tight_layout()
    wandb.log({"reconstruction": wandb.Image(fig), "metrics/accel_percent_error": acc_err})
    plt.close(fig)

    wandb_run.finish()


if __name__ == "__main__":
    main()
```