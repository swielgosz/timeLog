``` python
"""
Train a monolithic UDE for planar J2 perturbations on top of known 2BP dynamics.

Functionality:
- Loads a dataset artifact from Weights & Biases
- Normalizes 2BP states (optional)
- Splits orbits 80/20 into train/test
- Trains a residual UDE in one script (no external training helpers)
- Logs train/test loss, MPE, and NMRSE to W&B
- Saves model checkpoint and diagnostic plots to W&B

Example usage:
    python scripts/training/train_UDE_j2_single_script.py
"""

import json
import os
from pathlib import Path

# Force CPU platform (must be set before importing jax)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from equinox.nn import MLP
from mldsml.wandb_utils import init_wandb, load_dataset
from tqdm.auto import tqdm

import wandb
from neuralODE import constants
from neuralODE.data import format_data, sample_data
from neuralODE.normalization import Normalization2BP

# One-file training configuration.
CONFIG = {
    "wandb": {
        "project": "neuralODEs",
        "entity": "mlds-lab",
        "group": "ude-j2-single-script",
    },
    "data": {
        "dataset_name": "perturbed_2BP_j2_planar_100",
        "dataset_version": "latest",
        "num_trajs": 20,
        "normalize_2bp": True,
    },
    "train": {
        "seed": 41,
        "train_fraction": 0.8,
        "width": 64,
        "depth": 2,
        "accel_scale": 1,
        "mu": 3.986004418e5,
        "lr": 1e-3,
        "train_steps": 1500,
        "batch_size": 8,
        "l2_weight": 1e-5,
        "eval_every": 20,
        "rtol": 1e-5,
        "atol": 1e-6,
        "solver_max_steps": 50000,
    },
    "output": {
        "checkpoint_dir": "scripts/training/files/models/ude_j2_single",
        "plot_dir": "scripts/training/plots",
        "num_plot_orbits": 2,
    },
}


Array = jnp.ndarray


class PerturbationNet(eqx.Module):
    """Neural perturbation model a_pert = f_theta(state, time)."""

    mlp: MLP
    accel_scale: float

    def __init__(self, width: int, depth: int, *, accel_scale: float, key: Array):
        self.mlp = MLP(
            in_size=7,
            out_size=3,
            width_size=width,
            depth=depth,
            activation=jax.nn.tanh,
            final_activation=jax.nn.tanh,
            key=key,
        )
        self.accel_scale = float(accel_scale)

    def __call__(self, r: Array, v: Array, t_s: Array) -> Array:
        raw = self.mlp(jnp.concatenate([r, v, jnp.array([t_s])], axis=0))
        return self.accel_scale * raw


class UDEDynamics(eqx.Module):
    """Known 2BP + learned perturbation."""

    net: PerturbationNet
    mu: float

    def __init__(
        self,
        mu: float,
        width: int,
        depth: int,
        *,
        accel_scale: float,
        key: Array,
    ):
        self.net = PerturbationNet(
            width,
            depth,
            accel_scale=accel_scale,
            key=key,
        )
        self.mu = float(mu)

    def perturbation(self, t_s: Array, y: Array) -> Array:
        return self.net(y[:3], y[3:], t_s)

    def __call__(self, t_s: Array, y: Array, args=None) -> Array:
        r = y[:3]
        v = y[3:]
        r_norm = jnp.linalg.norm(r)
        a_grav = -self.mu * r / jnp.maximum(r_norm, 1e-12) ** 3
        a_pert = self.net(r, v, t_s)
        return jnp.concatenate([v, a_grav + a_pert], axis=0)


def split_train_test_indices(
    num_orbits: int,
    *,
    train_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split orbit indices into train/test while guaranteeing non-empty test set."""
    if num_orbits < 2:
        idx = np.array([0], dtype=int)
        return idx, idx

    rng = np.random.default_rng(seed)
    order = np.arange(num_orbits)
    rng.shuffle(order)

    n_train = max(1, int(np.floor(train_fraction * num_orbits)))
    n_train = min(n_train, num_orbits - 1)
    return order[:n_train], order[n_train:]


def solve_trajectory(
    model: UDEDynamics,
    ts: Array,
    y0: Array,
    *,
    rtol: float,
    atol: float,
    max_steps: int,
) -> Array:
    """Integrate one trajectory with full UDE dynamics."""
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(model),
        diffrax.Tsit5(),
        t0=float(ts[0]),
        t1=float(ts[-1]),
        dt0=float(ts[1] - ts[0]),
        y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=max_steps,
    )
    return sol.ys


def solve_base_trajectory(
    ts: Array,
    y0: Array,
    *,
    mu: float,
    rtol: float,
    atol: float,
    max_steps: int,
) -> Array:
    """Integrate one trajectory with only known 2BP dynamics."""

    def base_rhs(t_s, y, args=None):
        r = y[:3]
        v = y[3:]
        r_norm = jnp.linalg.norm(r)
        a_grav = -mu * r / jnp.maximum(r_norm, 1e-12) ** 3
        return jnp.concatenate([v, a_grav], axis=0)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(base_rhs),
        diffrax.Tsit5(),
        t0=float(ts[0]),
        t1=float(ts[-1]),
        dt0=float(ts[1] - ts[0]),
        y0=y0,
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=max_steps,
    )
    return sol.ys


def precompute_base_trajectories(
    t_data: Array,
    y_data: Array,
    *,
    mu: float,
    rtol: float,
    atol: float,
    max_steps: int,
) -> Array:
    """Precompute base 2BP trajectories for all orbits."""
    ys_base = []
    for i in range(int(t_data.shape[0])):
        ys_base.append(
            solve_base_trajectory(
                t_data[i],
                y_data[i, 0],
                mu=mu,
                rtol=rtol,
                atol=atol,
                max_steps=max_steps,
            ),
        )
    return jnp.stack(ys_base, axis=0)


def residual_loss_components(
    y_true: Array,
    y_pred: Array,
    y_base: Array,
    mask: Array,
    *,
    eps: float = 1e-8,
) -> tuple[Array, Array]:
    """Compute residual-state MPE and NMRSE components."""
    res_true = y_true[1:] - y_base[1:]
    res_pred = y_pred[1:] - y_base[1:]
    m = mask[1:]

    true_norm = jnp.linalg.norm(res_true, axis=-1)
    denom = jnp.where(true_norm < eps, eps, true_norm)
    mpe_steps = jnp.linalg.norm(res_true - res_pred, axis=-1) / denom * 100.0
    mpe_steps = jnp.where(m, mpe_steps, jnp.nan)
    mpe = jnp.nanmean(mpe_steps)

    m_aug = m[..., None]
    sq_err = jnp.where(m_aug, (res_true - res_pred) ** 2, jnp.nan)
    sq_true = jnp.where(m_aug, res_true**2, jnp.nan)
    mse = jnp.nanmean(sq_err)
    ref_power = jnp.nanmean(sq_true) + eps
    nmrse = jnp.sqrt(mse / ref_power + eps) * 100.0

    return mpe, nmrse


def trajectory_loss(
    model: UDEDynamics,
    ts: Array,
    ys: Array,
    mask: Array,
    y_base: Array,
    *,
    l2_weight: float,
    rtol: float,
    atol: float,
    max_steps: int,
) -> Array:
    """Residual-state objective for one orbit."""
    pred = solve_trajectory(
        model,
        ts,
        ys[0],
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    mpe, nmrse = residual_loss_components(ys, pred, y_base, mask)
    pert = jax.vmap(model.perturbation)(ts, ys)
    reg = jnp.mean(jnp.sum(pert**2, axis=-1))
    return mpe + nmrse + l2_weight * reg


def batch_loss(
    model: UDEDynamics,
    t_batch: Array,
    y_batch: Array,
    m_batch: Array,
    b_batch: Array,
    *,
    l2_weight: float,
    rtol: float,
    atol: float,
    max_steps: int,
) -> Array:
    """Mean loss across an orbit mini-batch."""
    vals = []
    for i in range(int(t_batch.shape[0])):
        vals.append(
            trajectory_loss(
                model,
                t_batch[i],
                y_batch[i],
                m_batch[i],
                b_batch[i],
                l2_weight=l2_weight,
                rtol=rtol,
                atol=atol,
                max_steps=max_steps,
            ),
        )
    return jnp.mean(jnp.stack(vals))


def evaluate_split(
    model: UDEDynamics,
    t_data: Array,
    y_data: Array,
    m_data: Array,
    b_data: Array,
    *,
    l2_weight: float,
    rtol: float,
    atol: float,
    max_steps: int,
) -> dict[str, float]:
    """Evaluate mean loss, MPE, and NMRSE over a split."""
    losses = []
    mpes = []
    nmrses = []

    for i in range(int(t_data.shape[0])):
        pred = solve_trajectory(
            model,
            t_data[i],
            y_data[i, 0],
            rtol=rtol,
            atol=atol,
            max_steps=max_steps,
        )
        mpe, nmrse = residual_loss_components(y_data[i], pred, b_data[i], m_data[i])
        pert = jax.vmap(model.perturbation)(t_data[i], y_data[i])
        reg = jnp.mean(jnp.sum(pert**2, axis=-1))
        losses.append(float(mpe + nmrse + l2_weight * reg))
        mpes.append(float(mpe))
        nmrses.append(float(nmrse))

    return {
        "loss": float(np.mean(losses)),
        "mpe": float(np.mean(mpes)),
        "nmrse": float(np.mean(nmrses)),
    }


def plot_orbit_xy(
    ts: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_base: np.ndarray,
    out_path: str,
) -> None:
    """Save XY trajectory comparison plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(y_base[:, 0], y_base[:, 1], "--", label="2BP base")
    ax.plot(y_true[:, 0], y_true[:, 1], label="True")
    ax.plot(y_pred[:, 0], y_pred[:, 1], "--", label="Pred")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"XY trajectory (T={len(ts)})")
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_perturbation_components(
    ts: np.ndarray, y_true: np.ndarray, model: UDEDynamics, mu: float, out_path: str
) -> None:
    """Save true-vs-pred perturbation acceleration component plot."""
    import matplotlib.pyplot as plt

    t = ts
    r = y_true[:, :3]
    v = y_true[:, 3:]

    dvdt = np.gradient(v, t, axis=0)
    r_norm = np.linalg.norm(r, axis=-1, keepdims=True)
    a_base = -mu * r / np.maximum(r_norm, 1e-12) ** 3
    a_true = dvdt - a_base

    a_pred = np.array(jax.vmap(model.perturbation)(jnp.array(ts), jnp.array(y_true)))

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    labels = ["ax", "ay", "az"]
    for i, ax in enumerate(axes):
        ax.plot(t, a_true[:, i], label=f"true {labels[i]}")
        ax.plot(t, a_pred[:, i], "--", label=f"pred {labels[i]}")
        ax.legend(loc="best")
        ax.set_ylabel("acc")
    axes[-1].set_xlabel("time")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_checkpoint(path: Path, model: UDEDynamics, metadata: dict) -> None:
    """Save checkpoint with JSON metadata header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        fp.write((json.dumps(metadata) + "\n").encode("utf-8"))
        eqx.tree_serialise_leaves(fp, model)


def main() -> None:
    cfg = CONFIG

    run = init_wandb(
        config=cfg,
        group=cfg["wandb"]["group"],
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        mode="online",
        reinit=True,
    )

    data_dict = load_dataset(
        cfg["data"]["dataset_name"],
        version=cfg["data"]["dataset_version"],
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
    )

    if int(cfg["data"]["num_trajs"]) > 0:
        data_dict = sample_data(data_dict, int(cfg["data"]["num_trajs"]))

    normalize_2bp = bool(cfg["data"]["normalize_2bp"])
    if normalize_2bp:
        normalizer = Normalization2BP(
            l_char=constants.RADIUS_EARTH,
            mu=constants.MU_EARTH,
        )
        data_dict = normalizer.normalize_dataset(data_dict)

    data = format_data(data_dict)
    t_all = data["t"]
    y_all = data["y"]
    m_all = data["mask"]

    train_idx, test_idx = split_train_test_indices(
        int(y_all.shape[0]),
        train_fraction=float(cfg["train"]["train_fraction"]),
        seed=int(cfg["train"]["seed"]),
    )

    t_train, y_train, m_train = t_all[train_idx], y_all[train_idx], m_all[train_idx]
    t_test, y_test, m_test = t_all[test_idx], y_all[test_idx], m_all[test_idx]

    model_mu = 1.0 if normalize_2bp else float(cfg["train"]["mu"])
    b_train = precompute_base_trajectories(
        t_train,
        y_train,
        mu=model_mu,
        rtol=float(cfg["train"]["rtol"]),
        atol=float(cfg["train"]["atol"]),
        max_steps=int(cfg["train"]["solver_max_steps"]),
    )
    b_test = precompute_base_trajectories(
        t_test,
        y_test,
        mu=model_mu,
        rtol=float(cfg["train"]["rtol"]),
        atol=float(cfg["train"]["atol"]),
        max_steps=int(cfg["train"]["solver_max_steps"]),
    )

    model = UDEDynamics(
        mu=model_mu,
        width=int(cfg["train"]["width"]),
        depth=int(cfg["train"]["depth"]),
        accel_scale=float(cfg["train"]["accel_scale"]),
        key=jr.PRNGKey(int(cfg["train"]["seed"])),
    )

    optimizer = optax.adam(float(cfg["train"]["lr"]))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    loss_and_grad = eqx.filter_value_and_grad(
        lambda m, tb, yb, mb, bb: batch_loss(
            m,
            tb,
            yb,
            mb,
            bb,
            l2_weight=float(cfg["train"]["l2_weight"]),
            rtol=float(cfg["train"]["rtol"]),
            atol=float(cfg["train"]["atol"]),
            max_steps=int(cfg["train"]["solver_max_steps"]),
        ),
    )

    rng = np.random.default_rng(int(cfg["train"]["seed"]))
    steps = int(cfg["train"]["train_steps"])
    batch_size = int(cfg["train"]["batch_size"])
    eval_every = int(cfg["train"]["eval_every"])

    history = {
        "step": [],
        "train_loss": [],
        "test_loss": [],
        "train_mpe": [],
        "test_mpe": [],
        "train_nmrse": [],
        "test_nmrse": [],
    }

    progress = tqdm(total=steps, desc="Train UDE-J2 single", unit="step")
    for step in range(1, steps + 1):
        n_train = int(t_train.shape[0])
        replace = n_train < batch_size
        idx = rng.choice(n_train, size=min(batch_size, n_train), replace=replace)

        loss, grads = loss_and_grad(
            model,
            t_train[idx],
            y_train[idx],
            m_train[idx],
            b_train[idx],
        )
        updates, opt_state = optimizer.update(
            grads,
            opt_state,
            params=eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)

        progress.update(1)

        if step % eval_every == 0 or step == 1 or step == steps:
            train_metrics = evaluate_split(
                model,
                t_train,
                y_train,
                m_train,
                b_train,
                l2_weight=float(cfg["train"]["l2_weight"]),
                rtol=float(cfg["train"]["rtol"]),
                atol=float(cfg["train"]["atol"]),
                max_steps=int(cfg["train"]["solver_max_steps"]),
            )
            test_metrics = evaluate_split(
                model,
                t_test,
                y_test,
                m_test,
                b_test,
                l2_weight=float(cfg["train"]["l2_weight"]),
                rtol=float(cfg["train"]["rtol"]),
                atol=float(cfg["train"]["atol"]),
                max_steps=int(cfg["train"]["solver_max_steps"]),
            )

            history["step"].append(float(step))
            history["train_loss"].append(float(train_metrics["loss"]))
            history["test_loss"].append(float(test_metrics["loss"]))
            history["train_mpe"].append(float(train_metrics["mpe"]))
            history["test_mpe"].append(float(test_metrics["mpe"]))
            history["train_nmrse"].append(float(train_metrics["nmrse"]))
            history["test_nmrse"].append(float(test_metrics["nmrse"]))

            progress.set_postfix(
                train_loss=f"{train_metrics['loss']:.3e}",
                test_loss=f"{test_metrics['loss']:.3e}",
                train_mpe=f"{train_metrics['mpe']:.2f}",
                test_mpe=f"{test_metrics['mpe']:.2f}",
            )

            wandb.log(
                {
                    "step": float(step),
                    "train_loss": float(train_metrics["loss"]),
                    "test_loss": float(test_metrics["loss"]),
                    "train_mpe": float(train_metrics["mpe"]),
                    "test_mpe": float(test_metrics["mpe"]),
                    "train_nmrse": float(train_metrics["nmrse"]),
                    "test_nmrse": float(test_metrics["nmrse"]),
                },
            )

    progress.close()

    # Save checkpoint and log as artifact.
    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_path = ckpt_dir / f"ude_j2_single_{run.id}.eqx"
    metadata = {
        "run_id": run.id,
        "config": cfg,
        "train_indices": train_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "history": history,
        "model_mu": float(model_mu),
    }
    save_checkpoint(ckpt_path, model, metadata)

    model_artifact = wandb.Artifact(
        name=f"ude-j2-single-model-{run.id}",
        type="model",
        metadata=metadata,
    )
    model_artifact.add_file(str(ckpt_path), name=ckpt_path.name)
    run.log_artifact(model_artifact)

    # Diagnostic plots on a few train and test orbits.
    plot_dir = Path(cfg["output"]["plot_dir"])
    n_plot = int(cfg["output"]["num_plot_orbits"])

    def _choose_plot_indices(n_orbits: int, count: int) -> list[int]:
        if n_orbits <= 0:
            return []
        count = max(1, min(count, n_orbits))
        return [
            int(i)
            for i in np.unique(
                np.round(np.linspace(0, n_orbits - 1, count)).astype(int)
            )
        ]

    train_plot_idx = _choose_plot_indices(int(t_train.shape[0]), n_plot)
    test_plot_idx = _choose_plot_indices(int(t_test.shape[0]), n_plot)

    for split_name, ts_data, ys_data, bs_data, idxs in [
        ("train", t_train, y_train, b_train, train_plot_idx),
        ("test", t_test, y_test, b_test, test_plot_idx),
    ]:
        for local_i in idxs:
            pred = solve_trajectory(
                model,
                ts_data[local_i],
                ys_data[local_i, 0],
                rtol=float(cfg["train"]["rtol"]),
                atol=float(cfg["train"]["atol"]),
                max_steps=int(cfg["train"]["solver_max_steps"]),
            )

            xy_path = str(
                plot_dir / f"ude_j2_single_{run.id}_{split_name}_{local_i}_xy.png"
            )
            plot_orbit_xy(
                np.array(ts_data[local_i]),
                np.array(ys_data[local_i]),
                np.array(pred),
                np.array(bs_data[local_i]),
                xy_path,
            )
            wandb.log({f"{split_name}/orbit_{local_i}_xy": wandb.Image(xy_path)})

            force_path = str(
                plot_dir / f"ude_j2_single_{run.id}_{split_name}_{local_i}_force.png"
            )
            plot_perturbation_components(
                np.array(ts_data[local_i]),
                np.array(ys_data[local_i]),
                model,
                model_mu,
                force_path,
            )
            wandb.log({f"{split_name}/orbit_{local_i}_force": wandb.Image(force_path)})

    wandb.log(
        {
            "num_orbits_total": int(y_all.shape[0]),
            "num_orbits_train": int(y_train.shape[0]),
            "num_orbits_test": int(y_test.shape[0]),
            "final_train_loss": float(history["train_loss"][-1]),
            "final_test_loss": float(history["test_loss"][-1]),
            "checkpoint_path": str(ckpt_path),
        },
    )

    print("Training complete.")
    print(f"Run id: {run.id}")
    print(f"Checkpoint: {ckpt_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
```