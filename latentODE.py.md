```python
 """ 
 ODE model used for 2BP reconstruction tasks.

Defines a GRU encoder, latent ODE dynamics, and linear decoder.

Example usage:
    model = LatentODE(
        data_size=6,
        hidden_size=16,
        latent_size=16,
        width_size=64,
        depth=2,
        key=jr.PRNGKey(0),
    )
    z0, mean, logstd = model.encode(ts, ys, key=jr.PRNGKey(1))
    ys_hat = model.decode(ts, z0)
"""

from __future__ import annotations

from typing import Optional

import diffrax as dfx
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr


class LatentFunc(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP
    output_activation: callable = eqx.field(static=True)

    def __call__(self, t, y, args):
        return self.scale * self.output_activation(self.mlp(y))


class LatentODE(eqx.Module):
    """
    GRU encoder + latent ODE decoder.

    Encodes (t, y) into a latent z0, then decodes via ODE in hidden space.
    """

    func: LatentFunc
    rnn_cell: eqx.nn.GRUCell
    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden: eqx.nn.MLP
    hidden_to_data: eqx.nn.Linear

    hidden_size: int
    latent_size: int
    latent_activation_name: str = eqx.field(static=True)
    latent_final_activation_name: str = eqx.field(static=True)
    drift_output_activation_name: str = eqx.field(static=True)
    solver_rtol: Optional[float] = eqx.field(static=True)
    solver_atol: Optional[float] = eqx.field(static=True)
    solver_max_steps: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        data_size: int,
        hidden_size: int,
        latent_size: int,
        width_size: int,
        depth: int,
        latent_activation: str = "softplus",
        latent_final_activation: str = "tanh",
        drift_output_activation: str = "tanh",
        solver_rtol: Optional[float] = None,
        solver_atol: Optional[float] = None,
        solver_max_steps: int = 4096,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        mkey, gkey, hlkey, lhkey, hdkey = jr.split(key, 5)
        latent_activation_fn = getattr(jnn, latent_activation)
        latent_final_activation_fn = getattr(jnn, latent_final_activation)
        drift_output_activation_fn = getattr(jnn, drift_output_activation)
        scale = jnp.ones(())
        mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=latent_activation_fn,
            final_activation=latent_final_activation_fn,
            key=mkey,
        )
        self.func = LatentFunc(scale, mlp, drift_output_activation_fn)
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
        self.latent_activation_name = latent_activation
        self.latent_final_activation_name = latent_final_activation
        self.drift_output_activation_name = drift_output_activation
        self.solver_rtol = solver_rtol
        self.solver_atol = solver_atol
        self.solver_max_steps = int(solver_max_steps)

    def encode(self, ts, ys, *, key):
        # ts: (T,), ys: (T, D)
        data = jnp.concatenate([ts[:, None], ys], axis=1)
        h = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            h = self.rnn_cell(data_i, h)
        stats = self.hidden_to_latent(h)
        mean, logstd = stats[: self.latent_size], stats[self.latent_size :]
        std = jnp.exp(logstd)
        z = mean + jr.normal(key, (self.latent_size,)) * std
        return z, mean, logstd

    def decode(self, ts, z0):
        h0 = self.latent_to_hidden(z0)
        term = dfx.ODETerm(self.func)
        solver = dfx.Tsit5()
        dt0 = jnp.maximum(jnp.asarray(ts[1] - ts[0]), jnp.asarray(1e-6))
        solve_kwargs = {
            "saveat": dfx.SaveAt(ts=ts),
            "max_steps": self.solver_max_steps,
        }
        if self.solver_rtol is not None and self.solver_atol is not None:
            solve_kwargs["stepsize_controller"] = dfx.PIDController(
                rtol=self.solver_rtol,
                atol=self.solver_atol,
            )
        sol = dfx.diffeqsolve(
            term,
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=dt0,
            y0=h0,
            **solve_kwargs,
        )
        ys_hat = jax.vmap(self.hidden_to_data)(sol.ys)
        return ys_hat
```