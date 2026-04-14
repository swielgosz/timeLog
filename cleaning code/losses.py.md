``` python
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp

"""
Loss functions for neuralODE training.

Functionality:
- Standard state-space losses (MSE, percent error, residual-state losses)
- Optional regularization and physics-inspired penalties
- Combined residual-state + acceleration-error loss for perturbed orbit training

Example usage:
    loss_fn = get_loss_function("residual_state_normalized_mse_plus_acceleration_mpe")
    loss = loss_fn(model, ti, yi, mask_i)
"""


def get_y_pred(model, ti, yi, mask_i):
    """
    # IMPORTANT in_axes(0,0) specifies that both ti and yi[:,0] should be batched along their first dimension - this is what allows us to use multiple times series vectors rather than just one
    """
    y_pred = jax.vmap(model, in_axes=(0, 0))(ti, yi[:, 0, :])
    return y_pred


def compute_l1_loss(model, mag=1e-5):
    params = eqx.filter(model, eqx.is_inexact_array)
    l1_loss = sum(jnp.nansum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params))
    return l1_loss * mag


def compute_l2_loss(model, mag=1e-4):
    params = eqx.filter(model, eqx.is_inexact_array)
    l2_loss = sum(jnp.nansum(p**2) for p in jax.tree_util.tree_leaves(params))
    return l2_loss * mag


def mean_squared_error_loss(model, ti, yi, mask_i):
    y_pred = get_y_pred(model, ti, yi, mask_i)
    return jnp.nanmean((yi - y_pred) ** 2)


def mse_with_reg_loss(model, ti, yi, l1_lambda=1e-5, l2_lambda=1e-4):
    mse_loss = mean_squared_error_loss(model, ti, yi)
    l1_loss = compute_l1_loss(model, mag=l1_lambda)
    l2_loss = compute_l2_loss(model, mag=l2_lambda)
    return mse_loss + l1_lambda * l1_loss + l2_lambda * l2_loss


def relative_squared_error_loss(model, ti, yi, mask_i):
    y_pred = get_y_pred(model, ti, yi, mask_i)
    error = yi - y_pred
    relative_squared_error = jnp.nansum(error**2, axis=-1) / (
        jnp.nansum(yi**2, axis=-1) + 1e-8
    )
    return relative_squared_error


def percent_error_loss(model, ti, yi, mask_i):
    y_pred = get_y_pred(model, ti, yi, mask_i)
    y_true = yi[:, 1:, :]
    y_pred = y_pred[:, 1:, :]

    threshold = 1e-8
    true_norm = jnp.linalg.norm(y_true, axis=-1)
    safe_denominator = jnp.where(true_norm < threshold, threshold, true_norm)
    mpe = jnp.linalg.norm((y_true - y_pred), axis=-1) / safe_denominator * 100
    mpe_mean = jnp.nanmean(mpe)
    return mpe_mean


def compute_residual_state_mpe(
    residual_true,
    residual_pred,
    mask=None,
    eps=1e-8,
):
    """Compute mean percent error over residual state trajectories."""
    true_norm = jnp.linalg.norm(residual_true, axis=-1)
    safe_denominator = jnp.where(true_norm < eps, eps, true_norm)
    mpe = (
        jnp.linalg.norm((residual_true - residual_pred), axis=-1)
        / safe_denominator
        * 100
    )
    if mask is not None:
        mpe = jnp.where(mask, mpe, jnp.nan)
    return jnp.nanmean(mpe)


def compute_residual_state_normalized_mse(
    residual_true,
    residual_pred,
    mask=None,
    eps=1e-8,
):
    """Compute per-dimension normalized MSE over residual-state trajectories."""
    sq_error = (residual_true - residual_pred) ** 2
    ref_power = residual_true**2

    if mask is not None:
        mask_expanded = mask[..., None]
        sq_error = jnp.where(mask_expanded, sq_error, jnp.nan)
        ref_power = jnp.where(mask_expanded, ref_power, jnp.nan)

    mse_per_dim = jnp.nanmean(sq_error, axis=(0, 1))
    ref_per_dim = jnp.nanmean(ref_power, axis=(0, 1))
    nmse_per_dim = mse_per_dim / (ref_per_dim + eps)
    return jnp.nanmean(nmse_per_dim)


def _predicted_acceleration_from_model(model, ti, y_pred):
    """
    Compute predicted acceleration directly from model dynamics.

    Example usage:
        a_pred = _predicted_acceleration_from_model(model, ti, y_pred)
    """

    def _one_orbit(ts, ys):
        dydt = jax.vmap(lambda t, y: model.func(t, y))(ts, ys)
        return dydt[:, 3:6]

    return jax.vmap(_one_orbit, in_axes=(0, 0))(ti, y_pred)


def compute_acceleration_mpe_from_states(
    y_true,
    y_pred,
    ti,
    model=None,
    mask=None,
    eps=1e-8,
    a_true=None,
):
    """
    Compute mean percent error between true and predicted accelerations.

    True acceleration uses `a_true` labels when provided, otherwise velocity
    finite differences. Predicted acceleration uses `model.func` when `model`
    is provided.

    Example usage:
        acc_mpe = compute_acceleration_mpe_from_states(
            y_true,
            y_pred,
            ti,
            model=model,
        )
    """
    if a_true is None:
        v_true = y_true[..., 3:6]
        a_true = jax.vmap(_finite_difference_acceleration, in_axes=(0, 0))(
            ti,
            v_true,
        )

    if model is None:
        v_pred = y_pred[..., 3:6]
        a_pred = jax.vmap(_finite_difference_acceleration, in_axes=(0, 0))(
            ti,
            v_pred,
        )
    else:
        a_pred = _predicted_acceleration_from_model(model, ti, y_pred)

    true_norm = jnp.linalg.norm(a_true, axis=-1)
    safe_denominator = jnp.where(true_norm < eps, eps, true_norm)
    mpe = jnp.linalg.norm(a_true - a_pred, axis=-1) / safe_denominator * 100.0

    if mask is not None:
        mpe = jnp.where(mask, mpe, jnp.nan)
    return jnp.nanmean(mpe)


def compute_residual_state_nmse_plus_acceleration_mpe(
    residual_true,
    residual_pred,
    y_true,
    y_pred,
    ti,
    model=None,
    mask=None,
    residual_weight=1.0,
    acceleration_weight=1,
    eps=1e-8,
    a_true=None,
):
    """
    Blend residual-state NMSE with trajectory acceleration MPE.

    Example usage:
        total, nmse, acc_mpe = compute_residual_state_nmse_plus_acceleration_mpe(
            residual_true, residual_pred, y_true, y_pred, ti,
        )
    """
    residual_nmse = compute_residual_state_normalized_mse(
        residual_true,
        residual_pred,
        mask=mask,
        eps=eps,
    )
    acceleration_mpe = compute_acceleration_mpe_from_states(
        y_true,
        y_pred,
        ti,
        model=model,
        mask=mask,
        eps=eps,
        a_true=a_true,
    )
    total = residual_weight * residual_nmse + acceleration_weight * acceleration_mpe
    return total, residual_nmse, acceleration_mpe


def _two_body_rhs(t, y, mu):
    del t
    r = y[:3]
    v = y[3:6]
    r_norm = jnp.linalg.norm(r)
    a = -mu * r / (jnp.maximum(r_norm, 1e-12) ** 3)
    return jnp.concatenate((v, a), axis=0)


def get_two_body_reference_trajectory(model, ti, yi0, mu=1.0, max_steps=100_000):
    """Integrate known 2BP dynamics for each batch sample to build a reference."""

    def _solve_one(ts, y0):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(_two_body_rhs),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            args=mu,
            stepsize_controller=diffrax.PIDController(
                rtol=getattr(model, "rtol", 1e-6),
                atol=getattr(model, "atol", 1e-8),
            ),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=max_steps,
        )
        return sol.ys

    return jax.vmap(_solve_one, in_axes=(0, 0))(ti, yi0)


def residual_state_percent_error_loss(model, ti, yi, mask_i):
    """Mean percent error on residual state relative to known 2BP reference."""
    y_pred = get_y_pred(model, ti, yi, mask_i)

    mu = getattr(getattr(model, "func", model), "scalar", 1.0)
    y_2bp = get_two_body_reference_trajectory(model, ti, yi[:, 0, :], mu=mu)
    y_2bp = jax.lax.stop_gradient(y_2bp)

    residual_true = yi[:, 1:, :] - y_2bp[:, 1:, :]
    residual_pred = y_pred[:, 1:, :] - y_2bp[:, 1:, :]

    mask = None if mask_i is None else mask_i[:, 1:]
    return compute_residual_state_mpe(
        residual_true,
        residual_pred,
        mask=mask,
    )


def residual_state_normalized_mse_loss(model, ti, yi, mask_i):
    """Normalized MSE on residual state relative to known 2BP reference."""
    y_pred = get_y_pred(model, ti, yi, mask_i)

    mu = getattr(getattr(model, "func", model), "scalar", 1.0)
    y_2bp = get_two_body_reference_trajectory(model, ti, yi[:, 0, :], mu=mu)
    y_2bp = jax.lax.stop_gradient(y_2bp)

    residual_true = yi[:, 1:, :] - y_2bp[:, 1:, :]
    residual_pred = y_pred[:, 1:, :] - y_2bp[:, 1:, :]

    mask = None if mask_i is None else mask_i[:, 1:]
    return compute_residual_state_normalized_mse(
        residual_true,
        residual_pred,
        mask=mask,
    )


def residual_state_normalized_mse_plus_acceleration_mpe_loss(
    model,
    ti,
    yi,
    mask_i,
    residual_weight=1.0,
    acceleration_weight=1e-2,
    a_true_i=None,
):
    """
    Residual-state NMSE plus weighted acceleration mean percent error.

    This extends `residual_state_normalized_mse` with an acceleration-shape term.

    Example usage:
        loss = residual_state_normalized_mse_plus_acceleration_mpe_loss(
            model, ti, yi, mask_i, acceleration_weight=1e-2,
        )
    """
    y_pred = get_y_pred(model, ti, yi, mask_i)

    mu = getattr(getattr(model, "func", model), "scalar", 1.0)
    y_2bp = get_two_body_reference_trajectory(model, ti, yi[:, 0, :], mu=mu)
    y_2bp = jax.lax.stop_gradient(y_2bp)

    y_true_eff = yi[:, 1:, :]
    y_pred_eff = y_pred[:, 1:, :]
    ti_eff = ti[:, 1:]

    residual_true = y_true_eff - y_2bp[:, 1:, :]
    residual_pred = y_pred_eff - y_2bp[:, 1:, :]
    a_true_eff = None if a_true_i is None else a_true_i[:, 1:, :]

    mask = None if mask_i is None else mask_i[:, 1:]

    total, _, _ = compute_residual_state_nmse_plus_acceleration_mpe(
        residual_true,
        residual_pred,
        y_true_eff,
        y_pred_eff,
        ti_eff,
        model=model,
        mask=mask,
        residual_weight=residual_weight,
        acceleration_weight=acceleration_weight,
        a_true=a_true_eff,
    )
    return total


def percent_error_plus_nmse_components(
    model,
    ti,
    yi,
    mask_i,
    eps=1e-8,
):
    """Return the component terms used in percent_error_plus_nmse loss."""
    y_pred = get_y_pred(model, ti, yi, mask_i)
    y_true = yi[:, 1:, :]
    y_pred = y_pred[:, 1:, :]

    threshold = 1e-8
    true_norm = jnp.linalg.norm(y_true, axis=-1)
    safe_denominator = jnp.where(true_norm < threshold, threshold, true_norm)
    mpe = jnp.linalg.norm((y_true - y_pred), axis=-1) / safe_denominator * 100
    mpe_mean = jnp.nanmean(mpe)

    mse = jnp.nanmean((y_true - y_pred) ** 2)
    ref_power = jnp.nanmean(y_true**2) + eps
    normalized_rmse = jnp.sqrt(mse / ref_power + eps) * 100.0

    return mpe_mean, normalized_rmse


def percent_error_plus_nmse_loss(
    model,
    ti,
    yi,
    mask_i,
    eps=1e-8,
):
    """Blend percent error with a normalized MSE term expressed as percent."""
    mpe_mean, normalized_rmse = percent_error_plus_nmse_components(
        model,
        ti,
        yi,
        mask_i,
        eps=eps,
    )
    return mpe_mean + normalized_rmse


def percent_error_norm_cr3bp_loss(model, ti, yi, mask_i):
    y_pred = jax.vmap(model, in_axes=(0, 0))(ti, yi[:, 0, :])  # (batch, time, state)
    # Exclude the first time step
    y_true = yi[:, 1:, :]
    y_pred = y_pred[:, 1:, :]
    mask_aug = mask_i[:, 1:, None]  # Exclude first time step and expand for state

    threshold = 1e-8
    true_norm = jnp.linalg.norm(y_true, axis=-1)
    safe_denominator = jnp.where(true_norm < threshold, threshold, true_norm)
    mpe = jnp.linalg.norm((y_true - y_pred), axis=-1) / safe_denominator * 100

    # Only include valid (unmasked) entries
    mpe = jnp.where(mask_aug.squeeze(-1), mpe, jnp.nan)
    mpe_mean = jnp.nanmean(mpe)
    return mpe_mean


def percent_error_with_reg_loss(
    model,
    ti,
    yi,
    mask_i,
    l1_lambda=1e-5,
    l2_lambda=1e-5,
):
    """
    Percent error norm loss with L1 and L2 regularization on model parameters.
    """
    mpe_mean = percent_error_loss(model, ti, yi, mask_i)

    # Regularization terms
    l1_loss = compute_l1_loss(model, mag=l1_lambda)
    l2_loss = compute_l2_loss(model, mag=l2_lambda)

    return mpe_mean + l1_lambda * l1_loss + l2_lambda * l2_loss


def percent_error_with_energy_loss(model, ti, yi, mask_i, mu=1.0, energy_weight=0.2):
    """
    Percent error loss (norm-based) with an added energy conservation component.
    """
    # Get model predictions
    y_pred = get_y_pred(model, ti, yi, mask_i)
    y_true = yi[:, 1:, :]
    y_pred = y_pred[:, 1:, :]

    threshold = 1e-8
    true_norm = jnp.linalg.norm(y_true, axis=-1)
    safe_denominator = jnp.where(true_norm < threshold, threshold, true_norm)
    mpe = jnp.linalg.norm((y_true - y_pred), axis=-1) / safe_denominator * 100
    mpe_mean = jnp.nanmean(mpe)

    # --- Energy conservation component ---
    # Calculate specific orbital energy for each state: E = v^2/2 - mu/r
    def calc_energy(state):
        pos, vel = state[:3], state[3:6]
        r = jnp.sqrt(jnp.nansum(pos**2))
        v2 = jnp.nansum(vel**2)
        return 0.5 * v2 - mu / r

    # Compute energy for true and predicted trajectories
    true_energy = jax.vmap(jax.vmap(calc_energy))(y_true)
    pred_energy = jax.vmap(jax.vmap(calc_energy))(y_pred)
    # Mean squared energy drift error (normalized)
    energy_scale = jnp.nanmean(jnp.abs(true_energy)) + threshold
    # Calculate energy for true and predicted trajectories
    instant_energy_error = jnp.nanmean((true_energy - pred_energy) ** 2) / (
        energy_scale**2
    )

    # Combine percent error and energy drift error
    total_loss = mpe_mean + energy_weight * instant_energy_error
    return total_loss


loss_functions = {
    "mean_squared_error": mean_squared_error_loss,
    "mse_with_reg": mse_with_reg_loss,
    "relative_squared_error": relative_squared_error_loss,
    "percent_error": percent_error_loss,
    "residual_state_percent_error": residual_state_percent_error_loss,
    "residual_state_normalized_mse": residual_state_normalized_mse_loss,
    "residual_state_normalized_mse_plus_acceleration_mpe": (
        residual_state_normalized_mse_plus_acceleration_mpe_loss
    ),
    "percent_error_plus_nmse": percent_error_plus_nmse_loss,
    "percent_error_norm_cr3bp": percent_error_norm_cr3bp_loss,
    "percent_error_with_reg": percent_error_with_reg_loss,
    "percent_error_with_energy": percent_error_with_energy_loss,
}


def get_loss_function(name):
    # If already a callable (function), just return it
    if callable(name):
        return name
    # If it's a string, look up in the dictionary
    if isinstance(name, str):
        try:
            return loss_functions[name]
        except KeyError:
            raise ValueError(
                f"Unsupported loss function: '{name}'. "
                f"Available options: {list(loss_functions.keys())}",
            )
    raise TypeError(
        f"Loss function must be a string or callable, got {type(name)}",
    )


def jacobian_frobenius_norm_sq(model, t, y):
    # Computes the squared Frobenius norm of the Jacobian of model(t, y) w.r.t. y
    jac = jax.jacrev(lambda y_: model(t, y_))(y)
    return jnp.nansum(jac**2)
```