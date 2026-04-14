new metrics.py:
```
import jax
import jax.numpy as jnp


# Percent error
def percent_error(y_true, y_pred):
    # Calculate the percent error for each feature at each timestep
    percent_error_per_element = (y_pred - y_true) / y_true * 100

    # Calculate average percent error of all features for each timestep
    mean_percent_error_per_timestep = jnp.mean(
        jnp.abs(percent_error_per_element),
        axis=2,
    )  # Shape: (num_orbits, timesteps)

    # Step 3: Calculate the mean percent error across all timesteps for each orbit
    percent_errors = jnp.mean(
        mean_percent_error_per_timestep,
        axis=1,
    )  # Shape: (num_orbits,)

    return percent_errors


def accumulated_state_error(y_true, y_pred, separate_rmse=True):
    """
    Computes the accumulated absolute error or RMSE for positions and velocities.

    Args:
        y_true (jnp.ndarray): True states (shape: [num_orbits, timesteps, features] or [timesteps, features]).
        y_pred (jnp.ndarray): Predicted states (same shape as y_true).
        rmse (bool): If True, compute RMSE instead of absolute error.
        separate_rmse (bool): If True, calculate RMSE separately for positions and velocities.

    Returns:
        jnp.ndarray: Accumulated error or RMSE.
                     - If separate_rmse=True, returns a tuple (accumulated_position_error, accumulated_velocity_error).
                     - Otherwise, returns a single accumulated error array.
    """
    # Ensure inputs are the same shape
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape."

    # Split positions and velocities
    r_true, v_true = y_true[..., :3], y_true[..., 3:]
    r_pred, v_pred = y_pred[..., :3], y_pred[..., 3:]

    if separate_rmse:
        # Calculate RMSE separately for positions and velocities
        position_error = jnp.sqrt(
            jnp.mean(
                (r_true - r_pred) ** 2,
                axis=y_true.ndim - 1,
            ),  # RMSE for positions; shape (num_orbits, timesteps) for multiple orbits or (timesteps,) for single orbit
        )
        velocity_error = jnp.sqrt(
            jnp.mean((v_true - v_pred) ** 2, axis=y_true.ndim - 1),
        )

        # Accumulate errors along timesteps
        accumulated_position_error = jnp.cumsum(
            position_error,
            axis=r_true.ndim - 2,
        )  # shape (num_orbits, timesteps) for multiple orbits or (timesteps,) for single orbit
        accumulated_velocity_error = jnp.cumsum(
            velocity_error,
            axis=v_true.ndim - 2,
        )

        accumulated_error = (accumulated_position_error, accumulated_velocity_error)

    else:
        # Calculate combined RMSE for all features
        state_error = jnp.sqrt(
            jnp.mean((y_true - y_pred) ** 2, axis=y_true.ndim - 1),
        )

        # Accumulate errors along timesteps
        accumulated_error = jnp.cumsum(state_error, axis=y_true.ndim - 2)

    return accumulated_error


def _eval_dynamics(dynamics, ti, yi):
    """Vectorize a dynamics function over (batch, time)."""

    def eval_sequence(t_seq, y_seq):
        return jax.vmap(
            dynamics,
            in_axes=(0, 0),
        )(t_seq, y_seq)

    return jax.vmap(eval_sequence)(ti, yi)


def _compute_predicted_acceleration(model, ti, yi):
    deriv_pred = _eval_dynamics(model.func, ti, yi)
    return deriv_pred[..., 3:]


def _compute_true_acceleration(true_dynamics, ti, yi):
    deriv_true = _eval_dynamics(true_dynamics, ti, yi)
    return deriv_true[..., 3:]


class AccelerationMetric:
    def __init__(self, true_dynamics):
        self.name = "acceleration_error"
        self.true_dynamics = true_dynamics

    def __call__(self, model, ti, yi, mask_i):
        metric_acc_error = compute_acceleration_errors(
            model,
            self.true_dynamics,
            ti,
            yi,
            mask_i,
        )
        return jnp.mean(metric_acc_error)


def compute_acceleration_errors(model, true_dynamics, ti, yi, mask_i):
    """
    Compute percent acceleration errors for every orbit and timestep, respecting masks.

    Returns a 1D array of errors (percentage) for all valid samples so that
    callers can aggregate however they like (mean, std, quantiles, etc.).
    """
    true_acc = _compute_true_acceleration(true_dynamics, ti, yi)
    pred_acc = _compute_predicted_acceleration(model, ti, yi)

    numer = jnp.linalg.norm(true_acc - pred_acc, axis=-1)
    denom = jnp.linalg.norm(true_acc, axis=-1)
    denom_safe = jnp.where(denom > 0, denom, 1.0)
    acc_error = numer / denom_safe * 100.0

    nan_mask = ~jnp.isnan(acc_error)
    full_mask = jnp.logical_and(nan_mask, mask_i)
    return acc_error[full_mask]


class AccelerationAngleMetric:
    """Mean angular error between predicted and true acceleration vectors."""

    def __init__(self, true_dynamics, eps=1e-8):
        self.name = "accel_angle_deg"
        self.true_dynamics = true_dynamics
        self.eps = eps

    def __call__(self, model, ti, yi, mask_i):
        true_acc = _compute_true_acceleration(self.true_dynamics, ti, yi)
        pred_acc = _compute_predicted_acceleration(model, ti, yi)

        dots = jnp.sum(true_acc * pred_acc, axis=-1)
        true_norm = jnp.linalg.norm(true_acc, axis=-1)
        pred_norm = jnp.linalg.norm(pred_acc, axis=-1)
        denom = true_norm * pred_norm
        denom_safe = jnp.where(denom > self.eps, denom, self.eps)
        cosine = jnp.clip(dots / denom_safe, -1.0, 1.0)
        angles = jnp.degrees(jnp.arccos(cosine))

        mask = jnp.logical_and(mask_i, jnp.isfinite(angles))
        masked_angles = jnp.where(mask, angles, jnp.nan)
        return jnp.nanmean(masked_angles)


class RadialAccelerationSignMetric:
    """Percent of steps where accel points outward instead of inward."""

    def __init__(self, eps=1e-8):
        self.name = "radial_sign_violation_pct"
        self.eps = eps

    def __call__(self, model, ti, yi, mask_i):
        pred_acc = _compute_predicted_acceleration(model, ti, yi)
        positions = yi[..., :3]

        dots = jnp.sum(positions * pred_acc, axis=-1)
        pos_norm = jnp.linalg.norm(positions, axis=-1)
        acc_norm = jnp.linalg.norm(pred_acc, axis=-1)
        denom = pos_norm * acc_norm
        denom_safe = jnp.where(denom > self.eps, denom, self.eps)
        cosine = dots / denom_safe

        valid_mask = jnp.logical_and(mask_i, jnp.isfinite(cosine))
        repulsive = jnp.logical_and(valid_mask, cosine > 0.0)
        total = jnp.sum(valid_mask).astype(jnp.float32)
        repulsive_count = jnp.sum(repulsive).astype(jnp.float32)
        violation_pct = jnp.where(
            total > 0,
            repulsive_count / total * 100.0,
            0.0,
        )
        return violation_pct


def compute_direction_flip_pct(
    pred_acc,
    mask,
    *,
    min_cosine_flip: float = 0.0,
    eps: float = 1e-8,
):
    """Percent of adjacent steps where acceleration direction changes sign."""
    has_prev = mask[..., :-1]
    has_next = mask[..., 1:]
    valid_pairs = jnp.logical_and(has_prev, has_next)

    curr_acc = pred_acc[..., :-1, :]
    next_acc = pred_acc[..., 1:, :]
    curr_norm = jnp.linalg.norm(curr_acc, axis=-1)
    next_norm = jnp.linalg.norm(next_acc, axis=-1)
    denom = curr_norm * next_norm
    safe_denom = jnp.where(denom > eps, denom, eps)

    cosine = jnp.sum(curr_acc * next_acc, axis=-1) / safe_denom
    finite_cosine = jnp.isfinite(cosine)
    considered = jnp.logical_and(valid_pairs, finite_cosine)
    flips = jnp.logical_and(considered, cosine < min_cosine_flip)

    total_pairs = jnp.sum(considered).astype(jnp.float32)
    flip_count = jnp.sum(flips).astype(jnp.float32)

    return jnp.where(
        total_pairs > 0.0,
        flip_count / total_pairs * 100.0,
        0.0,
    )


class AccelerationDirectionFlipMetric:
    """Percent of successive steps where the predicted accel direction flips."""

    def __init__(self, min_cosine_flip=0.0, eps=1e-8):
        self.name = "accel_dir_flip_pct"
        self.min_cosine_flip = min_cosine_flip
        self.eps = eps

    def __call__(self, model, ti, yi, mask_i):
        pred_acc = _compute_predicted_acceleration(model, ti, yi)
        return compute_direction_flip_pct(
            pred_acc,
            mask_i,
            min_cosine_flip=self.min_cosine_flip,
            eps=self.eps,
        )
```

old metrics.py:
``` python
import jax
import jax.numpy as jnp


# Percent error
def percent_error(y_true, y_pred):
    # Calculate the percent error for each feature at each timestep
    percent_error_per_element = (y_pred - y_true) / y_true * 100

    # Calculate average percent error of all features for each timestep
    mean_percent_error_per_timestep = jnp.mean(
        jnp.abs(percent_error_per_element),
        axis=2,
    )  # Shape: (num_orbits, timesteps)

    # Step 3: Calculate the mean percent error across all timesteps for each orbit
    percent_errors = jnp.mean(
        mean_percent_error_per_timestep,
        axis=1,
    )  # Shape: (num_orbits,)

    return percent_errors


def accumulated_state_error(y_true, y_pred, separate_rmse=True):
    """
    Computes the accumulated absolute error or RMSE for positions and velocities.

    Args:
        y_true (jnp.ndarray): True states (shape: [num_orbits, timesteps, features] or [timesteps, features]).
        y_pred (jnp.ndarray): Predicted states (same shape as y_true).
        rmse (bool): If True, compute RMSE instead of absolute error.
        separate_rmse (bool): If True, calculate RMSE separately for positions and velocities.

    Returns:
        jnp.ndarray: Accumulated error or RMSE.
                     - If separate_rmse=True, returns a tuple (accumulated_position_error, accumulated_velocity_error).
                     - Otherwise, returns a single accumulated error array.
    """
    # Ensure inputs are the same shape
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape."

    # Split positions and velocities
    r_true, v_true = y_true[..., :3], y_true[..., 3:]
    r_pred, v_pred = y_pred[..., :3], y_pred[..., 3:]

    if separate_rmse:
        # Calculate RMSE separately for positions and velocities
        position_error = jnp.sqrt(
            jnp.mean(
                (r_true - r_pred) ** 2,
                axis=y_true.ndim - 1,
            ),  # RMSE for positions; shape (num_orbits, timesteps) for multiple orbits or (timesteps,) for single orbit
        )
        velocity_error = jnp.sqrt(
            jnp.mean((v_true - v_pred) ** 2, axis=y_true.ndim - 1),
        )

        # Accumulate errors along timesteps
        accumulated_position_error = jnp.cumsum(
            position_error,
            axis=r_true.ndim - 2,
        )  # shape (num_orbits, timesteps) for multiple orbits or (timesteps,) for single orbit
        accumulated_velocity_error = jnp.cumsum(
            velocity_error,
            axis=v_true.ndim - 2,
        )

        accumulated_error = (accumulated_position_error, accumulated_velocity_error)

    else:
        # Calculate combined RMSE for all features
        state_error = jnp.sqrt(
            jnp.mean((y_true - y_pred) ** 2, axis=y_true.ndim - 1),
        )

        # Accumulate errors along timesteps
        accumulated_error = jnp.cumsum(state_error, axis=y_true.ndim - 2)

    return accumulated_error


class AccelerationMetric:
    def __init__(self, true_dynamics):
        self.name = "acceleration_error"
        self.true_dynamics = true_dynamics

    def __call__(self, model, ti, yi, mask_i):
        y_true = jax.vmap(
            lambda t_seq, y_seq: jax.vmap(
                self.true_dynamics,
                in_axes=(0, 0),
            )(t_seq, y_seq),
        )(ti, yi)

        # double vmap to apply model.func over both time and state sequences
        y_pred = jax.vmap(
            lambda t_seq, y_seq: jax.vmap(
                model.func,
                in_axes=(0, 0),
            )(t_seq, y_seq),
        )(ti, yi)

        acc_error = (
            jnp.linalg.norm(y_true - y_pred, axis=-1)
            / jnp.linalg.norm(y_true, axis=-1)
            * 100
        )
        # remove nans and mask
        nan_mask = ~jnp.isnan(acc_error)

        # select out mask_i values and nan values
        full_mask = jnp.logical_and(nan_mask, mask_i)
        metric_acc_error = acc_error[full_mask]
        acc_error_mean = jnp.mean(metric_acc_error)
        return acc_error_mean
```


