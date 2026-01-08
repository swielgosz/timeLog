# Feature layer
``` python
def rhat_vhat_speed(y, eps=1e-8):
    """General feature set: unit position, inverse radius, speed, unit velocity.

    Adds radial/transverse velocity fractions (wrt r_hat) to expose alignment of
    motion without assuming a central body or mu. Works in 2BP or CR3BP.
    """

    r = y[:3]
    v = y[3:6]

    r_norm = jnp.linalg.norm(r) + eps
    v_norm = jnp.linalg.norm(v) + eps

    r_hat = r / r_norm
    v_hat = v / v_norm

    vr_frac = jnp.dot(v_hat, r_hat)
    vt_frac = jnp.linalg.norm(v_hat - vr_frac * r_hat)

    return jnp.concatenate(
        [
            jnp.array(
                [
                    1.0 / r_norm,
                    *r_hat,
                    v_norm,
                    *v_hat,
                    vr_frac,
                    vt_frac,
                ]
            ),
        ],
    )

	...

feature_layers = {...
    "rhat_vhat_speed": FeatureLayerSpec(
        rhat_vhat_speed,
        10,
        (
            "1/r",
            "sx",
            "sy",
            "sz",
            "|v|",
            "vx_hat",
            "vy_hat",
            "vz_hat",
            "vr_frac",
            "vt_frac",
        ),
    ),
}
```

# Attraction penalty in loss
This loss function penalizes accelerations pointing radially outward (dot product between position and acceleration vector is positive). This isn't very useful because this assumes that we know something about the physics, but we don't want to make such assumptions.
``` python
def _attraction_penalty(model, ti, y_pred, mask_i, eps=1e-8):
    """
    Encourage accelerations to remain attractive (dot(position, accel) <= 0).
    Returns a penalty scaled to roughly match percent error (0-100).
    """

    def eval_sequence(ts, states):
        return jax.vmap(lambda t, y: model.func(t, y))(ts, states)

    derivs = jax.vmap(eval_sequence)(ti, y_pred)
    positions = y_pred[..., :3]
    accelerations = derivs[..., 3:]
    dots = jnp.sum(positions * accelerations, axis=-1)
    pos_norm = jnp.linalg.norm(positions, axis=-1)
    accel_norm = jnp.linalg.norm(accelerations, axis=-1)
    denom = pos_norm * accel_norm
    denom = jnp.where(denom < eps, eps, denom)
    cosine = dots / denom
    penalty = jnp.maximum(cosine, 0.0)  # positive values indicate repulsive accel
    if mask_i is not None:
        mask = jnp.asarray(mask_i, dtype=bool)
        penalty = jnp.where(mask, penalty, jnp.nan)
    return jnp.nanmean(penalty) * 100.0


def percent_error_with_attraction_loss(
    model,
    ti,
    yi,
    mask_i,
    attraction_weight=1.0,
    *,
    return_components=False,
):
    """
    Percent error loss augmented with an attraction penalty that discourages
    repulsive accelerations (dot(position, acceleration) > 0).
    """
    y_pred = get_y_pred(model, ti, yi, mask_i)
    y_true = yi[:, 1:, :]
    y_pred_aligned = y_pred[:, 1:, :]

    threshold = 1e-8
    true_norm = jnp.linalg.norm(y_true, axis=-1)
    safe_denominator = jnp.where(true_norm < threshold, threshold, true_norm)
    mpe = jnp.linalg.norm((y_true - y_pred_aligned), axis=-1) / safe_denominator * 100
    mpe_mean = jnp.nanmean(mpe)

    attraction_term = attraction_weight * _attraction_penalty(
        model,
        ti,
        y_pred,
        mask_i,
    )
    total = mpe_mean + attraction_term

    components = {
        "percent_error_component": mpe_mean,
        "attraction_component": attraction_term,
        "percent_error_with_attraction": total,
    }
    if return_components:
        return total, components
    return total


percent_error_with_attraction_loss.supports_components = True
```