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