``` python
        pred_acc = jax.vmap(self.model.func, in_axes=(1, 1))(t_zeros, states)

        # Compute the difference between accelerations
        acc_diff = true_acc - pred_acc

        # Create a mask for points inside Earth's radius
        positions = grid_points
        distances = jnp.linalg.norm(positions, axis=1)
        inside_earth = distances <= self.earth_radius_normalized * 1.1

        # Set accelerations to NaN for points inside Earth's radius
        true_acc = jnp.where(inside_earth[:, None], jnp.nan, true_acc)
        pred_acc = jnp.where(inside_earth[:, None], jnp.nan, pred_acc)
        acc_diff = jnp.where(inside_earth[:, None], jnp.nan, acc_diff)```