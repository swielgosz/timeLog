If the learned dynamics were perfect and your loss is trajectory error against the true 2BP trajectory, then the adjoint would not be “equal around the orbit.” It would usually be zero everywhere, or numerically very close to zero.

Reason: if the predicted trajectory exactly matches the true trajectory, then the state error is zero at every comparison time. For a standard squared-error loss, that means the loss is zero, and the derivative of the loss with respect to the predicted state is also zero. The adjoint starts from that loss sensitivity. If the loss sensitivity is zero, then the backward-propagated adjoint is also zero.

For terminal loss, the logic is:

\hat{\mathbf{z}}(T) = \mathbf{z}(T)

so

L = 0

and

\frac{\partial L}{\partial \hat{\mathbf{z}}(T)} = 0.

Therefore the terminal adjoint is zero:

\lambda(T) = 0.

If you propagate zero sensitivity backward through the dynamics, it stays zero:

\lambda(t) = 0

for all t.

For full-trajectory loss, the same idea applies at every saved time. If

\hat{\mathbf{z}}(t_k) = \mathbf{z}(t_k)

for all k, then each trajectory-error contribution is zero, so each loss-sensitivity contribution is also zero. The adjoint receives no nonzero “kicks” along the trajectory, so it remains zero everywhere.

There is one technical caveat. If your loss uses a raw norm or percent error,

\frac{\|\hat{\mathbf{z}}-\mathbf{z}\|}{\|\mathbf{z}\|},

then the derivative at exactly zero error can be nonsmooth because the norm has a cusp at zero. In practice, implementations often use MSE, RMSE with smoothing, or automatic differentiation conventions that return something effectively zero or numerically small. But conceptually, perfect prediction means no error signal, so no meaningful adjoint signal for training.

The important distinction is this: perfect dynamics do not imply equal nonzero adjoint around the orbit. Perfect dynamics plus an error-based loss imply zero adjoint around the orbit.

However, if you define a different objective that is not zero along the true trajectory, then the adjoint can be nonzero even with perfect dynamics. For example, suppose the objective is not “match the truth,” but instead “maximize final x-position” or “minimize distance from the Moon at final time.” Then even under perfect dynamics, the adjoint would describe how that objective depends on the state along the orbit. In that case, the adjoint would generally not be equal around the orbit. It would vary according to orbital sensitivity.

So for your neural ODE training case:

If the model perfectly learns 2BP and the loss is prediction error against 2BP data, the adjoint should vanish.

If the model perfectly learns 2BP but the loss is some nonzero mission objective, the adjoint can be nonzero and will generally vary around the orbit.

If the model is nearly perfect but not exact, the adjoint will be small only if the loss derivative is small. It may still show structure because even small trajectory errors can be dynamically amplified over a full orbit, especially in phase-sensitive regions.