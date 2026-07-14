Sundman Time and Neural ODE Training for 2BP

Yes, training a neural ODE in Sundman time is mathematically valid, but only if we are explicit about what vector field the model is learning. A Sundman transform changes the independent variable from physical nondimensional time, usually denoted (\tau), to a fictitious time variable (s). If the original dynamics are

$$
\frac{d\mathbf{x}}{d\tau} = \mathbf{f}(\mathbf{x}),
$$

and we define the Sundman transformation by

$$
\frac{d\tau}{ds} = g(\mathbf{x}),
$$

then the transformed dynamics become

$$
\frac{d\mathbf{x}}{ds} = g(\mathbf{x})\mathbf{f}(\mathbf{x}).
$$

This is still a valid ODE. However, it is not the same vector field as the physical-time system. The state-space trajectory is the same, assuming (g(\mathbf{x}) > 0), but the trajectory is traversed with respect to a different independent variable.

For normalized two-body dynamics with (\mu = 1), the physical-time system is

$$
\mathbf{x} =
\begin{bmatrix}
\mathbf{r} \
\mathbf{v}
\end{bmatrix},
\qquad
\frac{d\mathbf{x}}{d\tau}

\begin{bmatrix}
\mathbf{v} \
-\frac{\mathbf{r}}{r^3}
\end{bmatrix}.
$$

A common Sundman choice is

$$
\frac{d\tau}{ds} = r.
$$

Then the transformed two-body dynamics are

$$
\frac{d\mathbf{x}}{ds}

r
\begin{bmatrix}
\mathbf{v} \
-\frac{\mathbf{r}}{r^3}
\end{bmatrix}

\begin{bmatrix}
r\mathbf{v} \
-\frac{\mathbf{r}}{r^2}
\end{bmatrix}.
$$

So if a neural ODE is trained in Sundman time, its target vector field is

$$
\frac{d\mathbf{x}}{ds}

\begin{bmatrix}
r\mathbf{v} \
-\frac{\mathbf{r}}{r^2}
\end{bmatrix},
$$

not the physical-time vector field

$$
\frac{d\mathbf{x}}{d\tau}

\begin{bmatrix}
\mathbf{v} \
-\frac{\mathbf{r}}{r^3}
\end{bmatrix}.
$$

This distinction matters because the network output no longer has the same physical interpretation. In physical time, the first three components of the state derivative are velocity, and the last three components are acceleration. In Sundman time, the first three components are

$$
\frac{d\mathbf{r}}{ds} = r\mathbf{v},
$$

and the last three are

$$
\frac{d\mathbf{v}}{ds} = r\mathbf{a}.
$$

Therefore, if the architecture hard-codes the physical-time kinematic relationship

$$
\dot{\mathbf{r}} = \mathbf{v},
$$

that structure cannot be used unchanged in Sundman time. In Sundman time, the corresponding relationship is

$$
\mathbf{r}’ = r\mathbf{v},
$$

where the prime denotes differentiation with respect to (s).

A clean way to use this in a neural ODE is to keep the network responsible for learning the physical acceleration, but apply the Sundman scaling outside the network. Instead of training a physical-time model of the form

$$
\dot{\mathbf{x}}

\begin{bmatrix}
\mathbf{v} \
\mathbf{a}_\theta(\mathbf{x})
\end{bmatrix},
$$

we can integrate the Sundman-time system

$$
\mathbf{x}’

\begin{bmatrix}
r\mathbf{v} \
r\mathbf{a}_\theta(\mathbf{x})
\end{bmatrix}.
$$

This is attractive because the network still represents the physical acceleration field,

$$
\mathbf{a}_\theta(\mathbf{x}) \approx -\frac{\mathbf{r}}{r^3},
$$

while the ODE solver integrates the reparameterized dynamics in Sundman time. For the exact normalized 2BP dynamics,

$$
r\mathbf{a}

r\left(-\frac{\mathbf{r}}{r^3}\right)

-\frac{\mathbf{r}}{r^2}.
$$

This may help training because the physical acceleration magnitude scales like

$$
|\mathbf{a}| = \frac{1}{r^2}.
$$

Near periapsis, (r) is small, so the physical-time vector field changes more rapidly and has larger magnitude. With the Sundman choice (d\tau/ds = r), the transformed velocity derivative scales like

$$
\left|\frac{d\mathbf{v}}{ds}\right|

\left|-\frac{\mathbf{r}}{r^2}\right|

\frac{1}{r}.
$$

This is still larger near periapsis, but it is less extreme than the physical-time acceleration scaling of (1/r^2). So the Sundman transform can make the learning and integration problem less uneven around an eccentric orbit.

However, there is an important data issue. If the training data are sampled at physical times (\tau_k), but the neural ODE is integrated in (s), then the corresponding Sundman-time grid (s_k) must be known. Since

$$
\frac{d\tau}{ds} = g(\mathbf{x}),
$$

we have

$$
\frac{ds}{d\tau} = \frac{1}{g(\mathbf{x})}.
$$

For (g(\mathbf{x}) = r),

$$
ds = \frac{d\tau}{r}.
$$

Therefore,

$$
s_k = \int_{\tau_0}^{\tau_k} \frac{1}{r(\tau)},d\tau.
$$

This means the Sundman-time grid is trajectory-dependent. For 2BP, this is manageable because (r(\tau)) is known from the trajectory data, but it must be accounted for.

Another clean option is to augment the state with physical time:

$$
\frac{d}{ds}
\begin{bmatrix}
\mathbf{x} \
\tau
\end{bmatrix}

\begin{bmatrix}
g(\mathbf{x})\mathbf{f}_\theta(\mathbf{x}) \
g(\mathbf{x})
\end{bmatrix}.
$$

Then the model integrates in (s), while physical time (\tau) is tracked as an additional dependent variable.

For 2BP neural ODE training, there are three related but distinct ideas:

1. Train/integrate in Sundman time. This changes the independent variable and therefore changes the learned vector field.
2. Sample data uniformly in anomaly instead of uniformly in time. This changes the distribution of training points around the orbit, but does not necessarily change the vector field being learned.
3. Weight or normalize the loss by radius or dynamics magnitude. This changes the optimization objective, but not the underlying ODE.

These ideas can be related, but they are not equivalent.

For my use case, a good first test would be: nondimensionalize the 2BP equations first so that (\mu = 1), then use a Sundman-wrapped physical acceleration model,

$$
\mathbf{x}’

\begin{bmatrix}
r\mathbf{v} \
r\mathbf{a}_\theta(\mathbf{x})
\end{bmatrix},
\qquad
\frac{d\tau}{ds} = r.
$$

This keeps the learned acceleration physically interpretable while still allowing the solver to integrate the trajectory in a stretched time variable.

In summary, training in Sundman time is valid. It may help if the training failure is related to nonuniform orbital speed, periapsis sensitivity, solver difficulty, or poor gradient conditioning over eccentric trajectories. However, the derivative being learned changes from (d\mathbf{x}/d\tau) to (d\mathbf{x}/ds). Therefore, the cleanest approach is usually not to make the network learn an opaque Sundman-time vector field directly, but to let the network learn the physical acceleration and apply the Sundman scaling outside the network.


Applying Sundman Scaling Outside the Neural Network

The phrase “let the network learn the physical acceleration and apply the Sundman scaling outside the network” means: keep the neural network’s job the same as before. The network predicts the physical-time acceleration, but the ODE wrapper applies the Sundman scaling when constructing the derivative that gets integrated.

In the usual physical-time 2BP neural ODE, the state is

$$
\mathbf{x}

\begin{bmatrix}
\mathbf{r}\
\mathbf{v}
\end{bmatrix}
$$

and the model is

$$
\dot{\mathbf{x}}

\frac{d}{d\tau}
\begin{bmatrix}
\mathbf{r}\
\mathbf{v}
\end{bmatrix}

\begin{bmatrix}
\mathbf{v}\
\mathbf{a}_\theta(\mathbf{r},\mathbf{v})
\end{bmatrix}.
$$

Here, the network predicts

$$
\mathbf{a}_\theta(\mathbf{r},\mathbf{v})
\approx
-\frac{\mathbf{r}}{r^3}.
$$

So the network output has a clean physical meaning: acceleration with respect to nondimensional physical time (\tau).

If we apply Sundman time with

$$
\frac{d\tau}{ds}=r,
$$

then the transformed dynamics should be

$$
\frac{d}{ds}
\begin{bmatrix}
\mathbf{r}\
\mathbf{v}
\end{bmatrix}

r
\begin{bmatrix}
\mathbf{v}\
\mathbf{a}_\theta(\mathbf{r},\mathbf{v})
\end{bmatrix}

\begin{bmatrix}
r\mathbf{v}\
r\mathbf{a}_\theta(\mathbf{r},\mathbf{v})
\end{bmatrix}.
$$

This is what it means to apply the Sundman scaling outside the network. The network itself still returns

$$
\mathbf{a}_\theta(\mathbf{r},\mathbf{v}).
$$

Then the ODE function multiplies the physical-time vector field by (r):

$$
\mathbf{x}’

\begin{bmatrix}
r\mathbf{v}\
r\mathbf{a}_\theta
\end{bmatrix}.
$$

The physical-time model is

$$
\mathbf{a}\theta = \mathrm{NN}\theta(\mathbf{x}),
$$

$$
\dot{\mathbf{x}}

\begin{bmatrix}
\mathbf{v}\
\mathbf{a}_\theta
\end{bmatrix}.
$$

The Sundman-wrapped model is

$$
\mathbf{a}\theta = \mathrm{NN}\theta(\mathbf{x}),
$$

$$
\mathbf{x}’

\begin{bmatrix}
r\mathbf{v}\
r\mathbf{a}_\theta
\end{bmatrix}.
$$

The useful part is that the network does not have to learn the Sundman scaling. We already know the scaling analytically. For the choice

$$
\frac{d\tau}{ds}=r,
$$

the factor is just

$$
r=|\mathbf{r}|.
$$

So it is better to hard-code that known transformation than to ask the network to discover it from data.

If instead we ask the network to directly learn the Sundman-time vector field, then the target is

$$
\mathbf{x}’

\begin{bmatrix}
r\mathbf{v}\
-\frac{\mathbf{r}}{r^2}
\end{bmatrix}.
$$

That means the network has to represent both the physical acceleration law and the time scaling. The acceleration-like part is now

$$
\frac{d\mathbf{v}}{ds}

-\frac{\mathbf{r}}{r^2}.
$$

But this is not physical acceleration. It is physical acceleration multiplied by (r):

$$
\frac{d\mathbf{v}}{ds}

r\frac{d\mathbf{v}}{d\tau}.
$$

So if the network directly outputs (d\mathbf{x}/ds), its output is less physically interpretable.

By applying the scaling outside the network, we preserve the physical interpretation:

$$
\mathbf{a}_\theta
\approx
\frac{d\mathbf{v}}{d\tau}.
$$

Then the Sundman transformation is handled deterministically by the ODE wrapper:

$$
\frac{d\mathbf{x}}{ds}

r\frac{d\mathbf{x}}{d\tau}.
$$

This is useful for several reasons.

First, it reduces the burden on the network. The network only needs to learn the physical force or acceleration law. It does not need to also learn the radius-dependent time reparameterization.

Second, it keeps the learned quantity comparable to the known 2BP acceleration. We can still evaluate whether

$$
\mathbf{a}_\theta(\mathbf{r},\mathbf{v})
\approx
-\frac{\mathbf{r}}{r^3}.
$$

That means acceleration error plots, vector-field diagnostics, and physical interpretation remain meaningful.

Third, it makes the architecture consistent with mechanics. In physical time, the kinematic relationship is

$$
\dot{\mathbf{r}} = \mathbf{v}.
$$

In Sundman time, the relationship is

$$
\mathbf{r}’ = r\mathbf{v}.
$$

We do not want the network to learn that relationship. We already know it exactly. So we hard-code

$$
\mathbf{r}’ = r\mathbf{v}
$$

and only use the network for the learned acceleration term.

Fourth, it makes it easier to recover physical-time dynamics. Since the network still outputs physical acceleration, we do not need to undo the scaling to interpret the learned force field. The scaling only affects how the ODE is integrated.

In code, the physical-time model is conceptually:
```

def f_physical(tau, x, params):
    r_vec = x[:3]
    v_vec = x[3:]
    a = NN(params, x)  # physical acceleration
    return concatenate([v_vec, a])
```

The Sundman-wrapped model is:

```

def f_sundman(s, x, params):
    r_vec = x[:3]
    v_vec = x[3:]
    r = norm(r_vec)
    a = NN(params, x)  # physical acceleration
    return concatenate([r * v_vec, r * a])
```

So the network is identical. The ODE function around it changes.

For exact 2BP, if

$$
\mathbf{a}_\theta(\mathbf{x})

-\frac{\mathbf{r}}{r^3},
$$

then the Sundman-wrapped system becomes

$$
\mathbf{x}’

\begin{bmatrix}
r\mathbf{v}\
r\left(-\frac{\mathbf{r}}{r^3}\right)
\end{bmatrix}

\begin{bmatrix}
r\mathbf{v}\
-\frac{\mathbf{r}}{r^2}
\end{bmatrix}.
$$

So we get the correct Sundman-time dynamics while still learning the physical acceleration.

This is probably the cleanest version for 2BP neural ODE training because it gives the possible numerical or conditioning benefit of Sundman integration without making the neural network’s output harder to interpret.