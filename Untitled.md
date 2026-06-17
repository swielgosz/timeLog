So if I say “perturb the state at time t,” I mean something like:

\mathbf{r}(t)
\rightarrow
\mathbf{r}(t) + \delta \mathbf{r}(t)

and/or

\mathbf{v}(t)
\rightarrow
\mathbf{v}(t) + \delta \mathbf{v}(t)

not

\mathbf{a}(t)
\rightarrow
\mathbf{a}(t) + \delta \mathbf{a}(t)

Acceleration is not part of the state unless you explicitly define an augmented state that includes acceleration, which you usually would not for standard 2BP.

However, changing the state does change the vector field.

For 2BP,

f(\mathbf{z})
=
\begin{bmatrix}
\mathbf{v} \\
-\mu \dfrac{\mathbf{r}}{\|\mathbf{r}\|^3}
\end{bmatrix}

If you perturb velocity, you directly perturb the first half of the vector field:

\dot{\mathbf{r}} = \mathbf{v}

If you perturb position, you perturb the second half of the vector field:

\dot{\mathbf{v}} = -\mu \frac{\mathbf{r}}{\|\mathbf{r}\|^3}

So the relationship is:

\delta \mathbf{z}
=
\begin{bmatrix}
\delta \mathbf{r} \\
\delta \mathbf{v}
\end{bmatrix}

causes

\delta f
=
\frac{\partial f}{\partial \mathbf{z}}
\delta \mathbf{z}

For 2BP, the Jacobian has the block structure

\frac{\partial f}{\partial \mathbf{z}}
=
\begin{bmatrix}
\dfrac{\partial \dot{\mathbf{r}}}{\partial \mathbf{r}}
&
\dfrac{\partial \dot{\mathbf{r}}}{\partial \mathbf{v}}
\\
\dfrac{\partial \dot{\mathbf{v}}}{\partial \mathbf{r}}
&
\dfrac{\partial \dot{\mathbf{v}}}{\partial \mathbf{v}}
\end{bmatrix}

Since

\dot{\mathbf{r}} = \mathbf{v}

and

\dot{\mathbf{v}} = \mathbf{a}(\mathbf{r})

this becomes

\frac{\partial f}{\partial \mathbf{z}}
=
\begin{bmatrix}
\mathbf{0}_{3 \times 3}
&
\mathbf{I}_{3 \times 3}
\\
\dfrac{\partial \mathbf{a}}{\partial \mathbf{r}}
&
\mathbf{0}_{3 \times 3}
\end{bmatrix}

That tells you how state perturbations affect the vector field.

For a neural ODE learning 2BP, there are two common setups.

In the first setup, the neural network learns the full vector field:

f_\theta(\mathbf{z})
=
\begin{bmatrix}
\dot{\mathbf{r}}_\theta \\
\dot{\mathbf{v}}_\theta
\end{bmatrix}

In that case, the network outputs both predicted velocity-like components and predicted acceleration-like components. The model learns

\dot{x}, \dot{y}, \dot{z}, \dot{v}_x, \dot{v}_y, \dot{v}_z

directly.

In the second setup, which is usually more physically structured, you force the kinematics to be exact:

\dot{\mathbf{r}} = \mathbf{v}

and only ask the neural network to learn acceleration:

\dot{\mathbf{v}} = \mathbf{a}_\theta(\mathbf{r}, \mathbf{v}, t)

Then the learned vector field is

f_\theta(\mathbf{z})
=
\begin{bmatrix}
\mathbf{v} \\
\mathbf{a}_\theta(\mathbf{r}, \mathbf{v}, t)
\end{bmatrix}

In this case, the training data are still position and velocity, and the loss is still based on position and velocity error. But the neural network’s trainable part only affects the acceleration component of the vector field.

That means the adjoint still measures sensitivity with respect to the state:

\boldsymbol{\lambda}(t)
=
\begin{bmatrix}
\dfrac{\partial L}{\partial \mathbf{r}(t)} \\
\dfrac{\partial L}{\partial \mathbf{v}(t)}
\end{bmatrix}

but the parameter update comes through the acceleration model:

\frac{\partial f_\theta}{\partial \theta}
=
\begin{bmatrix}
\mathbf{0} \\
\dfrac{\partial \mathbf{a}_\theta}{\partial \theta}
\end{bmatrix}

because the top half,

\dot{\mathbf{r}} = \mathbf{v}

has no trainable parameters.

So in your specific case, where the training data are position and velocity and the neural ODE is learning 2BP dynamics, the clean interpretation is:

The loss is computed from errors in position and velocity.

The adjoint is the sensitivity of that loss to perturbations in position and velocity.

The vector field is what maps position and velocity to their time derivatives, namely velocity and acceleration.

The neural network may be learning the whole vector field, or only the acceleration part, depending on your architecture.

If your architecture enforces

\dot{\mathbf{r}} = \mathbf{v}

and learns only

\dot{\mathbf{v}} = \mathbf{a}_\theta

then the adjoint still lives in the 6D state space (\mathbf{r}, \mathbf{v}), but the network update mostly comes from how the velocity-adjoint component pushes on the learned acceleration.

Very intuitively:

\boldsymbol{\lambda}_{\mathbf{r}}

says: “How much does the loss care if the spacecraft position at this time changes?”

\boldsymbol{\lambda}_{\mathbf{v}}

says: “How much does the loss care if the spacecraft velocity at this time changes?”

The learned acceleration affects future velocity first, and future position second. So even though your loss is on position and velocity, the network learns acceleration because acceleration is the control knob inside the dynamics that shapes the future trajectory.