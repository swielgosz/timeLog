Yes — the confusion is exactly about which “future” we mean. I should have been more explicit.

When I say “future loss points,” I mean future in the original physical/forward time direction of the trajectory, not future in the reverse adjoint integration direction.

Let the forward trajectory times be

t_0 < t_1 < t_2 < \cdots < t_N.

The neural ODE forward solve goes this way:

t_0 \rightarrow t_1 \rightarrow t_2 \rightarrow \cdots \rightarrow t_N.

The adjoint solve goes the opposite way:

t_N \rightarrow t_{N-1} \rightarrow t_{N-2} \rightarrow \cdots \rightarrow t_0.

So if we are computing the adjoint at some middle time, say t_j, then “future points” means points with larger forward time:

t_{j+1}, t_{j+2}, \dots, t_N.

Those are physically future relative to t_j. But during the backward adjoint solve, those points have already been visited, because we started from t_N and are moving backward.

So the slightly confusing but correct statement is:

At time t_j, the adjoint contains sensitivity from loss terms at t_j and at forward-future times t_{j+1}, \dots, t_N. During the reverse integration, those future-in-forward-time loss terms were encountered earlier in the backward computation.

It may help to use a tiny discrete example. Suppose the forward trajectory is

\mathbf{z}_0
\rightarrow
\mathbf{z}_1
\rightarrow
\mathbf{z}_2
\rightarrow
\mathbf{z}_3.

Now suppose the full-trajectory loss is

L =
\ell_1(\mathbf{z}_1)
+
\ell_2(\mathbf{z}_2)
+
\ell_3(\mathbf{z}_3).

The loss has a contribution at each saved state. The adjoint at the final state \mathbf{z}_3 only needs to account for the loss at \mathbf{z}_3, because \mathbf{z}_3 does not affect any later states.

The adjoint at \mathbf{z}_2 needs to account for two things: the direct loss at \mathbf{z}_2, and the fact that \mathbf{z}_2 affects \mathbf{z}_3, which affects the loss at \mathbf{z}_3.

The adjoint at \mathbf{z}_1 needs to account for the direct loss at \mathbf{z}_1, plus the fact that \mathbf{z}_1 affects \mathbf{z}_2, which affects \ell_2, and also affects \mathbf{z}_3, which affects \ell_3.

So, in forward-time interpretation:

\lambda(t_3)

contains sensitivity from the loss at t_3.

\lambda(t_2)

contains sensitivity from the loss at t_2 and t_3.

\lambda(t_1)

contains sensitivity from the loss at t_1, t_2, and t_3.

That is what I mean by accumulation from future points. At an earlier physical time, the state can influence more of the remaining trajectory, so the adjoint can contain sensitivity from more future loss terms.

But during the actual backward computation, the order is reversed. The algorithm starts at t_3, then goes to t_2, then t_1. So by the time it reaches t_2, it has already carried backward the sensitivity from t_3. Then it adds the local loss contribution from t_2. By the time it reaches t_1, it has already carried backward the sensitivity from t_3 and t_2, then it adds the local contribution from t_1.

So the clean rule is:

At a given time, the adjoint tells you how the total remaining forward-time loss depends on the state at that time.

“Remaining” means the loss at that time and after that time, in the original forward trajectory.

It does not include loss contributions from earlier physical times, because the state at t_j cannot affect the state at t_{j-1} in the original forward simulation. Causality only goes forward.

So if you are standing at t_j in the original orbit, the adjoint asks:

“If I perturb the state here, how will that change the loss from now until the end of the rollout?”

For terminal loss, “from now until the end” only contains the final loss. For full-trajectory loss, “from now until the end” contains all the trajectory loss terms from t_j through t_N.

This is why a full-trajectory adjoint can have a strong time-accumulation flavor. Earlier states have more forward-time loss terms downstream of them. Later states have fewer downstream loss terms. That does not guarantee the adjoint is larger earlier, because the dynamics can amplify, rotate, or damp sensitivities, but it does explain why the adjoint can look more like a global time-dependent accumulation than a purely local periapsis diagnostic.

So to answer your specific question: when I said the adjoint accumulates contribution from future points, I meant future in the original forward integration. In the backward adjoint integration, those same points are encountered earlier, because the computation is running in reverse.