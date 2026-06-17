Why ||a(t)|| is large at t_0 and small at t_T
The boundary condition at t_T is:


a(T) = ∂L/∂z(T) = (z(T) - y_true(T)) / ||y_true(T)||
This is a unit-scale vector — its norm is O(1) (the percent error, divided out). So ||a(T)|| starts small.

As you integrate backward in time, you're propagating sensitivity: a(t) answers "how much does the loss change if I perturb z(t)?" A perturbation at t_0 has the entire remaining trajectory to amplify through — it gets integrated forward through the full dynamics before reaching the terminal loss. So early-time perturbations have larger influence on the terminal loss, meaning ||a(t_0)|| should be large.

This is the correct physics for a chaotic/unstable orbit. The adjoint grows backward in time at a rate governed by the Lyapunov exponents of the system (or specifically the eigenvalues of ∂f/∂z along the trajectory). For elliptical orbits with eccentricity, you'd expect the adjoint to grow most strongly near periapsis where the dynamics are fastest.

So is the code correct?
The code is correct. The adjoint growing as you go backward from t_T to t_0 means gradient signal is strongest at the beginning of each training window — perturbations there propagate forward and accumulate error. This is the opposite of the vanishing gradient problem: it's actually gradient amplification backward in time, which is healthy.

What would be a problem is if the adjoint were decaying as you integrate backward (large at t_T, small at t_0) — that would mean the model has learned dynamics where early perturbations don't affect the terminal state, which would starve the gradient. The divergence row in your plots diagnoses exactly that risk



Understodd. However, we still have the issue that since our adjoint is small near the end, divergence near the end of the trajectory ends up unchecked. do you agree?
Yes, exactly right. The gradient signal near t_T is weak (small ||a(T)||), so if the model develops bad divergence behavior near the end of a segment, the loss gradient barely penalizes it. The model can freely "cheat" there — e.g. contracting phase space at the terminal end — without paying much cost in the loss.



This is compounded by the fact that your loss is a terminal loss (endpoint error only, not integrated over the trajectory). A midpoint divergence that happens to cancel out by t_T goes completely unpunished.

The practical implication: the divergence row of your plots being non-zero near periapsis at late times in the segment is a sign the model is exploiting this blind spot. Adding a physics regularizer on tr(∂f/∂z) (penalizing deviation from zero divergence) would be the direct fix — it enforces the Hamiltonian constraint independently of what the adjoint weights.

