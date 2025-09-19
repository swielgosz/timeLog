``` python
# Planar L1 Lyapunov Orbit in the CR3BP with Diffrax (JAX)

**Goal.** Find a planar Lyapunov orbit about L1 in the Earth–Moon CR3BP using
- variational equations → STM,
- **single shooting** (half-period symmetry), and
- **multiple shooting** (full period, m arcs).

**Key ideas.**
- The STM gives the Jacobian of the flow map: \( \Phi(t) = \partial x(t)/\partial x_0 \).
- For a planar Lyapunov orbit, we exploit a **half-period symmetry**:
  start at \(y_0=0, \dot x_0=0, z_0=\dot z_0 = 0\). At half period \(\tau = T/2\),
  the symmetric conditions are \(y(\tau)=0\) and \(\dot x(\tau)=0\).
- Newton updates come from linearizing with \(\Phi(\tau)\) and \(\partial x(\tau)/\partial \tau = f(x(\tau))\).

**Notebook contents**
1. CR3BP dynamics & effective potential \(\Omega\)
2. Libration points finder (L1–L5)
3. Linear Lyapunov initial guess near L1/L2/L3 (Hessian-based)
4. Variational system (state + STM) and Diffrax integrator
5. Single shooting (half-period)
6. Multiple shooting (full period)
7. Plotting the final orbit```