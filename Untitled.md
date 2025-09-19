``` python
# --- Minimal CR3BP Lyapunov orbit finder with JAX + diffrax (ODETerm) ---
# Single-shooting with STM; half-period symmetry conditions.

import jax
import jax.numpy as jnp
from jax import jit
from functools import partial
import diffrax as dfx

# -----------------------------
# Problem constants (Earth-Moon)
# -----------------------------
mu = 0.0121505856  # Earth-Moon mass parameter (nondimensional)

# -----------------------------
# CR3BP Dynamics (rotating frame)
# state y = [x, y, z, vx, vy, vz]
# -----------------------------
def cr3bp_accel(y, mu):
    x, y_, z = y[0], y[1], y[2]
    mu1 = 1.0 - mu
    r1 = jnp.sqrt((x + mu)**2 + y_**2 + z**2)
    r2 = jnp.sqrt((x - mu1)**2 + y_**2 + z**2)
    Ux = x - mu1*(x + mu)/r1**3 - mu*(x - mu1)/r2**3
    Uy = y_ - mu1*y_/r1**3 - mu*y_/r2**3
    Uz =     - mu1*z /r1**3 - mu*z     /r2**3
    ax = 2.0*y[4] + Ux
    ay = -2.0*y[3] + Uy
    az = Uz
    return jnp.array([ax, ay, az])

def f_state(t, y, mu):
    # y: (6,) -> dy/dt
    return jnp.array([y[3], y[4], y[5], *cr3bp_accel(y, mu)])

# -----------------------------
# Variational dynamics (STM)
# We integrate [state(6); Phi(36)] together.
# dPhi/dt = A(t) * Phi, where A = df/dy
# -----------------------------
def A_matrix(y, mu):
    # Jacobian of f_state w.r.t. y
    # We use jacfwd on a wrapper that only depends on y.
    return jax.jacfwd(lambda yy: f_state(0.0, yy, mu))(y)

def f_aug(t, yaug, mu):
    # Split state and Phi
    y = yaug[:6]
    Phi = yaug[6:].reshape(6, 6)
    # Dynamics
    dy = f_state(t, y, mu)
    A = A_matrix(y, mu)
    dPhi = A @ Phi
    return jnp.concatenate([dy, dPhi.reshape(-1)])

# Wrap in diffrax ODETerm
term = dfx.ODETerm(lambda t, y, args: f_aug(t, y, args))

# -----------------------------
# Integrator
# -----------------------------
def propagate_with_stm(y0, tf, mu):
    Phi0 = jnp.eye(6).reshape(-1)
    y0_aug = jnp.concatenate([y0, Phi0])
    solver = dfx.Tsit5()
    # Simple, stable controller (can tweak rtol/atol as needed)
    stepsize_controller = dfx.PIDController(rtol=1e-10, atol=1e-12)
    sol = dfx.diffeqsolve(
        term, solver, t0=0.0, t1=tf, dt0=None,
        y0=y0_aug, args=mu,
        saveat=dfx.SaveAt(t1=True),
        stepsize_controller=stepsize_controller,

    )
    ytf_aug = sol.ys
    ytf = ytf_aug[:6]
    Phitf = ytf_aug[6:].reshape(6, 6)
    f_tf = f_state(tf, ytf, mu)
    return ytf, Phitf, f_tf

# -----------------------------
# L1 abscissa (1D Newton) to get a decent xL1
# Solve the collinear equilibrium equation along x-axis, y=z=0, vx=vy=vz=0
# Equation: dOmega/dx = 0 -> x - (1-mu)(x+mu)/|x+mu|^3 - mu(x-1+mu)/|x-1+mu|^3 = 0
# -----------------------------
def dOmega_dx_on_x_axis(x, mu):
    mu1 = 1.0 - mu
    r1 = jnp.abs(x + mu)
    r2 = jnp.abs(x - mu1)
    return x - mu1*(x + mu)/r1**3 - mu*(x - mu1)/r2**3

def d2Omega_dx2_on_x_axis(x, mu):
    mu1 = 1.0 - mu
    r1 = jnp.abs(x + mu)
    r2 = jnp.abs(x - mu1)
    # derivative of (x +/- a)/|x +/- a|^3 = (x +/- a) * |x +/- a|^{-3}
    # d/dx = |x+a|^{-3} - 3(x+a)^2 |x+a|^{-5} = (1 - 3)*|x+a|^{-3} with sign via (x+a)^2
    # Careful with abs; for x near L1 between primaries, r1,r2>0 so it's fine:
    term1 = 1.0 - 3.0*(x + mu)**2 / r1**2
    term2 = 1.0 - 3.0*(x - mu1)**2 / r2**2
    return 1.0 - mu1 * (term1) / r1**3 - mu * (term2) / r2**3

def find_xL1(mu, x_init=0.8, iters=20):
    x = x_init
    for _ in range(iters):
        g = dOmega_dx_on_x_axis(x, mu)
        gp = d2Omega_dx2_on_x_axis(x, mu)
        x = x - g / gp
    return x

# -----------------------------
# Newton correction on [x0, vy0, tf]
# Residuals at t = tf:
#   R1 = y(tf)         -> enforce symmetry crossing
#   R2 = vx(tf)        -> enforce symmetry condition
#   R3 = x(tf) - x0    -> enforce half-period mirror
# Jacobian columns:
#   d/d[x0]  : [Phi[1,0], Phi[3,0], Phi[0,0] - 1]
#   d/d[vy0] : [Phi[1,4], Phi[3,4], Phi[0,4]]
#   d/d[tf]  : [ f_y(tf),  f_vx(tf), f_x(tf) ]
# -----------------------------
def correct_lyapunov(mu, x0, vy0, tf, max_iters=15, tol=1e-11):
    y0 = jnp.array([x0, 0.0, 0.0, 0.0, vy0, 0.0])
    for k in range(max_iters):
        ytf, Phitf, f_tf = propagate_with_stm(y0, tf, mu)

        R = jnp.array([
            ytf[1],         # y(tf)
            ytf[3],         # vx(tf)
            ytf[0] - y0[0]  # x(tf) - x0
        ])

        normR = jnp.linalg.norm(R, ord=jnp.inf)
        # print(f"iter {k:02d} |R|_inf = {normR:.3e}")
        if normR < tol:
            break

        J = jnp.column_stack([
            jnp.array([Phitf[1,0], Phitf[3,0], Phitf[0,0] - 1.0]),   # dR/dx0
            jnp.array([Phitf[1,4], Phitf[3,4], Phitf[0,4]]),         # dR/dvy0
            jnp.array([f_tf[1],    f_tf[3],    f_tf[0]])             # dR/dtf
        ])

        delta = jnp.linalg.solve(J, -R)

        # Update variables
        x0  = x0  + float(delta[0])
        vy0 = vy0 + float(delta[1])
        tf  = tf  + float(delta[2])
        y0 = y0.at[0].set(x0)
        y0 = y0.at[4].set(vy0)

    return x0, vy0, tf, y0, ytf

# -----------------------------
# Run it
# -----------------------------
if __name__ == "__main__":
    # Initial guess from linear approx: start slightly displaced from L1 on x-axis, add small vy0
    xL1 = float(find_xL1(mu))          # ~0.8369 for Earth-Moon
    A = 1.5e-3                         # small amplitude in x
    x0_guess  = xL1 - A
    vy0_guess = 0.06                   # modest guess; tune if needed
    tf_guess  = 3.0                    # half-period guess (nondimensional time)

    x0, vy0, tf, y0, ytf = correct_lyapunov(mu, x0_guess, vy0_guess, tf_guess)

    print("---- Lyapunov Orbit (half-period) ----")
    print(f"x0    = {x0:.12f}")
    print(f"vy0   = {vy0:.12f}")
    print(f"tf    = {tf:.12f}")
    print("Residuals at tf (should be ~0): y(tf), vx(tf), x(tf)-x0")
    print(ytf[1], ytf[3], ytf[0] - x0)

    # Optional: integrate full period by mirroring (just rerun with 2*tf and y0)
    # Or sample trajectory for plotting (dense SaveAt or reintegrate with saveat=t_eval).```