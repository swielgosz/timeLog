``` python
# cr3bp_effective_potential_anim.py
# ------------------------------------------------------------
# Generates:
#   - effective_potential_3d.png
#   - zvc_slices.gif
#   - combined.gif (3D rotation then forbidden-region fill)
#
# Notes:
#   * Requires matplotlib and pillow (pip install matplotlib pillow)
#   * Ω kept positive for logic; 3D renders ϕ = -Ω (clipped below L1).
#   * Contour levels sorted to satisfy Matplotlib's increasing-level requirement.
# ------------------------------------------------------------

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# User parameters
# -----------------------------
mu = 0.1  # Mass ratio (Earth–Moon ~0.0121505856)
xy_lim = 1.5  # Plot window: [-xy_lim, xy_lim] for both x & y
N_grid = 400  # Grid density for 2D potential
N_grid_3d = 120  # Grid density for 3D surface

# Animation settings
rot_frames = 60  # Frames for rotating the 3D surface
slice_frames = 45  # Frames for 2D forbidden-region fill
interval_ms = 120  # Frame interval for GIFs

# Output filenames
PNG_3D = "effective_potential_3d.png"
GIF_ZVC = "zvc_slices.gif"
GIF_COMBINED = "combined.gif"


# -----------------------------
# Effective potential Ω(x,y)
# Ω = 1/2 (x^2 + y^2) + (1-μ)/r1 + μ/r2
# Primaries at x=-μ and x=1-μ in nondimensional CR3BP units
# -----------------------------
def omega(x, y, mu):
    eps = 1e-12  # avoid singularities at primaries
    r1 = np.sqrt((x + mu) ** 2 + y**2) + eps
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2) + eps
    return 0.5 * (x**2 + y**2) + (1 - mu) / r1 + mu / r2


# dΩ/dx along y=0 (for locating collinear L-points)
def dOmega_dx_on_axis(x, mu):
    y = 0.0
    r1 = np.sqrt((x + mu) ** 2 + y**2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y**2)
    return x - (1 - mu) * (x + mu) / r1**3 - mu * (x - 1 + mu) / r2**3


# Simple bisection to find L1 between the primaries
def bisect(f, a, b, tol=1e-12, it=200):
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("Bisection interval does not bracket a root.")
    for _ in range(it):
        c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) < tol or 0.5 * (b - a) < tol:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5 * (a + b)


# -----------------------------
# Grids & Ω evaluation
# -----------------------------
x = np.linspace(-xy_lim, xy_lim, N_grid)
y = np.linspace(-xy_lim, xy_lim, N_grid)
X, Y = np.meshgrid(x, y)
Omega = omega(X, Y, mu)  # keep Ω positive for contours / C logic

x3d = np.linspace(-xy_lim, xy_lim, N_grid_3d)
y3d = np.linspace(-xy_lim, xy_lim, N_grid_3d)
X3d, Y3d = np.meshgrid(x3d, y3d)
Omega3d = omega(X3d, Y3d, mu)  # keep Ω positive for 3D, then flip sign for rendering
Phi3d = -Omega3d  # negative potential for 3D view only

# -----------------------------
# Choose a safe C range automatically
# C = 2Ω - v^2. ZVC (forbidden) is where 2Ω < C  ⇔  Ω < C/2
# We cap high percentiles to avoid singular terrain near primaries.
# -----------------------------
Omega_min = float(np.nanmin(Omega))
Omega_cap = float(np.nanpercentile(Omega, 99.8))  # cap near the spikes

C_min = 2.0 * Omega_min * 1.02  # a hair above the minimum
C_max = 2.0 * Omega_cap * 0.98  # a hair below the cap
C_levels = np.linspace(C_min, C_max, slice_frames)

# Primaries (for plotting)
x1, x2 = -mu, 1 - mu

# -----------------------------
# Locate L1 (for clipping the 3D negative potential just below ϕ(L1))
# -----------------------------
eps = 1e-6
L1x = bisect(lambda xx: dOmega_dx_on_axis(xx, mu), x1 + eps, x2 - eps)
Phi_L1 = -omega(L1x, 0.0, mu)  # ϕ(L1) = -Ω(L1)
clip_margin = 0.08 * abs(Phi_L1)  # tweak to taste
phi_floor = Phi_L1 - clip_margin
Phi3d_clipped = np.maximum(Phi3d, phi_floor)

# ============================================================
# (A) Static 3D surface snapshot (negative potential, clipped below L1)
# ============================================================
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

fig3d = plt.figure(figsize=(8, 6))
ax3d = fig3d.add_subplot(111, projection="3d")
surf = ax3d.plot_surface(
    X3d, Y3d, Phi3d_clipped, linewidth=0, antialiased=False, alpha=0.95
)
ax3d.set_title(
    "3D Negative Effective Potential ϕ(x, y) = −Ω(x, y)\n(clipped slightly below ϕ(L1))"
)
ax3d.set_xlabel("x")
ax3d.set_ylabel("y")
ax3d.set_zlabel("ϕ")
ax3d.set_zlim(phi_floor, np.max(Phi3d_clipped))
plt.tight_layout()
fig3d.savefig(PNG_3D, dpi=160)
plt.close(fig3d)

# ============================================================
# (B) GIF: forbidden regions (ZVC) filling in
# ============================================================
fig2d, ax2d = plt.subplots(figsize=(6, 6))
ax2d.set_xlim(-xy_lim, xy_lim)
ax2d.set_ylim(-xy_lim, xy_lim)
ax2d.set_aspect("equal")
ax2d.set_title("Forbidden Regions (Zero-Velocity Curves)")
ax2d.plot(x1, 0, "o", ms=8)  # Primary 1 marker
ax2d.plot(x2, 0, "o", ms=4)  # Primary 2 marker

filled = None  # will hold a QuadContourSet


def update_zvc(i):
    global filled
    if filled is not None:
        try:
            for c in getattr(filled, "collections", []):
                c.remove()
        except Exception:
            pass

    C = C_levels[i]
    # Forbidden region = where Ω < C/2
    level = 0.5 * C
    low = Omega_min - 1.0
    lo, hi = sorted([low, level])  # ensure increasing levels

    filled = ax2d.contourf(X, Y, Omega, levels=[lo, hi], alpha=0.55)
    ax2d.set_title(f"Forbidden Regions (2Ω < C)\nC = {C:.3f}")
    return []  # blit=False


ani_zvc = animation.FuncAnimation(
    fig2d,
    update_zvc,
    frames=len(C_levels),
    interval=interval_ms,
    blit=False,
)
ani_zvc.save(GIF_ZVC, writer="pillow", dpi=120)
plt.close(fig2d)

# ============================================================
# (C) Combined GIF: rotating 3D → 2D fill
# ============================================================
fig_comb = plt.figure(figsize=(10, 5))
ax_left = fig_comb.add_subplot(121, projection="3d")
ax_right = fig_comb.add_subplot(122)

# Left panel (3D, negative potential clipped)
ax_left.plot_surface(
    X3d, Y3d, Phi3d_clipped, linewidth=0, antialiased=False, alpha=0.95
)
ax_left.set_title("3D ϕ(x,y) (−Ω), clipped")
ax_left.set_xlabel("x")
ax_left.set_ylabel("y")
ax_left.set_zlabel("ϕ")
ax_left.set_zlim(phi_floor, np.max(Phi3d_clipped))

# Right panel (2D forbidden)
ax_right.set_xlim(-xy_lim, xy_lim)
ax_right.set_ylim(-xy_lim, xy_lim)
ax_right.set_aspect("equal")
ax_right.plot(x1, 0, "o", ms=8)
ax_right.plot(x2, 0, "o", ms=4)
ax_right.set_title("Forbidden Regions")

filled_right = None


def update_combined(frame):
    global filled_right

    if frame < rot_frames:
        # Rotate the 3D view
        azim = 30 + 360.0 * (frame / rot_frames)
        elev = 25 + 10.0 * np.sin(2 * np.pi * frame / rot_frames)
        ax_left.view_init(elev=elev, azim=azim)
        ax_right.set_title("Forbidden Regions (waiting…)")
        return []

    # ZVC fill phase
    i = frame - rot_frames
    if filled_right is not None:
        try:
            for c in getattr(filled_right, "collections", []):
                c.remove()
        except Exception:
            pass

    C = C_levels[min(i, slice_frames - 1)]
    level = 0.5 * C
    low = Omega_min - 1.0
    lo, hi = sorted([low, level])  # ensure increasing levels

    filled_right = ax_right.contourf(X, Y, Omega, levels=[lo, hi], alpha=0.55)
    ax_right.set_title(f"Forbidden Regions (2Ω < C)\nC = {C:.3f}")
    return []


ani_comb = animation.FuncAnimation(
    fig_comb,
    update_combined,
    frames=rot_frames + slice_frames,
    interval=interval_ms,
    blit=False,
)
ani_comb.save(GIF_COMBINED, writer="pillow", dpi=120)
plt.close(fig_comb)

print(f"Saved:\n - {PNG_3D}\n - {GIF_ZVC}\n - {GIF_COMBINED}")
print(f"L1 ≈ ({L1x:.6f}, 0),  ϕ(L1) = {Phi_L1:.6f}")

```