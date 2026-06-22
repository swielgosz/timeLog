``` python
"""
Plot first 4 orbits from the training dataset with markers at each timestep.
Run from the repo root:
    python scripts/training/plot_timestep_spacing.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import matplotlib.pyplot as plt
from mldsml.wandb_utils import load_dataset
from neuralODE import constants
from neuralODE.normalization import Normalization2BP

DATASET = "complex_TBP_planar_100_train"
N_ORBITS = 4
OUT_FULL = "scripts/training/files/results/timestep_spacing_full.png"
OUT_ZOOM = "scripts/training/files/results/timestep_spacing_zoom.png"

raw = load_dataset(DATASET, version="latest", project="neuralODEs", entity="mlds-lab")
keys = sorted(raw.keys())[:N_ORBITS]

transform = Normalization2BP(l_char=constants.RADIUS_EARTH, mu=constants.MU_EARTH)
raw_norm = transform.normalize_dataset(raw)

BG = "#1a1a2e"
AX_BG = "#16213e"
GRID = "#2a2a4a"
TC = "white"

def style_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors=TC)
    ax.xaxis.label.set_color(TC)
    ax.yaxis.label.set_color(TC)
    ax.title.set_color(TC)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.5)

os.makedirs("scripts/training/files/results", exist_ok=True)

# ── Figure 1: full orbits ──────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 14), constrained_layout=True)
fig1.patch.set_facecolor(BG)
fig1.suptitle(f"Timestep spacing — {DATASET} (full orbits)", color=TC, fontsize=13)

for ax, key in zip(axes1.ravel(), keys):
    entry = raw_norm[key]
    states = np.array(entry["y"]).T
    t = np.array(entry["t"])
    x, y = states[:, 0], states[:, 1]
    e = entry.get("metadata", {}).get("eccentricity", "?")
    style_ax(ax)
    ax.plot(x, y, color="white", lw=0.5, alpha=0.3)
    sc = ax.scatter(x, y, c=t, cmap="plasma", s=12, zorder=3)
    ax.plot(0, 0, "o", color="goldenrod", ms=8, zorder=5)
    plt.colorbar(sc, ax=ax, label="time (norm)").ax.yaxis.set_tick_params(colors=TC)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (norm)")
    ax.set_ylabel("y (norm)")
    ax.set_title(f"{key}  e={float(e):.3f}", fontsize=10)

fig1.savefig(OUT_FULL, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig1)
print(f"saved: {OUT_FULL}")

# ── Figure 2: zoom panels, periapsis vs apoapsis, one row per orbit ────────────
fig2, axes2 = plt.subplots(N_ORBITS, 2, figsize=(16, N_ORBITS * 5), constrained_layout=True)
fig2.patch.set_facecolor(BG)
fig2.suptitle(f"Timestep spacing — periapsis (cyan) vs apoapsis (orange)", color=TC, fontsize=13)

zoom_half = 15

for row, key in enumerate(keys):
    entry = raw_norm[key]
    states = np.array(entry["y"]).T
    t = np.array(entry["t"])
    x, y = states[:, 0], states[:, 1]
    r = np.sqrt(x**2 + y**2)
    e = entry.get("metadata", {}).get("eccentricity", "?")
    i_peri = int(np.argmin(r))
    i_apo = int(np.argmax(r))

    for col, (center, region_label, color) in enumerate([
        (i_peri, "periapsis", "cyan"),
        (i_apo, "apoapsis", "orange"),
    ]):
        ax = axes2[row, col]
        style_ax(ax)
        idx = np.arange(center - zoom_half, center + zoom_half + 1) % len(x)
        ax.plot(x[idx], y[idx], "-", color=color, lw=1.2, alpha=0.4)
        ax.scatter(x[idx], y[idx], color=color, s=14, zorder=4)
        for ii in idx:
            ax.annotate(
                str(ii), (x[ii], y[ii]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=6, color=color, alpha=0.9,
            )
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x (norm)")
        ax.set_ylabel("y (norm)")
        ax.set_title(f"{key}  e={float(e):.3f} — {region_label}", fontsize=10)

fig2.savefig(OUT_ZOOM, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig2)
print(f"saved: {OUT_ZOOM}")
```