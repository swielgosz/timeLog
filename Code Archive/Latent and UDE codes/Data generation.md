# generate_2BP_const_drag_data.py
``` python
"""
Generate datasets for 2BP dynamics with constant-density quadratic drag.

Format matches sampling_2BP.generate_datasets: dict of trajectories with keys
{t, y, metadata}, where y has shape (num_states, num_points).
"""

import os

import numpy as np
from mldsml.wandb_utils import save_dataset
from scipy.integrate import solve_ivp

from neuralODE import astro_utils, constants, coordinate_conversions


def _two_body_accel(r, mu):
    r_norm = np.linalg.norm(r)
    return -mu * r / r_norm**3


def _drag_accel_km_s2(r, v, *, rho_kg_m3, cd_area_over_m, omega_earth):
    rho_kg_km3 = rho_kg_m3 * 1e9
    omega_vec = np.array([0.0, 0.0, omega_earth])
    v_rel = v - np.cross(omega_vec, r)
    v_rel_norm = np.linalg.norm(v_rel)
    if v_rel_norm == 0.0:
        return np.zeros(3)
    return -0.5 * cd_area_over_m * rho_kg_km3 * v_rel_norm * v_rel


def _two_body_rhs(t, y, mu):
    r = y[:3]
    v = y[3:]
    a_grav = _two_body_accel(r, mu)
    return np.array([v[0], v[1], v[2], a_grav[0], a_grav[1], a_grav[2]])


def _drag_rhs(t, y, params):
    r = y[:3]
    v = y[3:]
    a_grav = _two_body_accel(r, params["mu"])
    a_drag = _drag_accel_km_s2(
        r,
        v,
        rho_kg_m3=params["rho_kg_m3"],
        cd_area_over_m=params["cd_area_over_m"],
        omega_earth=params["omega_earth"],
    )
    a_total = a_grav + a_drag
    return np.array([v[0], v[1], v[2], a_total[0], a_total[1], a_total[2]])


def _integrate_orbit(oe, num_points, params, *, include_drag):
    a, e, inc, raan, w, nu0 = oe
    period = astro_utils.calculate_orbital_period(a, params["mu"])
    times = np.linspace(
        0.0,
        params["num_periods"] * period,
        num_points,
        endpoint=False,
    )
    r0, v0 = coordinate_conversions.standard_to_cartesian(
        a,
        e,
        inc,
        raan,
        w,
        nu0,
        mu=params["mu"],
    )
    y0 = np.hstack([r0, v0])

    if include_drag:
        rhs = lambda t, y: _drag_rhs(t, y, params)
    else:
        rhs = lambda t, y: _two_body_rhs(t, y, params["mu"])

    sol = solve_ivp(
        rhs,
        (times[0], times[-1]),
        y0=y0,
        t_eval=times,
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y


def generate_2bp_const_drag_dataset(params, name):
    data_dict = {}
    rng = np.random.default_rng(params["seed"])

    for _ in range(params["num_trajs"]):
        a = rng.uniform(*params["a_range"])
        e = rng.uniform(*params["e_range"])
        inc = rng.uniform(*params["i_range"])
        raan = rng.uniform(*params["raan_range"])
        w = rng.uniform(*params["w_range"])
        nu0 = rng.uniform(*params["nu_range"])

        periapsis = a * (1 - e)
        if periapsis < params["body_radius"]:
            continue

        oe = np.array([a, e, inc, raan, w, nu0])
        t_eval, y_eval = _integrate_orbit(
            oe,
            params["num_points"],
            params,
            include_drag=True,
        )

        traj_key = f"orbit_{len(data_dict) + 1}"
        data_dict[traj_key] = {
            "t": t_eval.tolist(),
            "y": y_eval.tolist(),
            "metadata": {
                "sma": a,
                "eccentricity": e,
                "inclination": inc,
                "raan": raan,
                "argument_of_periapsis": w,
                "initial_true_anomaly": nu0,
                "num_points": params["num_points"],
                "num_periods": params["num_periods"],
                "drag": {
                    "rho_kg_m3": params["rho_kg_m3"],
                    "cd_area_over_m": params["cd_area_over_m"],
                    "omega_earth": params["omega_earth"],
                },
            },
        }

    save_dataset(
        name,
        data_dict,
        project="neuralODEs",
        entity="mlds-lab",
    )
    return data_dict


def plot_orbit_diagnostics(traj, params, out_path):
    import matplotlib.pyplot as plt

    t = np.array(traj["t"])
    y = np.array(traj["y"])
    r = y[:3]
    v = y[3:]

    oe = np.array(
        [
            traj["metadata"]["sma"],
            traj["metadata"]["eccentricity"],
            traj["metadata"]["inclination"],
            traj["metadata"]["raan"],
            traj["metadata"]["argument_of_periapsis"],
            traj["metadata"]["initial_true_anomaly"],
        ],
    )
    base_t, base_y = _integrate_orbit(
        oe,
        params["num_points"],
        params,
        include_drag=False,
    )
    base_r = base_y[:3]
    base_v = base_y[3:]

    mu = params["mu"]
    r_norm = np.linalg.norm(r, axis=0)
    base_accel = -mu * r / r_norm**3
    drag_accel = np.array(
        [
            _drag_accel_km_s2(
                r[:, idx],
                v[:, idx],
                rho_kg_m3=params["rho_kg_m3"],
                cd_area_over_m=params["cd_area_over_m"],
                omega_earth=params["omega_earth"],
            )
            for idx in range(r.shape[1])
        ],
    ).T
    total_accel = base_accel + drag_accel

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t, base_r[0], label="x (base)")
    axes[0, 0].plot(t, base_r[1], label="y (base)")
    axes[0, 0].plot(t, r[0], ":", label="x (drag)")
    axes[0, 0].plot(t, r[1], ":", label="y (drag)")
    axes[0, 0].set_title("Position states (planar)")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("km")
    axes[0, 0].legend()

    axes[0, 1].plot(t, base_v[0], label="vx (base)")
    axes[0, 1].plot(t, base_v[1], label="vy (base)")
    axes[0, 1].plot(t, v[0], ":", label="vx (drag)")
    axes[0, 1].plot(t, v[1], ":", label="vy (drag)")
    axes[0, 1].set_title("Velocity states (planar)")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("km/s")
    axes[0, 1].legend()

    base_mag = np.linalg.norm(base_accel, axis=0)
    drag_mag = np.linalg.norm(drag_accel, axis=0)
    total_mag = np.linalg.norm(total_accel, axis=0)
    axes[1, 0].plot(t, base_mag, label="base |a|")
    axes[1, 0].plot(t, drag_mag, label="drag |a|")
    axes[1, 0].plot(t, total_mag, ":", label="total |a|")
    axes[1, 0].plot(
        t,
        np.linalg.norm(total_accel - base_accel, axis=0),
        label="|total - base|",
    )
    ratio = total_mag / base_mag
    ratio_ax = axes[1, 0].twinx()
    ratio_ax.plot(t, ratio, "--", color="gray", label="total/base")
    ratio_ax.set_ylabel("ratio")
    axes[1, 0].set_title("Acceleration magnitudes")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("km/s^2")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend()

    axes[1, 1].plot(base_r[0], base_r[1], "--", label="2BP base")
    axes[1, 1].plot(r[0], r[1], label="2BP + drag")
    axes[1, 1].set_title("XY trajectory")
    axes[1, 1].set_xlabel("x (km)")
    axes[1, 1].set_ylabel("y (km)")
    axes[1, 1].axis("equal")
    axes[1, 1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    area_m2 = 10.0
    mass_kg = 1000.0
    params = {
        "num_trajs": 1,
        "num_points": 360,
        "num_periods": 5,
        "seed": 7,
        "a_range": [constants.RADIUS_EARTH + 500, constants.RADIUS_EARTH + 500],
        "e_range": [0.02, 0.02],
        "i_range": np.radians([0, 0]),
        "raan_range": np.radians([0, 180]),
        "w_range": np.radians([0, 180]),
        "nu_range": np.radians([0, 360]),
        "body_radius": constants.RADIUS_EARTH,
        "mu": constants.MU_EARTH,
        "omega_earth": 0.0,
        "rho_kg_m3": 2.418e-11,
        "cd_area_over_m": 50
        * 2.2
        * area_m2
        * 1e-6
        / mass_kg,  # HACK: artificially increase drag to see effects
    }

    dataset_name = "simple_2BP_const_drag_example"
    data = generate_2bp_const_drag_dataset(params, dataset_name)
    traj = data[list(data.keys())[0]]
    plot_path = "scripts/data_gen/plots/2bp_const_drag_example.png"
    plot_orbit_diagnostics(traj, params, plot_path)
    print(f"Saved plot to {plot_path}")
```

# generate_2BP_drag_data.py
``` python
"""
Generate datasets for 2BP dynamics with atmospheric drag perturbation.

INTERNAL UNITS (SI):
- r: meters
- v: m/s
- t: seconds
- a: m/s^2
- mu: m^3/s^2
- rho: kg/m^3

DATASET OUTPUT:
- r saved in km
- v saved in km/s
"""

import os

import numpy as np
from mldsml.wandb_utils import save_dataset
from scipy.integrate import solve_ivp
from tqdm import tqdm

from neuralODE import astro_utils, constants, coordinate_conversions

# ----------------------------
# Physics models (SI units)
# ----------------------------


def _two_body_accel_si(r_m, mu_m3_s2):
    r_norm = np.linalg.norm(r_m)
    return -mu_m3_s2 * r_m / r_norm**3


def _atmospheric_density_si(h_m, rho_ref, h_ref_m, scale_height_m):
    h_m = np.maximum(h_m, 0.0)
    return rho_ref * np.exp(-(h_m - h_ref_m) / scale_height_m)


def _drag_accel_si(
    r_m,
    v_m_s,
    *,
    omega_earth,
    cd_area_over_m,
    rho_ref,
    h_ref_m,
    scale_height_m,
    body_radius_m,
):
    r_norm = np.linalg.norm(r_m)
    h_m = r_norm - body_radius_m

    rho = _atmospheric_density_si(h_m, rho_ref, h_ref_m, scale_height_m)

    omega_vec = np.array([0.0, 0.0, omega_earth])
    v_rel = v_m_s - np.cross(omega_vec, r_m)
    v_rel_norm = np.linalg.norm(v_rel)

    if v_rel_norm == 0.0:
        return np.zeros(3)

    return -0.5 * cd_area_over_m * rho * v_rel_norm * v_rel


def _rhs_2bp_si(t, y, mu):
    r = y[:3]
    v = y[3:]
    a = _two_body_accel_si(r, mu)
    return np.hstack([v, a])


def _rhs_2bp_drag_si(t, y, params):
    r = y[:3]
    v = y[3:]

    a_grav = _two_body_accel_si(r, params["mu"])
    a_drag = _drag_accel_si(
        r,
        v,
        omega_earth=params["omega_earth"],
        cd_area_over_m=params["cd_area_over_m"],
        rho_ref=params["rho_ref"],
        h_ref_m=params["h_ref_m"],
        scale_height_m=params["scale_height_m"],
        body_radius_m=params["body_radius_m"],
    )

    return np.hstack([v, a_grav + a_drag])


# ----------------------------
# Orbit integration
# ----------------------------


def _integrate_orbit(oe, num_points, params, *, include_drag):
    a_km, e, inc, raan, w, nu0 = oe

    # Period from Kepler (SI)
    a_m = a_km * 1e3
    period = astro_utils.calculate_orbital_period(a_km, params["mu"] / 1e9)
    num_periods = int(params["num_periods"])
    times = np.linspace(0.0, num_periods * period, num_points, endpoint=False)

    # Initial state (convert to SI)
    r0_km, v0_km_s = coordinate_conversions.standard_to_cartesian(
        a_km,
        e,
        inc,
        raan,
        w,
        nu0,
        mu=params["mu"] / 1e9,
    )
    r0 = r0_km * 1e3
    v0 = v0_km_s * 1e3
    y0 = np.hstack([r0, v0])

    rhs = (
        (lambda t, y: _rhs_2bp_drag_si(t, y, params))
        if include_drag
        else (lambda t, y: _rhs_2bp_si(t, y, params["mu"]))
    )

    t_out = []
    y_out = []
    y_curr = y0
    progress = tqdm(range(num_periods), desc="Integrating periods", leave=False)
    for k in progress:
        t_start = k * period
        t_end = (k + 1) * period
        mask = (times >= t_start) & (times < t_end)
        t_eval = times[mask]
        if t_eval.size == 0:
            t_eval = np.array([t_end])
        elif t_eval[-1] != t_end:
            t_eval = np.concatenate([t_eval, [t_end]])

        sol = solve_ivp(
            rhs,
            (t_start, t_end),
            y_curr,
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        t_chunk = sol.t
        y_chunk = sol.y
        if t_chunk.size > 0 and np.isclose(t_chunk[-1], t_end):
            t_chunk = t_chunk[:-1]
            y_chunk = y_chunk[:, :-1]
        if t_chunk.size > 0:
            t_out.append(t_chunk)
            y_out.append(y_chunk)
        y_curr = sol.y[:, -1]

    if t_out:
        t_all = np.concatenate(t_out)
        y_all = np.concatenate(y_out, axis=1)
    else:
        t_all = times
        y_all = np.zeros((6, times.size))
    return t_all, y_all


# ----------------------------
# Plotting
# ----------------------------


def plot_orbit_diagnostics(traj, params, out_path):
    import matplotlib.pyplot as plt

    t = np.array(traj["t"])
    y = np.array(traj["y"])
    r = y[:3]
    v = y[3:]

    oe = np.array(
        [
            traj["metadata"]["sma_km"],
            traj["metadata"]["eccentricity"],
            traj["metadata"]["inclination"],
            traj["metadata"]["raan"],
            traj["metadata"]["argument_of_periapsis"],
            traj["metadata"]["initial_true_anomaly"],
        ],
    )
    base_t, base_y_si = _integrate_orbit(
        oe,
        params["num_points"],
        params,
        include_drag=False,
    )
    base_y = base_y_si.copy()
    base_y[:3] /= 1e3
    base_y[3:] /= 1e3
    base_r = base_y[:3]
    base_v = base_y[3:]

    mu_km = params["mu"] / 1e9
    r_norm = np.linalg.norm(r, axis=0)
    base_accel = -mu_km * r / r_norm**3
    drag_accel = np.array(
        [
            _drag_accel_si(
                (r[:, idx] * 1e3),
                (v[:, idx] * 1e3),
                omega_earth=params["omega_earth"],
                cd_area_over_m=params["cd_area_over_m"],
                rho_ref=params["rho_ref"],
                h_ref_m=params["h_ref_m"],
                scale_height_m=params["scale_height_m"],
                body_radius_m=params["body_radius_m"],
            )
            for idx in range(r.shape[1])
        ],
    ).T
    drag_accel = drag_accel / 1e3
    total_accel = base_accel + drag_accel

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t, base_r[0], label="x (base)")
    axes[0, 0].plot(t, base_r[1], label="y (base)")
    axes[0, 0].plot(t, r[0], ":", label="x (drag)")
    axes[0, 0].plot(t, r[1], ":", label="y (drag)")
    axes[0, 0].set_title("Position states (planar)")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("km")
    axes[0, 0].legend()

    axes[0, 1].plot(t, base_v[0], label="vx (base)")
    axes[0, 1].plot(t, base_v[1], label="vy (base)")
    axes[0, 1].plot(t, v[0], ":", label="vx (drag)")
    axes[0, 1].plot(t, v[1], ":", label="vy (drag)")
    axes[0, 1].set_title("Velocity states (planar)")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("km/s")
    axes[0, 1].legend()

    base_mag = np.linalg.norm(base_accel, axis=0)
    drag_mag = np.linalg.norm(drag_accel, axis=0)
    total_mag = np.linalg.norm(total_accel, axis=0)
    axes[1, 0].plot(t, base_mag, label="base |a|")
    axes[1, 0].plot(t, drag_mag, label="drag |a|")
    axes[1, 0].plot(t, total_mag, ":", label="total |a|")
    # axes[1, 0].plot(
    #     t,
    #     np.linalg.norm(total_accel - base_accel, axis=0),
    #     label="|total - base|",
    # )
    axes[1, 0].set_title("Acceleration magnitudes")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("km/s^2")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend()

    axes[1, 1].plot(base_r[0], base_r[1], "--", label="2BP base")
    axes[1, 1].plot(r[0], r[1], label="2BP + drag")
    axes[1, 1].set_title("XY trajectory")
    axes[1, 1].set_xlabel("x (km)")
    axes[1, 1].set_ylabel("y (km)")
    axes[1, 1].axis("equal")
    axes[1, 1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ----------------------------
# Dataset generation
# ----------------------------


def generate_2bp_drag_dataset(params, name):
    data_dict = {}
    rng = np.random.default_rng(params["seed"])

    for _ in tqdm(range(params["num_trajs"]), desc="Generating drag orbits"):
        a = rng.uniform(*params["a_range"])
        e = rng.uniform(*params["e_range"])
        inc = rng.uniform(*params["i_range"])
        raan = rng.uniform(*params["raan_range"])
        w = rng.uniform(*params["w_range"])
        nu0 = rng.uniform(*params["nu_range"])

        periapsis = a * (1 - e)
        if periapsis < params["body_radius_km"]:
            continue

        oe = np.array([a, e, inc, raan, w, nu0])
        t, y_si = _integrate_orbit(
            oe,
            params["num_points"],
            params,
            include_drag=True,
        )

        # Convert to km / km/s for storage
        y = y_si.copy()
        y[:3] /= 1e3
        y[3:] /= 1e3

        key = f"orbit_{len(data_dict) + 1}"
        data_dict[key] = {
            "t": t.tolist(),
            "y": y.tolist(),
            "metadata": {
                "sma_km": a,
                "eccentricity": e,
                "inclination": inc,
                "raan": raan,
                "argument_of_periapsis": w,
                "initial_true_anomaly": nu0,
                "num_points": params["num_points"],
                "drag": {
                    "cd_area_over_m": params["cd_area_over_m"],
                    "rho_ref_kg_m3": params["rho_ref"],
                    "h_ref_km": params["h_ref_m"] / 1e3,
                    "scale_height_km": params["scale_height_m"] / 1e3,
                    "omega_earth": params["omega_earth"],
                },
            },
        }

    save_dataset(name, data_dict, project="neuralODEs", entity="mlds-lab")
    return data_dict


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    area_m2 = 10.0
    mass_kg = 1000.0
    num_periods = 1

    params = {
        "num_trajs": 1,
        "num_periods": num_periods,
        "num_points": 360 * num_periods,
        "seed": 7,
        "a_range": [constants.RADIUS_EARTH + 300, constants.RADIUS_EARTH + 300],
        "e_range": [0.01, 0.01],
        "i_range": np.radians([0, 0]),
        "raan_range": np.radians([0, 180]),
        "w_range": np.radians([0, 180]),
        "nu_range": np.radians([0, 360]),
        # SI constants
        "mu": constants.MU_EARTH * 1e9,  # m^3/s^2
        "body_radius_m": constants.RADIUS_EARTH * 1e3,
        "body_radius_km": constants.RADIUS_EARTH,
        "omega_earth": 7.2921159e-5,  # rad/s
        "rho_ref": 2.418e-11,  # kg/m^3 @ 300 km
        "h_ref_m": 300e3,
        "scale_height_m": 53.628e3,
        "cd_area_over_m": 2.2
        * area_m2
        / mass_kg,  # HACK: artificially increase drag to see effects
    }

    dataset_name = "simple_2BP_drag_example"
    data = generate_2bp_drag_dataset(params, dataset_name)
    traj = data[list(data.keys())[0]]
    plot_path = "scripts/data_gen/plots/2bp_drag_example.png"
    plot_orbit_diagnostics(traj, params, plot_path)
    print(f"Saved plot to {plot_path}")
```

# generate_2BP_srp_data.py
``` python
"""
Generate GEO-like 2BP dynamics with Vallado cannonball SRP perturbation.

SRP model:
a_srp = - (p_srp * C_R * A / m) * r_hat_sat_sun

Earth-centered inertial (ECI), Sun fixed along +x (pedagogical setup).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from mldsml.wandb_utils import save_dataset
from scipy.integrate import solve_ivp

from neuralODE import constants

# ============================================================================
# SRP MODEL (Vallado Eq. 8-43)
# ============================================================================


def cannonball_srp_accel_km_s2(r_sat_sun_km, params):
    """
    Vallado cannonball SRP acceleration.

    Parameters
    ----------
    r_sat_sun_km : ndarray (3,)
        Vector from Sun -> spacecraft (ECI, km)
    params : dict
        p_srp   [N/m^2]
        cr
        area_m2
        mass_kg
    """
    r = np.asarray(r_sat_sun_km, dtype=float)
    r_hat = r / np.linalg.norm(r)

    accel_m_s2 = params["p_srp"] * params["cr"] * params["area_m2"] / params["mass_kg"]

    return -accel_m_s2 * r_hat / 1000.0  # km/s^2


# ============================================================================
# DYNAMICS
# ============================================================================


def two_body_srp_rhs(t, y, mu, srp_params, r_sun_eci_km):
    x, y_pos, z, vx, vy, vz = y
    r_vec = np.array([x, y_pos, z])
    r = np.linalg.norm(r_vec)

    # 2BP gravity
    a_grav = -mu * r_vec / r**3

    # SRP
    r_sat_sun = r_vec - r_sun_eci_km
    a_srp = cannonball_srp_accel_km_s2(r_sat_sun, srp_params)

    a_total = a_grav + a_srp

    return np.array([vx, vy, vz, *a_total])


# ============================================================================
# INITIAL CONDITIONS (GEO-like)
# ============================================================================


def geo_initial_state(mu, radius_km):
    r0 = np.array([radius_km, 0.0, 0.0])
    v_circ = np.sqrt(mu / radius_km)
    v0 = np.array([0.0, v_circ, 0.0])
    return np.hstack([r0, v0])


# ============================================================================
# INTEGRATION
# ============================================================================


def integrate_orbit(mu, srp_params, num_points):
    AU_km = 1.495978707e8
    r_sun_eci_km = np.array([AU_km, 0.0, 0.0])

    r_geo = 42164.0  # km
    y0 = geo_initial_state(mu, r_geo)

    period = 2 * np.pi * np.sqrt(r_geo**3 / mu)
    t_eval = np.linspace(0.0, period, num_points, endpoint=False)

    sol = solve_ivp(
        lambda t, y: two_body_srp_rhs(
            t,
            y,
            mu,
            srp_params,
            r_sun_eci_km,
        ),
        (t_eval[0], t_eval[-1]),
        y0,
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y


# ============================================================================
# DATASET GENERATION
# ============================================================================


def generate_geo_srp_dataset(params, name):
    t, y = integrate_orbit(
        params["mu"],
        params["srp"],
        params["num_points"],
    )

    data_dict = {
        "orbit_1": {
            "t": t.tolist(),
            "y": y.tolist(),
            "metadata": {
                "orbit_type": "GEO-like",
                "radius_km": 42164.0,
                "num_points": params["num_points"],
                "srp": dict(params["srp"]),
            },
        },
    }

    save_dataset(
        name,
        data_dict,
        project="neuralODEs",
        entity="mlds-lab",
    )

    return data_dict


# ============================================================================
# PLOTTING (diagnostics)
# ============================================================================


def plot_orbit_diagnostics(traj, params, out_path):
    import matplotlib.pyplot as plt

    t = np.array(traj["t"])
    y = np.array(traj["y"])

    r = y[:3]
    v = y[3:]

    mu = params["mu"]

    r_norm = np.linalg.norm(r, axis=0)
    a_grav = -mu * r / r_norm**3

    AU_km = 1.495978707e8
    r_sun_eci_km = np.array([AU_km, 0.0, 0.0])
    a_srp = np.array(
        [
            cannonball_srp_accel_km_s2(r[:, i] - r_sun_eci_km, params["srp"])
            for i in range(r.shape[1])
        ]
    ).T

    a_total = a_grav + a_srp

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    # Position
    axes[0, 0].plot(t, r[0], label="x")
    axes[0, 0].plot(t, r[1], label="y")
    axes[0, 0].plot(t, r[2], label="z")
    axes[0, 0].set_title("Position (km)")
    axes[0, 0].legend()

    # Velocity
    axes[0, 1].plot(t, v[0], label="vx")
    axes[0, 1].plot(t, v[1], label="vy")
    axes[0, 1].plot(t, v[2], label="vz")
    axes[0, 1].set_title("Velocity (km/s)")
    axes[0, 1].legend()

    # Acceleration magnitudes
    axes[1, 0].plot(t, np.linalg.norm(a_grav, axis=0), label="|a_grav|")
    axes[1, 0].plot(t, np.linalg.norm(a_srp, axis=0), label="|a_srp|")
    axes[1, 0].plot(t, np.linalg.norm(a_total, axis=0), label="|a_total|")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Acceleration magnitudes (km/s²)")
    axes[1, 0].legend()

    # XY trajectory
    axes[1, 1].plot(r[0], r[1])
    axes[1, 1].set_aspect("equal")
    axes[1, 1].set_title("XY trajectory (ECI)")
    axes[1, 1].set_xlabel("x (km)")
    axes[1, 1].set_ylabel("y (km)")

    # Acceleration components
    axes[2, 0].plot(t, a_grav[0], label="ax grav")
    axes[2, 0].plot(t, a_srp[0], ":", label="ax SRP")
    axes[2, 0].plot(t, a_total[0], "--", label="ax total")
    axes[2, 0].set_title("Acceleration components (x)")
    axes[2, 0].set_xlabel("Time")
    axes[2, 0].set_ylabel("km/s^2")
    axes[2, 0].set_yscale("symlog", linthresh=1e-12)
    axes[2, 0].legend()

    axes[2, 1].plot(t, a_grav[1], label="ay grav")
    axes[2, 1].plot(t, a_srp[1], ":", label="ay SRP")
    axes[2, 1].plot(t, a_total[1], "--", label="ay total")
    axes[2, 1].set_title("Acceleration components (y)")
    axes[2, 1].set_xlabel("Time")
    axes[2, 1].set_ylabel("km/s^2")
    axes[2, 1].set_yscale("symlog", linthresh=1e-12)
    axes[2, 1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    params = {
        "num_points": 720,
        "mu": constants.MU_EARTH,
        "srp": {
            "p_srp": 4.56e-6,  # N/m^2
            "cr": 1.3,
            "area_m2": 100.0,
            "mass_kg": 100.0,
        },
    }

    dataset_name = "GEO_TBP_SRP_Vallado"
    data = generate_geo_srp_dataset(params, dataset_name)

    traj = data["orbit_1"]
    plot_path = "scripts/data_gen/plots/geo_srp_example.png"
    plot_orbit_diagnostics(traj, params, plot_path)

    print(f"Saved dataset '{dataset_name}' and plot to {plot_path}")
```

# generate_2BP_tangential_thrust_data.py
``` python
"""
Generate datasets for 2BP dynamics with a tangential thrust perturbation.

Format matches sampling_2BP.generate_datasets: dict of trajectories with keys
{t, y, metadata}, where y has shape (num_states, num_points).
"""

import os

import numpy as np
from mldsml.wandb_utils import save_dataset
from scipy.integrate import solve_ivp

from neuralODE import astro_utils, constants, coordinate_conversions


def _two_body_accel(r, mu):
    r_norm = np.linalg.norm(r)
    return -mu * r / r_norm**3


def _tangential_unit(v):
    v_norm = np.linalg.norm(v)
    if v_norm == 0.0:
        return np.zeros(3)
    return v / v_norm


def _two_body_rhs(t, y, mu):
    r = y[:3]
    v = y[3:]
    a_grav = _two_body_accel(r, mu)
    return np.array([v[0], v[1], v[2], a_grav[0], a_grav[1], a_grav[2]])


def _tangential_thrust_rhs(t, y, mu, accel_mag):
    r = y[:3]
    v = y[3:]
    a_grav = _two_body_accel(r, mu)
    tangential = _tangential_unit(v)
    a_thrust = accel_mag * tangential
    a_total = a_grav + a_thrust
    return np.array([v[0], v[1], v[2], a_total[0], a_total[1], a_total[2]])


def _integrate_orbit(oe, num_points, mu, *, accel_mag=None):
    a, e, inc, raan, w, nu0 = oe
    period = astro_utils.calculate_orbital_period(a, mu)
    times = np.linspace(0.0, period, num_points, endpoint=False)
    r0, v0 = coordinate_conversions.standard_to_cartesian(
        a,
        e,
        inc,
        raan,
        w,
        nu0,
        mu=mu,
    )
    y0 = np.hstack([r0, v0])

    if accel_mag is None:
        rhs = lambda t, y: _two_body_rhs(t, y, mu)
    else:
        rhs = lambda t, y: _tangential_thrust_rhs(t, y, mu, accel_mag)

    sol = solve_ivp(
        rhs,
        (times[0], times[-1]),
        y0=y0,
        t_eval=times,
        rtol=1e-9,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.t, sol.y


def generate_2bp_tangential_thrust_dataset(params, name):
    data_dict = {}
    rng = np.random.default_rng(params["seed"])
    accel_mag = params["thrust"]["accel_km_s2"]

    for _ in range(params["num_trajs"]):
        a = rng.uniform(*params["a_range"])
        e = rng.uniform(*params["e_range"])
        inc = rng.uniform(*params["i_range"])
        raan = rng.uniform(*params["raan_range"])
        w = rng.uniform(*params["w_range"])
        nu0 = rng.uniform(*params["nu_range"])

        periapsis = a * (1 - e)
        if periapsis < params["body_radius"]:
            continue

        oe = np.array([a, e, inc, raan, w, nu0])
        t_eval, y_eval = _integrate_orbit(
            oe,
            params["num_points"],
            params["mu"],
            accel_mag=accel_mag,
        )

        traj_key = f"orbit_{len(data_dict) + 1}"
        data_dict[traj_key] = {
            "t": t_eval.tolist(),
            "y": y_eval.tolist(),
            "metadata": {
                "sma": a,
                "eccentricity": e,
                "inclination": inc,
                "raan": raan,
                "argument_of_periapsis": w,
                "initial_true_anomaly": nu0,
                "num_points": params["num_points"],
                "thrust": dict(params["thrust"]),
            },
        }

    save_dataset(
        name,
        data_dict,
        project="neuralODEs",
        entity="mlds-lab",
    )
    return data_dict


def plot_orbit_diagnostics(traj, params, out_path):
    import matplotlib.pyplot as plt

    t = np.array(traj["t"])
    y = np.array(traj["y"])
    r = y[:3]
    v = y[3:]

    oe = np.array(
        [
            traj["metadata"]["sma"],
            traj["metadata"]["eccentricity"],
            traj["metadata"]["inclination"],
            traj["metadata"]["raan"],
            traj["metadata"]["argument_of_periapsis"],
            traj["metadata"]["initial_true_anomaly"],
        ],
    )
    base_t, base_y = _integrate_orbit(
        oe,
        params["num_points"],
        params["mu"],
        accel_mag=None,
    )
    base_r = base_y[:3]
    base_v = base_y[3:]

    mu = params["mu"]
    r_norm = np.linalg.norm(r, axis=0)
    base_accel = -mu * r / r_norm**3
    tangential = np.array([_tangential_unit(v[:, idx]) for idx in range(v.shape[1])]).T
    thrust_accel = params["thrust"]["accel_km_s2"] * tangential
    total_accel = base_accel + thrust_accel

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t, base_r[0], label="x (base)")
    axes[0, 0].plot(t, base_r[1], label="y (base)")
    axes[0, 0].plot(t, r[0], ":", label="x (perturbed)")
    axes[0, 0].plot(t, r[1], ":", label="y (perturbed)")
    axes[0, 0].set_title("Position states (planar)")
    axes[0, 0].set_xlabel("Time")
    axes[0, 0].set_ylabel("km")
    axes[0, 0].legend()

    axes[0, 1].plot(t, base_v[0], label="vx (base)")
    axes[0, 1].plot(t, base_v[1], label="vy (base)")
    axes[0, 1].plot(t, v[0], ":", label="vx (perturbed)")
    axes[0, 1].plot(t, v[1], ":", label="vy (perturbed)")
    axes[0, 1].set_title("Velocity states (planar)")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("km/s")
    axes[0, 1].legend()

    base_mag = np.linalg.norm(base_accel, axis=0)
    thrust_mag = np.linalg.norm(thrust_accel, axis=0)
    total_mag = np.linalg.norm(total_accel, axis=0)
    axes[1, 0].plot(t, base_mag, label="base |a|")
    axes[1, 0].plot(t, thrust_mag, label="thrust |a|")
    axes[1, 0].plot(t, total_mag, ":", label="total |a|")
    axes[1, 0].plot(
        t,
        np.linalg.norm(total_accel - base_accel, axis=0),
        label="|total - base|",
    )
    ratio = total_mag / base_mag
    ratio_ax = axes[1, 0].twinx()
    ratio_ax.plot(t, ratio, "--", color="gray", label="total/base")
    ratio_ax.set_ylabel("ratio")
    axes[1, 0].set_title("Acceleration magnitudes")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel("km/s^2")
    axes[1, 0].set_yscale("log")
    axes[1, 0].legend()

    axes[1, 1].plot(base_r[0], base_r[1], "--", label="2BP base")
    axes[1, 1].plot(r[0], r[1], label="2BP + tangential thrust")
    axes[1, 1].set_title("XY trajectory")
    axes[1, 1].set_xlabel("x (km)")
    axes[1, 1].set_ylabel("y (km)")
    axes[1, 1].axis("equal")
    axes[1, 1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    params = {
        "num_trajs": 1,
        "num_points": 360,
        "seed": 42,
        "a_range": [constants.RADIUS_EARTH + 10000, constants.RADIUS_EARTH + 10000],
        "e_range": [0.3, 0.3],
        "i_range": np.radians([0, 0]),
        "raan_range": np.radians([0, 180]),
        "w_range": np.radians([0, 180]),
        "nu_range": np.radians([0, 360]),
        "body_radius": constants.RADIUS_EARTH,
        "mu": constants.MU_EARTH,
        "thrust": {
            "accel_km_s2": 1e-4,
        },
    }

    dataset_name = "simple_TBP_tangential_thrust_example"
    data = generate_2bp_tangential_thrust_dataset(params, dataset_name)
    traj = data[list(data.keys())[0]]
    plot_path = "scripts/data_gen/plots/2bp_tangential_thrust_example.png"
    plot_orbit_diagnostics(traj, params, plot_path)
    print(f"Saved plot to {plot_path}")
```

# generate_forced_oscillator_data.py
``` python
"""
Generate datasets for a forced harmonic oscillator with a state-dependent forcing term.

Format matches sampling_2BP.generate_datasets: dict of trajectories with keys
{t, y, metadata}, where y has shape (num_states, num_points).
"""

import os

import numpy as np
from mldsml.wandb_utils import save_dataset
from scipy.integrate import solve_ivp


def forcing_term(x, v, t, params):
    """Weak nonlinear state-dependent forcing term with acceleration units."""
    epsilon = params["epsilon"]
    x_shift = params["x_shift"]
    return epsilon * (x - x_shift) ** 3


def forced_oscillator_rhs(t, y, params):
    x, v = y
    omega = params["omega"]
    accel = -(omega**2) * x + forcing_term(x, v, t, params)
    return np.array([v, accel])


def generate_forced_oscillator_dataset(params, name):
    data_dict = {}
    rng = np.random.default_rng(params["seed"])

    for traj_idx in range(params["num_trajs"]):
        x0 = rng.uniform(*params["x0_range"])
        v0 = rng.uniform(*params["v0_range"])

        t_eval = np.linspace(0.0, params["t_final"], params["num_points"])
        sol = solve_ivp(
            lambda t, y: forced_oscillator_rhs(t, y, params),
            (t_eval[0], t_eval[-1]),
            y0=np.array([x0, v0]),
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-12,
        )

        if not sol.success:
            raise RuntimeError(f"Integration failed for trajectory {traj_idx + 1}")

        traj_key = f"traj_{traj_idx + 1}"
        data_dict[traj_key] = {
            "t": sol.t.tolist(),
            "y": sol.y.tolist(),
            "metadata": {
                "omega": params["omega"],
                "x0": x0,
                "v0": v0,
                "t_final": params["t_final"],
                "num_points": params["num_points"],
                "forcing": {
                    "epsilon": params["epsilon"],
                    "x_shift": params["x_shift"],
                },
            },
        }

    save_dataset(
        name,
        data_dict,
        project="neuralODEs",
        entity="mlds-lab",
    )

    return data_dict


def plot_trajectory_components(traj, params, out_path):
    import matplotlib.pyplot as plt

    t = np.array(traj["t"])
    y = np.array(traj["y"])
    x = y[0]
    v = y[1]

    x0 = traj["metadata"]["x0"]
    v0 = traj["metadata"]["v0"]
    base_sol = solve_ivp(
        lambda t, y: np.array([y[1], -(params["omega"] ** 2) * y[0]]),
        (t[0], t[-1]),
        y0=np.array([x0, v0]),
        t_eval=t,
        rtol=1e-9,
        atol=1e-12,
    )
    x_base = base_sol.y[0]
    v_base = base_sol.y[1]

    base_accel = -(params["omega"] ** 2) * x
    forcing = forcing_term(x, v, t, params)
    total_accel = base_accel + forcing

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, x, label="x(t) forced")
    axes[0].plot(t, v, label="v(t) forced")
    axes[0].plot(t, x_base, "--", label="x(t) base")
    axes[0].plot(t, v_base, "--", label="v(t) base")
    axes[0].set_ylabel("State")
    axes[0].legend()

    axes[1].plot(t, base_accel, label="base: -omega^2 x")
    axes[1].plot(t, forcing, label="forcing: f(x, v)")
    axes[1].plot(t, total_accel, label="total accel")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Acceleration")
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    params = {
        "num_trajs": 1,
        "num_points": 500,
        "t_final": 20.0,
        "omega": 1.4,
        "x0_range": [-1.0, 1.0],
        "v0_range": [-0.5, 0.5],
        "epsilon": 1.0,
        "x_shift": 0.0,
        "seed": 11,
    }

    dataset_name = "forced_oscillator_1traj"
    data = generate_forced_oscillator_dataset(params, dataset_name)

    traj = data["traj_1"]
    plot_path = "scripts/data_gen/plots/forced_oscillator_example.png"
    plot_trajectory_components(traj, params, plot_path)
    print(f"Saved plot to {plot_path}")
26```