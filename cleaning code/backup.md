# Feature layer
``` python
def rhat_vhat_speed(y, eps=1e-8):
    """General feature set: unit position, inverse radius, speed, unit velocity.

    Adds radial/transverse velocity fractions (wrt r_hat) to expose alignment of
    motion without assuming a central body or mu. Works in 2BP or CR3BP.
    """

    r = y[:3]
    v = y[3:6]

    r_norm = jnp.linalg.norm(r) + eps
    v_norm = jnp.linalg.norm(v) + eps

    r_hat = r / r_norm
    v_hat = v / v_norm

    vr_frac = jnp.dot(v_hat, r_hat)
    vt_frac = jnp.linalg.norm(v_hat - vr_frac * r_hat)

    return jnp.concatenate(
        [
            jnp.array(
                [
                    1.0 / r_norm,
                    *r_hat,
                    v_norm,
                    *v_hat,
                    vr_frac,
                    vt_frac,
                ]
            ),
        ],
    )

	...

feature_layers = {...
    "rhat_vhat_speed": FeatureLayerSpec(
        rhat_vhat_speed,
        10,
        (
            "1/r",
            "sx",
            "sy",
            "sz",
            "|v|",
            "vx_hat",
            "vy_hat",
            "vz_hat",
            "vr_frac",
            "vt_frac",
        ),
    ),
}
```

# Attraction penalty in loss
This loss function penalizes accelerations pointing radially outward (dot product between position and acceleration vector is positive). This isn't very useful because this assumes that we know something about the physics, but we don't want to make such assumptions.
``` python
def _attraction_penalty(model, ti, y_pred, mask_i, eps=1e-8):
    """
    Encourage accelerations to remain attractive (dot(position, accel) <= 0).
    Returns a penalty scaled to roughly match percent error (0-100).
    """

    def eval_sequence(ts, states):
        return jax.vmap(lambda t, y: model.func(t, y))(ts, states)

    derivs = jax.vmap(eval_sequence)(ti, y_pred)
    positions = y_pred[..., :3]
    accelerations = derivs[..., 3:]
    dots = jnp.sum(positions * accelerations, axis=-1)
    pos_norm = jnp.linalg.norm(positions, axis=-1)
    accel_norm = jnp.linalg.norm(accelerations, axis=-1)
    denom = pos_norm * accel_norm
    denom = jnp.where(denom < eps, eps, denom)
    cosine = dots / denom
    penalty = jnp.maximum(cosine, 0.0)  # positive values indicate repulsive accel
    if mask_i is not None:
        mask = jnp.asarray(mask_i, dtype=bool)
        penalty = jnp.where(mask, penalty, jnp.nan)
    return jnp.nanmean(penalty) * 100.0


def percent_error_with_attraction_loss(
    model,
    ti,
    yi,
    mask_i,
    attraction_weight=1.0,
    *,
    return_components=False,
):
    """
    Percent error loss augmented with an attraction penalty that discourages
    repulsive accelerations (dot(position, acceleration) > 0).
    """
    y_pred = get_y_pred(model, ti, yi, mask_i)
    y_true = yi[:, 1:, :]
    y_pred_aligned = y_pred[:, 1:, :]

    threshold = 1e-8
    true_norm = jnp.linalg.norm(y_true, axis=-1)
    safe_denominator = jnp.where(true_norm < threshold, threshold, true_norm)
    mpe = jnp.linalg.norm((y_true - y_pred_aligned), axis=-1) / safe_denominator * 100
    mpe_mean = jnp.nanmean(mpe)

    attraction_term = attraction_weight * _attraction_penalty(
        model,
        ti,
        y_pred,
        mask_i,
    )
    total = mpe_mean + attraction_term

    components = {
        "percent_error_component": mpe_mean,
        "attraction_component": attraction_term,
        "percent_error_with_attraction": total,
    }
    if return_components:
        return total, components
    return total


percent_error_with_attraction_loss.supports_components = True
```

# Solve with feature capture
I don't think I need these versions with "internal steps"
``` python
    def solve_with_feature_capture(
        self,
        ts,
        y0,
        *,
        capture_internal_steps=False,
        capture_accel=True,
        capture_output_features=False,
        output_features_post_activation=False,
    ):
        """
        Solve the ODE while also recording feature-layer outputs along the trajectory.
        """
        ts = jnp.asarray(ts)
        y0 = jnp.asarray(y0)
        if ts.shape[0] < 2:
            raise ValueError("Need at least two time samples to capture features.")
        dt0 = ts[1] - ts[0]
        feature_fn = lambda t, y, args: self.func.feature_layer(y)
        subs = {
            "states": SubSaveAt(ts=ts, fn=save_y),
            "features": SubSaveAt(ts=ts, fn=feature_fn),
        }
        if capture_internal_steps:
            subs["feature_steps"] = SubSaveAt(steps=True, fn=feature_fn)
        if capture_accel:
            subs["accelerations"] = SubSaveAt(
                ts=ts,
                fn=lambda t, y, args: self.func(t, y, args),
            )
            if capture_internal_steps:
                subs["accel_steps"] = SubSaveAt(
                    steps=True,
                    fn=lambda t, y, args: self.func(t, y, args),
                )
        if capture_output_features:
            output_fn = lambda t, y, args: self.func.get_output_features(
                y,
                post_activation=output_features_post_activation,
            )

            subs["output_features"] = SubSaveAt(ts=ts, fn=output_fn)
            if capture_internal_steps:
                subs["output_step_features"] = SubSaveAt(
                    steps=True,
                    fn=output_fn,
                )

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=float(ts[0]),
            t1=float(ts[-1]),
            dt0=float(dt0),
            y0=y0,
            stepsize_controller=diffrax.PIDController(
                rtol=self.rtol,
                atol=self.atol,
            ),
            saveat=diffrax.SaveAt(subs=subs),
        )
        result = {
            "ts": solution.ts["states"],
            "states": solution.ys["states"],
            "features": solution.ys["features"],
        }
        if capture_internal_steps and "feature_steps" in solution.ys:
            result["step_ts"] = solution.ts["feature_steps"]
            result["step_features"] = solution.ys["feature_steps"]
        if capture_accel and "accelerations" in solution.ys:
            result["accelerations"] = solution.ys["accelerations"]
        if capture_accel and capture_internal_steps and "accel_steps" in solution.ys:
            result["step_accelerations"] = solution.ys["accel_steps"]
        if capture_output_features and "output_features" in solution.ys:
            result["output_features"] = solution.ys["output_features"]
        if (
            capture_output_features
            and capture_internal_steps
            and "output_step_features" in solution.ys
        ):
            result["step_output_features"] = solution.ys["output_step_features"]
        return result

```

# Final activation 
``` python
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width,
            depth=depth,
            activation=getattr(jnn, config.parameters.activation),
            # final_activation=jnn.tanh,
            key=key,
        )
```

# Sensitivity Analysis
``` python
``` python
import numbers
import os
import pickle
from pathlib import Path
from typing import Any, Iterable, Optional

import jax.numpy as jnp
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import mldsml
import numpy as np
import pandas as pd
import ray
import seaborn as sns
from mldsml.wandb_utils import (
    load_dataset,
    query_runs_in_group,
)
from pandas.plotting import parallel_coordinates

import neuralODE
from neuralODE import constants
from neuralODE.data import (
    format_data,
)
from neuralODE.dynamics import get_dynamics_class
from neuralODE.experiments.IntegrationExperiment import IntegrationExperiment
from neuralODE.metrics import AccelerationMetric
from neuralODE.neuralODE import load_model_wandb
from neuralODE.normalization import Normalization2BP
from neuralODE.visualizers.integrationVisualizer import ModIntegrationVisualizer

neuralODE_path = neuralODE.__path__[0]

np.random.seed(42)  # For reproducibility
# use style sheet

mldsml_path = mldsml.__path__[0]
plt.style.use(f"{mldsml_path}/styles/dark_scientific.mplstyle")


matplotlib.use("Agg")  # must be before pyplot

# ...
plt.style.use(f"{mldsml_path}/styles/dark_scientific.mplstyle")
mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["font.family"] = "DejaVu Sans"

ROOT_DIR = Path(neuralODE_path).resolve().parent
FIGURES_DIR = ROOT_DIR / "files" / "figures"
RESULTS_DIR = ROOT_DIR / "files" / "results"


ALL_DATASETS = [
    "single_complex_TBP_planar_v3",
    "complex_TBP_planar_10",
    "complex_TBP_planar",
]
DEFAULT_WANDB_GROUP = "sensitivity-2D-v4"


def ensure_directory(path: Path) -> Path:
    """Create a directory (recursively) and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def flatten_numeric(sequence: Any) -> list[float]:
    """Recursively flatten nested numeric sequences."""
    if isinstance(sequence, numbers.Real):
        return [float(sequence)]
    if isinstance(sequence, (list, tuple)):
        flattened: list[float] = []
        for item in sequence:
            flattened.extend(flatten_numeric(item))
        return flattened
    return []


def format_length_strategy(strategy: Any) -> str:
    """Convert a length strategy definition into a readable label."""
    if strategy is None:
        return "unspecified"
    if isinstance(strategy, str):
        return strategy
    if isinstance(strategy, (list, tuple)):
        tokens: list[str] = []
        for stage in strategy:
            if isinstance(stage, (list, tuple)) and len(stage) >= 2:
                try:
                    start, end = stage[0], stage[1]
                    if isinstance(start, numbers.Real) and isinstance(
                        end, numbers.Real
                    ):
                        tokens.append(f"{float(start):.2f}-{float(end):.2f}")
                        continue
                except Exception:
                    pass
            if isinstance(stage, numbers.Real):
                tokens.append(f"{float(stage):.3g}")
            else:
                tokens.append(str(stage))
        if tokens:
            return " → ".join(tokens)
    return str(strategy)


def parse_schedule_extremes(strategy: Any) -> tuple[float, float]:
    """Return the first and last numeric entries from a (possibly nested) schedule."""
    values = flatten_numeric(strategy)
    if not values:
        return (np.nan, np.nan)
    return (values[0], values[-1])


def fetch_last_logged_value(run, key: str, default: Any = np.nan) -> Any:
    """Safely pull the latest logged value for a key from a W&B run."""
    try:
        history = run.history(keys=[key])
        history = history.dropna(subset=[key])
        if not history.empty:
            value = history.iloc[-1][key]
            if value is not None:
                return value
    except Exception:
        pass
    try:
        value = run.summary.get(key, default)
        return default if value is None else value
    except Exception:
        return default


def gather_run_metadata(config, run) -> dict[str, Any]:
    """Collect hyperparameter metadata required for plotting."""
    metadata: dict[str, Any] = {}
    metadata["segment_length"] = getattr(config, "segment_length", None)

    raw_length_strategy = getattr(config, "length_strategy", None)
    if raw_length_strategy is None:
        raw_length_strategy = getattr(
            getattr(config, "parameters", None), "length_strategy", None
        )
    metadata["length_strategy"] = raw_length_strategy
    metadata["length_strategy_label"] = format_length_strategy(raw_length_strategy)
    metadata["length_strategy_stages"] = (
        len(raw_length_strategy)
        if isinstance(raw_length_strategy, (list, tuple))
        else np.nan
    )

    lr_first, lr_last = parse_schedule_extremes(getattr(config, "lr_strategy", None))
    metadata["lr_initial"] = lr_first
    metadata["lr_final"] = lr_last

    steps_first, steps_last = parse_schedule_extremes(
        getattr(config, "steps_strategy", None)
    )
    metadata["steps_initial"] = steps_first
    metadata["steps_final"] = steps_last

    metadata["width"] = getattr(
        config, "width", getattr(config, "parameters", {}).get("width", np.nan)
    )
    metadata["depth"] = getattr(
        config, "depth", getattr(config, "parameters", {}).get("depth", np.nan)
    )
    metadata["feature_layer"] = getattr(config, "feature_layer", None)
    metadata["planar_constraint"] = getattr(config, "planar_constraint", None)

    metadata["num_train_orbits"] = fetch_last_logged_value(run, "num_train_orbits")
    metadata["segment_ratio"] = fetch_last_logged_value(run, "segment_ratio")
    metadata["num_total_segments"] = fetch_last_logged_value(run, "num_total_segments")
    metadata["num_total_orbits"] = fetch_last_logged_value(run, "num_total_orbits")

    metadata["wandb_run_id"] = getattr(run, "id", None)
    metadata["wandb_run_name"] = getattr(run, "name", None)
    metadata["wandb_group"] = getattr(run, "group", None)
    return metadata


def prepare_results_dataframe(results_all: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Create a tidy DataFrame with derived fields for plotting."""
    df = pd.DataFrame(results_all).copy()
    if df.empty:
        raise ValueError("No results available to visualize.")

    numeric_cols = [
        "mean_acc_err",
        "num_train_orbits",
        "segment_length",
        "segment_ratio",
        "lr_initial",
        "lr_final",
        "steps_initial",
        "steps_final",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["length_strategy_label"] = df.get("length_strategy_label", "unspecified").fillna(
        "unspecified"
    )
    if "test_dataset" not in df:
        df["test_dataset"] = "unknown"

    # Compute derived helper columns
    if {"num_total_segments", "num_total_orbits"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["segments_per_orbit"] = df["num_total_segments"] / df["num_total_orbits"]

    df["mean_acc_err"] = df["mean_acc_err"].replace({0: np.nan})
    df = df[df["mean_acc_err"].notna()]

    return df


def evaluate_acceleration_residuals(data_set, model, config, run_id):
    """Calculate the mean acceleration error for a model on a dataset."""
    dyn = get_dynamics_class(config.data.problem)

    # Get average acceleration error over trajectories
    data_dict = load_dataset(
        data_set,
        version="latest",
        project="neuralODEs",
        entity="mlds-lab",
    )
    data_dict_jnp = format_data(data_dict)

    ts = data_dict_jnp["t"]
    ys = data_dict_jnp["y"]
    mask_ys = data_dict_jnp["mask"]
    metric = AccelerationMetric(true_dynamics=dyn)
    acc_error = metric(model, ts, ys, mask_ys)
    return acc_error


def plot_integrated_orbits(test_dataset, model, config, run_id):
    """Plot the integrated orbits using the model and return the figures."""
    # Loading dataset to test the model on
    data_dict = load_dataset(
        test_dataset,
        version="latest",
        project="neuralODEs",
        entity="mlds-lab",
    )
    transform = Normalization2BP(
        l_char=constants.RADIUS_EARTH,
        mu=constants.MU_EARTH,
    )
    data_dict = transform.normalize_dataset(data_dict)
    data_dict_jnp = format_data(data_dict)

    # Get all initial conditions and times
    ICs = data_dict_jnp["y"][:, 0, :]  # All initial conditions
    times = data_dict_jnp["t"]  # All time arrays

    # Calculate t0, tf, and dt for each orbit
    t0_idx_list = [np.where(~np.isnan(times[i]))[0][0] for i in range(times.shape[0])]
    tf_idx_list = [np.where(~np.isnan(times[i]))[0][-1] for i in range(times.shape[0])]

    tf_list = jnp.array([times[i][tf_idx_list[i]] for i in range(times.shape[0])])
    dt_list = jnp.array(
        [(tf_list[i] - times[i][t0_idx_list[i]]) / 500 for i in range(times.shape[0])],
    )

    # If there are too many orbits, limit them
    max_orbits = 20  # Set a reasonable limit
    if ICs.shape[0] > max_orbits:
        # Select a subset of orbits evenly distributed across the dataset
        step = ICs.shape[0] // max_orbits
        indices = np.arange(0, ICs.shape[0], step)[:max_orbits]
        ICs = ICs[indices]
        tf_list = tf_list[indices]
        dt_list = dt_list[indices]

    # Create and run the experiment
    experiment = IntegrationExperiment(
        model=model,
        initial_conditions=ICs,
        tf=tf_list,
        dt=dt_list,
        true_dynamics=get_dynamics_class(config.data.problem),
    )
    experiment.run()

    # Visualize the results
    vis = ModIntegrationVisualizer(experiment)
    fig = vis.plot()
    fig_small = vis.plot_small()

    return fig, fig_small


@ray.remote(num_cpus=4, num_gpus=0)
def evaluate_model(model, config, run_id, **kwargs):
    """Remote function to evaluate a model across multiple datasets."""
    train_data_set = config.data.dataset_name
    metadata = dict(kwargs)

    results = []
    for data_set in ALL_DATASETS:
        value = evaluate_acceleration_residuals(data_set, model, config, run_id)

        # Base entry
        entry = {
            "train_data_set": train_data_set,
            "test_dataset": data_set,
            "mean_acc_err": value,
        }

        entry.update(metadata)
        results.append(entry)

    return results


def create_heatmap_visualization(
    df: pd.DataFrame, output_dir: Path
) -> Optional[plt.Figure]:
    """Heatmap of mean error vs. segment length and orbit count, faceted by strategy/test dataset."""
    required = {"segment_length", "num_train_orbits", "mean_acc_err"}
    if not required.issubset(df.columns):
        return None

    plot_df = df.dropna(subset=required).copy()
    if plot_df.empty:
        return None

    plot_df["segment_length"] = plot_df["segment_length"].astype(float)
    plot_df["num_train_orbits"] = plot_df["num_train_orbits"].astype(float)

    strategies = sorted(plot_df["length_strategy_label"].dropna().unique())
    datasets = sorted(plot_df["test_dataset"].dropna().unique())
    if not strategies:
        strategies = ["unspecified"]
        plot_df["length_strategy_label"] = "unspecified"
    if not datasets:
        datasets = ["unknown"]
        plot_df["test_dataset"] = "unknown"

    vmin = float(plot_df["mean_acc_err"].min())
    vmax = float(plot_df["mean_acc_err"].max())
    if np.isclose(vmin, vmax):
        eps = 1e-12 if vmin == 0 else abs(vmin) * 1e-6
        vmin -= eps
        vmax += eps

    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    n_rows = len(strategies)
    n_cols = len(datasets)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 + 3.2 * n_cols, 3.5 * n_rows + 1.5),
        squeeze=False,
    )

    mappable = None
    for i, strategy in enumerate(strategies):
        for j, dataset in enumerate(datasets):
            ax = axes[i, j]
            subset = plot_df[
                (plot_df["length_strategy_label"] == strategy)
                & (plot_df["test_dataset"] == dataset)
            ]
            if subset.empty:
                ax.text(
                    0.5,
                    0.5,
                    "no runs",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            pivot = subset.pivot_table(
                values="mean_acc_err",
                index="segment_length",
                columns="num_train_orbits",
                aggfunc="median",
            )
            pivot = pivot.sort_index(ascending=False).sort_index(axis=1)
            heatmap = sns.heatmap(
                pivot,
                ax=ax,
                cmap="viridis_r",
                norm=norm,
                cbar=False,
                annot=True,
                fmt=".2e",
                linewidths=0.5,
                linecolor="0.15",
            )
            if mappable is None and heatmap.collections:
                mappable = heatmap.collections[0]

            ax.set_xlabel("num_train_orbits")
            if j == 0:
                ax.set_ylabel("segment_length")
            else:
                ax.set_ylabel("")
            ax.set_title(f"{dataset}")
        axes[i, 0].set_ylabel(f"segment_length\n(strategy: {strategies[i]})")

    if mappable is not None:
        fig.colorbar(
            mappable,
            ax=axes,
            fraction=0.03,
            pad=0.02,
            label="Mean acceleration error (log scale)",
        )

    fig.suptitle("Sensitivity: segment_length × num_train_orbits", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.96, 0.97))

    ensure_directory(output_dir)
    pdf_path = output_dir / "heatmap_data_sensitivity.pdf"
    png_path = output_dir / "heatmap_data_sensitivity.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    return fig


def create_lineplot_visualization(
    df: pd.DataFrame, output_dir: Path
) -> Optional[plt.Figure]:
    """Line plots of error vs. segment length, colored by orbit count."""
    required = {"segment_length", "num_train_orbits", "mean_acc_err"}
    plot_df = df.dropna(subset=required).copy()
    if plot_df.empty:
        return None

    def _format_orbit_count(value: float) -> str:
        if pd.isna(value):
            return "missing"
        if np.isclose(value, round(value)):
            return f"{int(round(value))}"
        return f"{value:.2f}"

    plot_df["num_train_orbits_cat"] = plot_df["num_train_orbits"].apply(
        _format_orbit_count
    )

    g = sns.relplot(
        data=plot_df,
        x="segment_length",
        y="mean_acc_err",
        hue="num_train_orbits_cat",
        kind="line",
        marker="o",
        estimator="median",
        errorbar=None,
        col="test_dataset",
        row="length_strategy_label",
        facet_kws={"sharex": True, "sharey": False, "margin_titles": True},
        height=3.0,
        aspect=1.2,
        palette="viridis",
    )

    g.set_axis_labels("segment_length", "mean_acc_err")
    g.add_legend(title="num_train_orbits")

    for ax in g.axes.flatten():
        ax.set_yscale("log")

    g.fig.suptitle("Sensitivity trends across segment lengths", fontsize=16, y=1.02)
    g.fig.tight_layout()

    ensure_directory(output_dir)
    pdf_path = output_dir / "lineplot_segment_trends.pdf"
    png_path = output_dir / "lineplot_segment_trends.png"
    g.savefig(pdf_path)
    g.savefig(png_path, dpi=300)
    return g.fig


def create_parallel_coordinates_visualization(
    df: pd.DataFrame,
    output_dir: Path,
) -> Optional[plt.Figure]:
    """Parallel coordinates plot to inspect multi-parameter interactions."""
    candidate_cols = [
        "segment_length",
        "num_train_orbits",
        "lr_initial",
        "lr_final",
        "steps_initial",
        "steps_final",
    ]
    available_cols = [col for col in candidate_cols if col in df.columns]
    if len(available_cols) < 2:
        return None

    plot_df = df[available_cols + ["mean_acc_err"]].dropna()
    if len(plot_df) < 3:
        return None

    num_bins = min(len(plot_df), 4)
    try:
        error_bins = pd.qcut(
            plot_df["mean_acc_err"],
            q=num_bins,
            labels=[f"Q{i + 1}" for i in range(num_bins)],
            duplicates="drop",
        )
    except ValueError:
        return None

    plot_df = plot_df.assign(error_bin=error_bins).dropna(subset=["error_bin"])
    if plot_df.empty:
        return None

    # Normalize columns to [0, 1] for readability
    scaled_df = plot_df.copy()
    for col in available_cols:
        col_vals = scaled_df[col].astype(float)
        min_val = col_vals.min()
        max_val = col_vals.max()
        if np.isclose(min_val, max_val):
            scaled_df[col] = 0.5
        else:
            scaled_df[col] = (col_vals - min_val) / (max_val - min_val)

    scaled_df["error_bin"] = scaled_df["error_bin"].astype(str)

    fig, ax = plt.subplots(figsize=(1.8 * len(available_cols) + 3, 5))
    palette = sns.color_palette("viridis", scaled_df["error_bin"].nunique())
    parallel_coordinates(
        scaled_df[[*available_cols, "error_bin"]],
        class_column="error_bin",
        cols=available_cols,
        color=palette,
        ax=ax,
        linewidth=1.5,
    )

    ax.set_ylabel("Normalized value")
    ax.set_title("Parallel coordinates of hyperparameters vs. error")
    ax.legend(
        title="mean_acc_err quantile",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=scaled_df["error_bin"].nunique(),
    )

    fig.tight_layout()

    ensure_directory(output_dir)
    pdf_path = output_dir / "parallel_coordinates_hyperparams.pdf"
    png_path = output_dir / "parallel_coordinates_hyperparams.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    return fig


def run_analysis(
    force_recompute: bool = True, group_name: Optional[str] = None
) -> pd.DataFrame:
    """Load (or compute) evaluation results and render visualizations."""
    ensure_directory(FIGURES_DIR)
    ensure_directory(RESULTS_DIR)

    wandb_group = group_name or os.environ.get("SENSITIVITY_GROUP", DEFAULT_WANDB_GROUP)
    runs = query_runs_in_group(wandb_group)

    results_filename = f"2BP_results_sensitivity_{wandb_group}.pkl"
    results_path = RESULTS_DIR / results_filename

    if results_path.exists() and not force_recompute:
        with open(results_path, "rb") as f:
            results_all = pickle.load(f)
        print(f"Loaded cached results from {results_path}")
    else:
        print(f"Collecting models from W&B group '{wandb_group}'...")
        ray.init(ignore_reinit_error=True)
        try:
            results_all: list[dict[str, Any]] = []
            futures_all = []
            for run in runs:
                config, model, artifact_run = load_model_wandb(
                    run.id,
                    project="neuralODEs",
                    entity="mlds-lab",
                )
                metadata = gather_run_metadata(config, run)
                metadata["artifact_run_id"] = getattr(artifact_run, "id", None)
                future = evaluate_model.remote(
                    model,
                    config,
                    artifact_run.id,
                    **metadata,
                )
                futures_all.append(future)
            for future in futures_all:
                results_all.extend(ray.get(future))
        finally:
            ray.shutdown()

        ensure_directory(RESULTS_DIR)
        with open(results_path, "wb") as f:
            pickle.dump(results_all, f)
        print(f"Wrote aggregated results to {results_path}")

    df = prepare_results_dataframe(results_all)

    figures = []
    heatmap_fig = create_heatmap_visualization(df, FIGURES_DIR)
    if heatmap_fig is not None:
        figures.append(heatmap_fig)
    lineplot_fig = create_lineplot_visualization(df, FIGURES_DIR)
    if lineplot_fig is not None:
        figures.append(lineplot_fig)
    parallel_fig = create_parallel_coordinates_visualization(df, FIGURES_DIR)
    if parallel_fig is not None:
        figures.append(parallel_fig)

    if not figures:
        print(
            "No figures were generated. Check if the dataset has sufficient diversity."
        )

    return df


if __name__ == "__main__":
    """Main entry point for CLI execution."""
    results_df = run_analysis(force_recompute=True)
    if not results_df.empty:
        plt.show()
