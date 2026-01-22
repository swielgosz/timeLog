IT IS TIME. Let's clean this up. 

We don't need to commit the yaml files (or can do this last)
What did  I do first?
I think I worked on TrainingPhaseVisualizer around first. I was implementing segmentation strategy.

Acceleration metrics were near the last

# Files to get rid of
## train_separatedata.py
Unlike train.py, this uses a distinct dataset for validation, e.g. we have `<dataset>_train` for training and `<dataset>_test` and forces train_val_split to 1.0 so that validation loss is computed only based on the validation dataset. 
Code:
``` python
# import os

# os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["MKL_NUM_THREADS"] = "8"
# os.environ["XLA_FLAGS"] = (
#     "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=8"
# )
import jax.random as jr
from mldsml.config_utils import load_config, make_all_configs
from mldsml.wandb_utils import (
    init_wandb,
    load_dataset,
)

import neuralODE
import wandb
from neuralODE import constants
from neuralODE.data import (
    format_data,
    sample_data,
)
from neuralODE.dynamics import get_dynamics_class
from neuralODE.metrics import (
    AccelerationAngleMetric,
    AccelerationDirectionFlipMetric,
    AccelerationMetric,
    RadialAccelerationSignMetric,
)
from neuralODE.neuralODE import NeuralODE, save, train_model
from neuralODE.normalization import Normalization2BP

neuralODE_path = neuralODE.__path__[0]


def main():
    # default_config = load_config(
    #     f"{neuralODE_path}/config/hifi_2BP_test.yaml",
    # )
    # sweep_config = load_config(
    #     f"{neuralODE_path}/config/hifi_2BP_test.yaml",
    # )
    default_config = load_config(
        f"{neuralODE_path}/config/lofi_2BP.yaml",
    )
    sweep_config = load_config(
        f"{neuralODE_path}/config/lofi_2BP.yaml",
    )
    configs = make_all_configs(default_config, sweep_config)
    config = configs[0]  # Take the first (only) config

    # Explicitly use external train/val datasets (can override via config).
    dataset_name = config.data.dataset_name
    train_dataset_name = f"{dataset_name}_train"
    val_dataset_name = f"{dataset_name}_test"
    config.data.dataset_name = train_dataset_name
    config.data.val_dataset_name = val_dataset_name

    # Use the full training set; validation comes from the separate dataset.
    config.parameters.train_val_split = 1.0

    run = init_wandb(
        config=config,
        group=config.wandb.group,
        mode="online",
        reinit=True,
    )

    # # load the dataset from wandb
    data_dict_train = load_dataset(
        train_dataset_name,
        version="latest",
        project="neuralODEs",
        entity="mlds-lab",
    )
    data_dict_val = load_dataset(
        val_dataset_name,
        version="latest",
        project="neuralODEs",
        entity="mlds-lab",
    )

    data_dict_train = sample_data(data_dict_train, config.parameters.num_trajs)
    data_dict_val = sample_data(data_dict_val, config.parameters.num_trajs)

    # transform the data
    if config.data.problem == "2BP":
        transform = Normalization2BP(
            l_char=constants.RADIUS_EARTH,
            mu=constants.MU_EARTH,
        )
        data_dict_train_normalized = transform.normalize_dataset(data_dict_train)
        data_dict_val_normalized = transform.normalize_dataset(data_dict_val)
    else:
        data_dict_train_normalized = data_dict_train
        data_dict_val_normalized = data_dict_val

    data_dict_train_jnp = format_data(data_dict_train_normalized)
    data_dict_val_jnp = format_data(data_dict_val_normalized)
    num_total_orbits = len(data_dict_train_jnp["y"])
    data_size = data_dict_train_jnp["y"].shape[2]  # state variable dimension

    model = NeuralODE(
        data_size,
        config.parameters.width,
        config.parameters.depth,
        key=jr.PRNGKey(1234),
        config=config,
    )

    base_dynamics = get_dynamics_class(config.data.problem)
    metrics = [
        AccelerationMetric(base_dynamics),
        AccelerationAngleMetric(base_dynamics),
        RadialAccelerationSignMetric(),
        AccelerationDirectionFlipMetric(),
    ]

    model = train_model(
        config,
        model,
        train_data=data_dict_train_jnp,
        val_data=data_dict_val_jnp,
        metrics=metrics,
        seed=5678,
    )

    training_metadata = getattr(config, "training_metadata", {})
    stage_summaries = training_metadata.get("stage_summaries", [])
    final_stage = stage_summaries[-1] if stage_summaries else {}
    segment_ratio_schedule = training_metadata.get("segment_ratio_schedule", [])
    final_segment_ratio = segment_ratio_schedule[-1] if segment_ratio_schedule else None
    num_train_segments = final_stage.get("num_train_segments")
    num_val_segments = final_stage.get("num_val_segments")
    num_total_segments = (
        num_train_segments + num_val_segments
        if num_train_segments is not None and num_val_segments is not None
        else None
    )
    segments_per_orbit = training_metadata.get("final_segments_per_orbit")
    num_train_orbits = training_metadata.get("num_train_orbits")
    if num_train_orbits is None:
        num_train_orbits = (
            num_train_segments * segments_per_orbit
            if segments_per_orbit is not None and num_train_segments is not None
            else None
        )

    wandb.log(
        {
            "segment_ratio": segment_ratio_schedule,
            "segment_ratio_final": final_segment_ratio,
            "num_total_orbits": num_total_orbits,
            "num_train_orbits": num_train_orbits,
            "num_total_segments": num_total_segments,
        },
    )
    save(
        config,
        model,
        train_dataset_name,
        run.id,
        num_total_orbits=num_total_orbits,
        segment_ratio=segment_ratio_schedule,
        segment_ratio_schedule=segment_ratio_schedule,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
```
## list_crashed_runs.py
Lists crashed run ids from a queried W&B group. I was using this to debug why sweeps were crashing, but I believe this was rectified so the script is no longer needed.
``` python
"""
List failed/crashed WandB runs in a group to help spot the run that broke a sweep.

Usage:
    python scripts/training/list_crashed_runs.py

Adjust GROUP/PROJECT/ENTITY below as needed.
"""

import wandb

# Configure here
GROUP = "sweep-2BP-12-7"
PROJECT = "neuralODEs"
ENTITY = "mlds-lab"


def _safe_summary_val(run, key, default=0):
    """Safely pull a numeric summary value without raising on malformed data."""
    try:
        summary = getattr(run, "summary", None)
        if summary is None:
            return default
        val = summary.get(key, default)
        if isinstance(val, (int, float)):
            return val
        return default
    except Exception:  # pylint: disable=broad-except
        return default


def main():
    api = wandb.Api()
    runs = api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={
            "$and": [
                {"group": GROUP},
                {"state": {"$in": ["failed", "crashed"]}},
            ],
        },
    )

    crashed = sorted(
        runs,
        key=lambda r: (
            _safe_summary_val(r, "_timestamp", 0),
            _safe_summary_val(r, "_runtime", 0),
        ),
    )

    if not crashed:
        print("No crashed/failed runs found for the given group.")
        return

    print(f"Crashed/failed runs in group '{GROUP}':\n")
    for r in crashed:
        runtime = _safe_summary_val(r, "_runtime", "n/a")
        ts = _safe_summary_val(r, "_timestamp", "n/a")
        try:
            exit_code = r.summary.get("exit_code", "n/a")
        except Exception:  # pylint: disable=broad-except
            exit_code = "n/a"
        # Config can occasionally be non-dict; guard it.
        cfg = r.config if isinstance(r.config, dict) else {}
        seg = cfg.get("parameters", {}).get("segment_length_strategy") or cfg.get(
            "segment_length_strategy",
        )
        print(
            f"{r.id} | state={r.state} | exit={exit_code} | runtime={runtime} | ts={ts} | segment_length_strategy={seg}",
        )


if __name__ == "__main__":
    main()
```


# Notes
`feature_inputs.py`:
- It provides plotting utilities to visualize feature-layer outputs over time and as aggregate distributions.
- plot_feature_series(...) plots per-feature time series (with optional truth reference and error curves, plus a periapsis marker).
- plot_feature_scatter(...) shows a per-feature value distribution across all samples.
Used in `feature_diagnostics.py` imports and calls plot_feature_series and plot_feature_scatter, and `postprocess.py` calls export_training_feature_snapshots and run_feature_dynamics_capture

# neuralODE.py
`get_output_features`
``` python
    def get_output_features(self, y, *, post_activation=False):
        _, mlp_out, outputs = self._evaluate_layers(y)
        if post_activation:
            # Return the post-activation acceleration components (without velocities).
            if self.output_layer_name in {
                "mlp_4D",
                "mlp_4D_scaled",
                "mlp_4D_unit",
                "mlp_4D_unit_scaled",
            }:
                r_mag = jnp.abs(mlp_out[0:1])
                r_dir = mlp_out[1:4]
                return jnp.concatenate((r_mag, r_dir), axis=0)
            if self.output_layer_name == "mlp_4D_signed":
                r_mag = mlp_out[0:1]
                r_dir = mlp_out[1:4]
                return jnp.concatenate((r_mag, r_dir), axis=0)
            if self.output_layer_name == "mlp_4D_activation":
                r_mag = jnn.sigmoid(mlp_out[0:1])
                r_dir = jnn.tanh(mlp_out[1:4])
                return jnp.concatenate((r_mag, r_dir), axis=0)
            # For other output layers, fall back to the actual acceleration part of outputs.
            return outputs[3:]
        if self.output_layer_name in {
            "mlp_4D",
            "mlp_4D_scaled",
            "mlp_4D_unit",
            "mlp_4D_unit_scaled",
        }:
            r_mag = jnp.abs(mlp_out[0:1])
            r_dir = mlp_out[1:4]
            return jnp.concatenate((r_mag, r_dir), axis=0)
        if self.output_layer_name == "mlp_4D_signed":
            r_mag = mlp_out[0:1]
            r_dir = mlp_out[1:4]
            return jnp.concatenate((r_mag, r_dir), axis=0)
        return mlp_out
```

Let's deal with the segmentation curriculum first

Laebels aren't handled the same between output_layers and feature_layers?

# 1/22
Current state of neuralODE.py:
```python
# %%
import io
import json
import os
from datetime import datetime

import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.nn as jnn

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import wandb
from jax import Array  # JAX array type
from mldsml.config_utils import DotDict
from tqdm import tqdm

from neuralODE.data import segment_data, split_data
from neuralODE.feature_layers import get_feature_layer
from neuralODE.losses import (  # Import custom loss functions
    get_loss_function,
    percent_error_loss,
    percent_error_plus_nmse_components,
)
from neuralODE.output_layers import get_output_layer  # Import custom output layers
from neuralODE.parameter_utils import normalize_strategy_parameters
from neuralODE.visualizers.training_phase_utils import (
    build_dataset_preview,
    compute_min_valid_segment_length,
    select_reference_trajectory,
)

# SEED = 42
# key = jr.PRNGKey(SEED)
num_extra_features = 0


class Func(eqx.Module):
    mlp: eqx.nn.MLP
    feature_layer: callable = eqx.field(static=True)
    output_layer: callable = eqx.field(static=True)
    planar_constraint: bool = eqx.field(static=True, default=False)
    scalar: float = eqx.field(static=True, default=1.0)

    def __init__(self, data_size, width, depth, config, *, key, **kwargs):
        super().__init__(**kwargs)
        self.feature_layer, in_size = get_feature_layer(
            config.parameters.get("feature_layer", "rmag_v_and_direction"),
        )
        self.output_layer, out_size = get_output_layer(
            config.parameters.get("output_layer", "std_output_6"),
        )
        self.planar_constraint = config.parameters.get("planar_constraint", False)
        self.scalar = config.parameters.get("scalar", 1.0)

        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=out_size,
            width_size=width,
            depth=depth,
            activation=getattr(jnn, config.parameters.activation),
            key=key,
        )

    def evaluate_layers(self, y):
        features = self.feature_layer(y)
        mlp_out = self.mlp(features)
        outputs = self.output_layer(mlp_out, y, self.scalar)
        return features, mlp_out, outputs

    def __call__(self, t, y, args=None):
        _, _, outputs = self.evaluate_layers(y)

        if self.planar_constraint:
            outputs *= jnp.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0])

        return outputs


class NeuralODE(eqx.Module):
    func: Func
    rtol: float = eqx.field()
    atol: float = eqx.field()

    def __init__(self, data_size, width, depth, config, *, key, **kwargs):
        super().__init__(**kwargs)

        # Initialize NeuralODE with the first strategy as default
        self.func = Func(data_size, width, depth, key=key, config=config)
        self.rtol = config.parameters.rtol
        self.atol = config.parameters.atol

    def __call__(self, ts, y0):
        solution = None
        try:
            solution = diffrax.diffeqsolve(
                diffrax.ODETerm(self.func),
                diffrax.Tsit5(),
                t0=ts[0],
                t1=ts[-1],
                dt0=ts[1] - ts[0],
                y0=y0,
                stepsize_controller=diffrax.PIDController(
                    rtol=self.rtol,
                    atol=self.atol,
                ),
                saveat=diffrax.SaveAt(ts=ts),
            )
            return solution.ys
        except Exception as e:
            print(f"Integration failed: {e}")
            if solution is not None:
                return solution.ys


def get_max_length(data):
    """
    Get the maximum length of the trajectories in the dataset.
    """
    max_length = 0
    for trajectory in data["y"]:
        if len(trajectory[1]) > max_length:
            max_length = len(trajectory[1])
    return max_length


def get_batch(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)

    indices = jnp.arange(dataset_size)

    while True:
        # Shuffle indices at the start of each epoch
        key, subkey = jr.split(key)
        perm = jr.permutation(subkey, indices)

        # Yield batches for this epoch
        start = 0
        while start < dataset_size:
            end = min(start + batch_size, dataset_size)
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end


def assert_data_format(data):
    # Assert that each object has three dimensions
    # and are of correct JAX type
    assert isinstance(data, Array), "data must be a JAX array"
    assert data.ndim == 3, "data must have three dimensions"


def clone_model(model):
    """Deep copy an Equinox model via serialization."""
    buffer = io.BytesIO()
    eqx.tree_serialise_leaves(buffer, model)
    buffer.seek(0)
    return eqx.tree_deserialise_leaves(buffer, model)


def train_model(
    config,
    model,
    train_data=None,
    val_data=None,
    *,
    dataset=None,
    metrics=None,
    seed=5678,
    eval_loss_frequency=100,
    feature_capture=None,
):
    batch_size = config.parameters.get("batch_size", 10)

    # Normalize strategy parameters using the dedicated function
    config = normalize_strategy_parameters(config)
    lr_strategy = config.parameters.lr_strategy
    steps_strategy = config.parameters.steps_strategy
    length_strategy = config.parameters.length_strategy
    segment_strategy = list(config.parameters.segment_length_strategy)

    using_dataset = dataset is not None
    if not using_dataset and (train_data is None or val_data is None):
        raise ValueError(
            "train_model requires either `dataset` or both `train_data` and `val_data`.",
        )

    max_segment_length = max(segment_strategy) if segment_strategy else 1

    reference_trajectory = None
    min_valid_length = None
    dataset_preview = None
    train_orbit_data = None
    val_orbit_data = None
    num_train_orbits_split = None
    if using_dataset:
        mask_np = np.asarray(dataset["mask"])
        min_valid_length = compute_min_valid_segment_length(mask_np)
        segment_strategy = [min(value, min_valid_length) for value in segment_strategy]
        dataset_preview = build_dataset_preview(dataset, min_valid_length)
        ref_index = getattr(config.parameters, "reference_orbit_index", None)
        reference_trajectory = select_reference_trajectory(
            dataset,
            min_valid_length,
            config,
            preferred_index=ref_index,
        )
        ref_elements = getattr(config.parameters, "reference_orbital_elements", None)
        if reference_trajectory is not None and ref_elements is not None:
            reference_trajectory["orbital_elements"] = {
                key: float(value) if isinstance(value, (int, float)) else value
                for key, value in dict(ref_elements).items()
            }
        train_orbit_data, val_orbit_data = split_data(
            dataset,
            config,
            shuffle=True,
        )
        num_train_orbits_split = int(train_orbit_data["t"].shape[0])
    else:
        if train_data is None or val_data is None:
            raise ValueError("Both train_data and val_data must be provided.")
        min_valid_train = compute_min_valid_segment_length(train_data["mask"])
        min_valid_val = compute_min_valid_segment_length(val_data["mask"])
        min_valid_length = min(min_valid_train, min_valid_val)
        segment_strategy = [min(value, min_valid_length) for value in segment_strategy]

    # Cache segmented datasets by segment length
    segment_cache_train = {}
    segment_cache_val = {}

    if min_valid_length is None:
        denominator = float(max_segment_length) if max_segment_length else 1.0
    else:
        denominator = float(min_valid_length)
    segment_ratio_schedule = [
        100.0 * float(length) / denominator for length in segment_strategy
    ]

    stage_summaries = []
    segments_per_orbit_schedule = []
    phase_fidelity_snapshots = []
    training_metadata = DotDict(getattr(config, "training_metadata", {}))
    training_metadata.segment_schedule = list(segment_strategy)
    training_metadata.segment_ratio_schedule = list(segment_ratio_schedule)
    overall_best_value = np.inf
    overall_best_metric = None
    overall_best_phase = None
    overall_best_model = None
    history_rows = []
    global_step = 0

    # Get loss function
    loss_name = config.parameters.loss_fcn
    custom_loss = get_loss_function(loss_name)

    loss_supports_components = getattr(custom_loss, "supports_components", False)

    def _loss_wrapper(model, ti, yi, mask_i):
        if loss_supports_components:
            return custom_loss(model, ti, yi, mask_i, return_components=True)
        return custom_loss(model, ti, yi, mask_i)

    grad_loss = eqx.filter_value_and_grad(
        _loss_wrapper,
        has_aux=loss_supports_components,
    )

    @eqx.filter_jit
    def make_step(ti, yi, model, mask_i, opt_state):
        if loss_supports_components:
            (loss, aux_components), grads = grad_loss(model, ti, yi, mask_i)
        else:
            loss, grads = grad_loss(model, ti, yi, mask_i)
            aux_components = None
        params = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optim.update(
            grads,
            opt_state,
            params,
        )
        model = eqx.apply_updates(model, updates)
        return loss, aux_components, model, opt_state

    capture_callback = None
    capture_frequency = None
    capture_limit = None
    if feature_capture:
        capture_callback = feature_capture.get("callback")
        capture_frequency = feature_capture.get("frequency")
        capture_limit = feature_capture.get("max_snapshots")
    capture_taken = 0

    total_steps = int(np.sum(steps_strategy))
    progress_bar = tqdm(
        total=total_steps,
        desc="Training progress",
        unit="step",
    )

    base_key = jr.PRNGKey(seed)

    for phase, (lr, steps, length_fracs, segment_length) in enumerate(
        zip(
            lr_strategy,
            steps_strategy,
            length_strategy,
            segment_strategy,
        ),
    ):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        coverage_ratio = segment_ratio_schedule[phase]

        if using_dataset:
            if train_orbit_data is None or val_orbit_data is None:
                raise ValueError("Orbit-level train/val split failed.")
            if segment_length not in segment_cache_train:
                segmented_train = segment_data(train_orbit_data, segment_length)
                if len(segmented_train["y"]) == 0:
                    raise ValueError(
                        f"Segment length {segment_length} produced no usable trajectory segments.",
                    )
                segment_cache_train[segment_length] = segmented_train
            if segment_length not in segment_cache_val:
                segmented_val = segment_data(val_orbit_data, segment_length)
                segment_cache_val[segment_length] = segmented_val
            train_data_stage = segment_cache_train[segment_length]
            val_data_stage = segment_cache_val[segment_length]
            total_segments = int(train_data_stage["t"].shape[0])
            if total_segments <= 0:
                raise ValueError(
                    f"No segments available for segment length {segment_length}.",
                )
            num_orbits = max(1, int(train_orbit_data["t"].shape[0]))
            segments_per_orbit = float(total_segments) / float(num_orbits)
            segments_per_orbit_schedule.append(segments_per_orbit)
        else:
            if segment_length not in segment_cache_train:
                segmented_train = segment_data(train_data, segment_length)
                if len(segmented_train["t"]) == 0:
                    raise ValueError(
                        f"Segment length {segment_length} produced no usable training segments.",
                    )
                segment_cache_train[segment_length] = segmented_train
            if segment_length not in segment_cache_val:
                segmented_val = segment_data(val_data, segment_length)
                segment_cache_val[segment_length] = segmented_val
            train_data_stage = segment_cache_train[segment_length]
            val_data_stage = segment_cache_val[segment_length]
            segments_per_orbit = None

        ts = train_data_stage["t"]
        ys = train_data_stage["y"]
        mask_ys = train_data_stage["mask"]

        assert_data_format(ys)

        _, length_size, _ = ys.shape

        # length strategy subset
        start_idx = int(np.floor(length_size * length_fracs[0]))
        end_idx = int(np.ceil(length_size * length_fracs[1]))
        start_idx = max(start_idx, 0)
        end_idx = min(end_idx, length_size)
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, length_size)

        if end_idx <= start_idx:
            raise ValueError(
                f"Invalid length_strategy slice ({length_fracs}) produced empty window.",
            )

        _ts = ts[:, start_idx:end_idx]
        _ys = ys[:, start_idx:end_idx, :]
        _mask_ys = mask_ys[:, start_idx:end_idx]

        # Validation slices only if we actually have validation data
        has_val_data = (
            val_data_stage is not None
            and "t" in val_data_stage
            and getattr(val_data_stage["t"], "ndim", 0) >= 2
            and val_data_stage["t"].shape[0] > 0
        )
        if has_val_data:
            ts_val = val_data_stage["t"]
            ys_val = val_data_stage["y"]
            mask_ys_val = val_data_stage["mask"]
            _ts_val = ts_val[:, start_idx:end_idx]
            _ys_val = ys_val[:, start_idx:end_idx, :]
            _mask_ys_val = mask_ys_val[:, start_idx:end_idx]
        else:
            _ts_val = _ys_val = _mask_ys_val = None

        batch_key = jr.fold_in(base_key, phase)
        batch_iter = get_batch(
            (_ts, _ys, _mask_ys),
            batch_size,
            key=batch_key,
        )

        for step in range(steps):
            component_logs = {}
            val_metrics = {}
            ti, yi, mask_i = next(batch_iter)
            loss, aux_components, model, opt_state = make_step(
                ti,
                yi,
                model,
                mask_i,
                opt_state,
            )
            if (
                capture_callback
                and (capture_frequency is None or global_step % capture_frequency == 0)
                and (capture_limit is None or capture_taken < capture_limit)
            ):
                try:
                    capture_callback(
                        model=model,
                        times=ti,
                        states=yi,
                        mask=mask_i,
                        phase=phase,
                        step=step,
                        global_step=global_step,
                    )
                    capture_taken += 1
                except Exception as exc:
                    print(f"Training feature capture failed: {exc}")
            percent_error = percent_error_loss(
                model,
                ti,
                yi,
                mask_i,
            )
            if loss_name == "percent_error_plus_nmse":
                mpe_comp, rmse_comp = percent_error_plus_nmse_components(
                    model,
                    ti,
                    yi,
                    mask_i,
                )
                component_logs["loss_component_mpe"] = float(mpe_comp)
                component_logs["loss_component_rmse"] = float(rmse_comp)
            if loss_supports_components and aux_components is not None:
                aux_host = {
                    key: float(jax.device_get(value))
                    for key, value in aux_components.items()
                }
                component_logs["train_loss_component_percent_error"] = aux_host.get(
                    "percent_error_component",
                )
                component_logs["train_loss_component_attraction"] = aux_host.get(
                    "attraction_component",
                )

            # Calculate validation loss at the specified frequency
            val_step = step % eval_loss_frequency == 0
            if has_val_data and (val_step or step == steps - 1):
                if loss_supports_components:
                    val_loss_value, val_aux_components = custom_loss(
                        model,
                        _ts_val,
                        _ys_val,
                        _mask_ys_val,
                        return_components=True,
                    )
                else:
                    val_loss_value = custom_loss(model, _ts_val, _ys_val, _mask_ys_val)
                    val_aux_components = None
                val_percent_error_value = percent_error_loss(
                    model,
                    _ts_val,
                    _ys_val,
                    _mask_ys_val,
                )
                val_metrics["val_loss"] = float(val_loss_value)
                val_metrics["val_percent_error"] = float(val_percent_error_value)
                if loss_name == "percent_error_plus_nmse":
                    (
                        val_mpe_comp,
                        val_rmse_comp,
                    ) = percent_error_plus_nmse_components(
                        model,
                        _ts_val,
                        _ys_val,
                        _mask_ys_val,
                    )
                    val_metrics["val_loss_component_mpe"] = float(val_mpe_comp)
                    val_metrics["val_loss_component_rmse"] = float(val_rmse_comp)
                if loss_supports_components and val_aux_components is not None:
                    aux_host = {
                        key: float(jax.device_get(value))
                        for key, value in val_aux_components.items()
                    }
                    val_metrics["val_loss_component_percent_error"] = aux_host.get(
                        "percent_error_component",
                    )
                    val_metrics["val_loss_component_attraction"] = aux_host.get(
                        "attraction_component",
                    )
            else:
                val_loss_value = None
                val_percent_error_value = None

            metric_values = {}
            for metric in metrics or []:
                metric_value = metric(model, _ts, _ys, _mask_ys)
                metric_values[metric.name] = float(metric_value)

            monitored_value = None
            monitored_metric_name = None
            use_for_best = False
            if has_val_data and (val_step or step == steps - 1):
                if val_loss_value is not None:
                    monitored_value = float(val_loss_value)
                    monitored_metric_name = "val_loss"
                    use_for_best = True
                elif val_percent_error_value is not None:
                    monitored_value = float(val_percent_error_value)
                    monitored_metric_name = "val_percent_error"
                    use_for_best = True
            if monitored_value is None:
                monitored_value = float(loss)
                monitored_metric_name = "train_loss"
                use_for_best = not has_val_data

            if use_for_best and monitored_value < overall_best_value:
                overall_best_value = monitored_value
                overall_best_metric = monitored_metric_name
                overall_best_phase = phase
                overall_best_model = clone_model(model)

            log_payload = {
                "train_loss": float(loss),
                "percent_error": float(percent_error),
                "curriculum_phase": phase,
                "segment_length": segment_length,
                "segment_ratio": coverage_ratio,
            }
            if has_val_data:
                log_payload["val_loss"] = val_metrics.get("val_loss")
                log_payload["val_percent_error"] = val_metrics.get("val_percent_error")
                log_payload["val_loss_component_mpe"] = val_metrics.get(
                    "val_loss_component_mpe",
                )
                log_payload["val_loss_component_rmse"] = val_metrics.get(
                    "val_loss_component_rmse",
                )
            log_payload.update(metric_values)
            log_payload.update(component_logs)

            wandb.log(log_payload)
            history_rows.append(
                {
                    "step": global_step,
                    **log_payload,
                },
            )
            global_step += 1
            progress_bar.update(1)

        stage_summary = {
            "phase": phase,
            "segment_length": int(segment_length),
            "num_train_segments": int(train_data_stage["t"].shape[0]),
            "num_val_segments": int(val_data_stage["t"].shape[0]),
            "segment_ratio": float(coverage_ratio),
        }
        if segments_per_orbit is not None:
            stage_summary["segments_per_orbit"] = float(segments_per_orbit)
        stage_summaries.append(stage_summary)
        if using_dataset and reference_trajectory is not None:
            ref_ts = reference_trajectory["ts"]
            ref_y = reference_trajectory["y"]
            ref_y0 = reference_trajectory["y0"]
            try:
                preds = model(ref_ts, ref_y0)
                acceleration_error = None
                for metric in metrics or []:
                    if getattr(metric, "name", None) == "acceleration_error":
                        ts_batch = ref_ts[None, :]
                        ys_batch = ref_y[None, :, :]
                        mask_batch = jnp.ones(
                            (1, ref_ts.shape[0]),
                            dtype=bool,
                        )
                        acceleration_error = float(
                            metric(model, ts_batch, ys_batch, mask_batch),
                        )
                        break
                phase_fidelity_snapshots.append(
                    {
                        "phase": int(phase),
                        "segment_length": int(segment_length),
                        "ts": np.asarray(ref_ts).tolist(),
                        "y_true": np.asarray(ref_y).tolist(),
                        "y_pred": np.asarray(preds).tolist(),
                        "segment_ratio": float(coverage_ratio),
                        "acceleration_error": acceleration_error,
                    },
                )
            except Exception as exc:
                print(
                    f"Failed to capture fidelity snapshot for phase {phase}: {exc}",
                )

    progress_bar.close()

    if overall_best_model is not None:
        model = overall_best_model
        training_metadata.best_model = {
            "metric_name": overall_best_metric,
            "metric_value": float(overall_best_value),
            "phase": int(overall_best_phase)
            if overall_best_phase is not None
            else None,
        }

    training_metadata.stage_summaries = stage_summaries
    if stage_summaries:
        final_summary = stage_summaries[-1]
        training_metadata.final_segment_length = final_summary["segment_length"]
        training_metadata.final_num_train_segments = final_summary["num_train_segments"]
        training_metadata.final_num_val_segments = final_summary["num_val_segments"]
        training_metadata.final_segment_ratio = final_summary["segment_ratio"]
        if "segments_per_orbit" in final_summary:
            training_metadata.final_segments_per_orbit = final_summary[
                "segments_per_orbit"
            ]
    if segments_per_orbit_schedule:
        training_metadata.segments_per_orbit_schedule = segments_per_orbit_schedule
    if phase_fidelity_snapshots:
        training_metadata.phase_fidelity_snapshots = phase_fidelity_snapshots
    if reference_trajectory is not None:
        reference_payload = {
            "ts": np.asarray(reference_trajectory["ts"]).tolist(),
            "y": np.asarray(reference_trajectory["y"]).tolist(),
            "y0": np.asarray(reference_trajectory["y0"]).tolist(),
            "min_valid_length": int(reference_trajectory["min_valid_length"]),
            "orbit_index": int(reference_trajectory["orbit_index"]),
        }
        if "orbital_elements" in reference_trajectory:
            reference_payload["orbital_elements"] = dict(
                reference_trajectory["orbital_elements"],
            )
        training_metadata.reference_trajectory = reference_payload
    if dataset_preview is not None:
        training_metadata.dataset_preview = dataset_preview
    if num_train_orbits_split is not None:
        training_metadata.num_train_orbits = int(num_train_orbits_split)
        if val_orbit_data is not None:
            training_metadata.num_val_orbits = int(val_orbit_data["t"].shape[0])
    elif train_data is not None:
        training_metadata.num_train_orbits = int(train_data["t"].shape[0])
        if val_data is not None:
            training_metadata.num_val_orbits = int(val_data["t"].shape[0])
    training_metadata.logged_history = history_rows
    config.training_metadata = training_metadata

    return model


def save(config, model, dataset_name, model_name=None, **kwargs):
    if model_name is None:
        model_name = f"model-{model.id}"

    os.makedirs("files/models/", exist_ok=True)
    filename = f"files/models/{model_name}.eqx"
    config["input_shape"] = model.func.mlp.in_size
    with open(filename, "wb") as f:
        hyperparam_str = json.dumps(config.copy())
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

    # Build metadata dictionary starting with required fields
    metadata = {
        "dataset_name": dataset_name,
    }

    # Add optional fields if they exist in kwargs
    if "num_total_orbits" in kwargs:
        metadata["num_total_orbits"] = kwargs["num_total_orbits"]
        # Only add derived field if the source field exists
        if hasattr(config.parameters, "train_val_split"):
            metadata["num_train_orbits"] = (
                kwargs["num_total_orbits"] * config.parameters.train_val_split
            )

    if "segment_ratio" in kwargs:
        metadata["segment_ratio"] = kwargs["segment_ratio"]
    if "segment_ratio_schedule" in kwargs:
        metadata["segment_ratio_schedule"] = kwargs["segment_ratio_schedule"]

    # Log the model to wandb
    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        metadata=metadata,
    )
    artifact.add_file(f"files/models/{model_name}.eqx")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb.log_artifact(artifact, aliases=["latest", f"run_{timestamp}"])


def load(filename):
    try:
        return load_model_wandb(filename)

    except Exception as e:
        print(f"Loading from wandb failed with exception {e}. \n Trying local file")
        with open(filename, "rb") as f:
            config = json.loads(f.readline().decode())
            input_shape = config.input_shape
            model = NeuralODE(config, input_shape=input_shape, key=jr.PRNGKey(0))
            model = eqx.tree_deserialise_leaves(f, model)

        return config, model


def load_model_wandb(
    filename,
    project="neuralODEs",
    entity="mlds-lab",
    version="latest",
) -> tuple[DotDict, NeuralODE, wandb.apis.public.Run]:
    api = wandb.Api()

    model_name = filename.split("/")[-1].split(".")[0]

    if ":" in model_name:
        version = model_name.split(":")[-1]
        model_name = model_name.split(":")[0]

    # Retrieve the artifact
    artifact_name = f"{entity}/{project}/{model_name}:{version}"
    artifact = api.artifact(artifact_name, type="model")
    artifact_dir = artifact.download()

    # Get the run associated with the artifact
    artifact_run = artifact.logged_by()

    # Load the model from the artifact
    model_file_path = f"{artifact_dir}/{model_name}.eqx"
    with open(model_file_path, "rb") as f:
        config = json.loads(f.readline().decode())
        config = DotDict(config)  # Convert to DotDict for easier access

        # Copy parameters to top level
        # HACK
        for key, value in config.parameters.items():
            config[key] = value

        # HACK: input_shape is not in the config, so we need to set it manually
        data_shape = 6
        model = NeuralODE(
            data_size=data_shape,
            width=config.width,
            depth=config.depth,
            config=config,
            key=jr.PRNGKey(0),
        )
        model = eqx.tree_deserialise_leaves(f, model)
    return config, model, artifact_run


def fetch_run_data(run_id, project="neuralODEs"):
    """Fetch run data from wandb"""
    api = wandb.Api()
    run = api.run(f"mlds-lab/{project}/{run_id}")

    # Get the history data with samples=None to get ALL data points
    history = run.history(samples=9500, pandas=True)

    # Print available columns and row count for debugging
    print(f"Available columns: {list(history.columns)}")
    print(f"Total rows fetched: {len(history)}")

    return history
```