IT IS TIME. Let's clean this up. 

We don't need to commit the yaml files (or can do this last)
What did  I do first?
I think I worked on TrainingPhaseVisualizer around f

# Files to get rid of
## `train_separatedata.py`
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
## 