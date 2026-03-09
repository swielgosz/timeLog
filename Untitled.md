import os
import pickle

import jax.numpy as jnp
import matplotlib.pyplot as plt
import mldsml
import numpy as np
import ray
from mldsml.wandb_utils import (
    load_dataset,
    query_runs_in_group,
)
from tables import make_tables

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
from neuralODE.visualizers.IntegrationVisualizer import ModIntegrationVisualizer

neuralODE_path = neuralODE.__path__[0]

np.random.seed(42)  # For reproducibility
# use style sheet

mldsml_path = mldsml.__path__[0]
plt.style.use(f"{mldsml_path}/styles/dark_scientific.mplstyle")
ALL_DATASETS = ["single_TBP_planar", "simple_TBP_planar", "complex_TBP_planar"]


def evaluate_acceleration_residuals(data_set, model, config, run_id):
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
    # log_metric_to_existing_run(
    #     f"mean_acc_err_{data_set}",
    #     acc_error,
    #     run_id,
    # )
    return acc_error


def select_best_runs_by_train_dataset(runs):
    """
    Select the lowest-acceleration-error run per training dataset.

    Selection metric is acceleration error evaluated on the run's own
    training dataset.
    """
    best_by_dataset = {}
    for run in runs:
        try:
            config, model, run_obj = load_model_wandb(
                run.id,
                project="neuralODEs",
                entity="mlds-lab",
            )
            train_dataset = str(config.data.dataset_name)
            acc_err = float(
                evaluate_acceleration_residuals(
                    train_dataset,
                    model,
                    config,
                    run_obj.id,
                ),
            )
        except Exception as exc:
            print(f"[warn] skipping run {getattr(run, 'id', 'unknown')}: {exc}")
            continue

        current = best_by_dataset.get(train_dataset)
        if current is None or acc_err < current["score"]:
            best_by_dataset[train_dataset] = {
                "score": acc_err,
                "config": config,
                "model": model,
                "run_obj": run_obj,
            }

    selected = [best_by_dataset[key] for key in sorted(best_by_dataset.keys())]
    print(
        f"Selected {len(selected)} best runs "
        f"(one per training dataset): {[str(s['config'].data.dataset_name) for s in selected]}",
    )
    for sel in selected:
        print(
            "  "
            f"{sel['run_obj'].id} "
            f"dataset={sel['config'].data.dataset_name} "
            f"train_acc_err={sel['score']:.6f}",
        )
    return selected


def random_sample_indices(data_dict, num_traj):
    """
    Randomly sample indices from the dataset.
    """
    num_samples = data_dict["y"].shape[0]
    if num_traj > num_samples:
        raise ValueError(
            "num_traj cannot be greater than the number of samples in the dataset.",
        )
    return np.random.choice(num_samples, num_traj, replace=False)


def plot_integrated_orbits(test_dataset, model, config, run_id):
    # Plot the integrated orbits for each of the datasets via some form of integration visualizer
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
    data_idx = {
        "single_TBP_planar": np.array([0]),
        "simple_TBP_planar": np.array([0, 10, 20]),
        "complex_TBP_planar": np.array([10, 50, 90]),
    }

    data_dict_jnp = format_data(data_dict)
    indices = data_idx.get(test_dataset)
    ICs = data_dict_jnp["y"][indices, 0, :]
    times = data_dict_jnp["t"][indices]

    # filter the times for the last element in each row that isn't a nan
    t0_idx_list = [np.where(~np.isnan(times[i]))[0][0] for i in range(len(times))]
    tf_idx_list = [np.where(~np.isnan(times[i]))[0][-1] for i in range(len(times))]

    tf_list = jnp.array([times[i][tf_idx_list[i]] for i in range(len(times))])
    dt_list = jnp.array(
        [(tf_list[i] - times[i][t0_idx_list[i]]) / 500 for i in range(len(times))],
    )

    experiment = IntegrationExperiment(
        model=model,
        initial_conditions=ICs,
        tf=tf_list,
        dt=dt_list,
        true_dynamics=get_dynamics_class(config.data.problem),
    )
    experiment.run()
    vis = ModIntegrationVisualizer(experiment)
    fig = vis.plot()
    fig_small = vis.plot_small()
    return fig, fig_small
    # upload_fig_to_existing_run(f"IntegratedOrbits_{test_dataset}", fig, run_id)


@ray.remote(num_cpus=4, num_gpus=0)
def evaluate_model(model, config, run_id):
    train_data_set = config.data.dataset_name

    # Get average acceleration error over trajectories

    results = []
    for data_set in ALL_DATASETS:
        value = evaluate_acceleration_residuals(data_set, model, config, run_id)
        results.append(
            {
                "train_data_set": train_data_set,
                "test_dataset": data_set,
                "mean_acc_err": value,
            },
        )

    for dataset in ALL_DATASETS:
        try:
            fig, fig_small = plot_integrated_orbits(dataset, model, config, run_id)

            fig_path = f"{neuralODE_path}/../files/figures/IntegratedOrbits_{train_data_set}_{dataset}.pdf"
            fig_small_path = f"{neuralODE_path}/../files/figures/IntegratedOrbits_{train_data_set}_{dataset}_small.pdf"
            # save the figures
            plt.figure(fig.number)
            plt.savefig(fig_path)
            plt.figure(fig_small.number)
            plt.savefig(fig_small_path)
            plt.close("all")  # Close all figures to free memory
        except Exception as e:
            raise e
            print(f"Error plotting integrated orbits for {dataset}: {e}")
            fig_path, fig_small_path = "example-image-a", "example-image-a"

        results.append(
            {
                "train_data_set": train_data_set,
                "test_dataset": dataset,
                "integrated_orbits": fig_path,
                "integrated_orbits_small": fig_small_path,
            },
        )

    # for LO_dataset in LO_datasets:
    #     plot_residuals(LO_dataset, model, config, run_id)

    return results


def main():
    os.makedirs(f"{neuralODE_path}/../files/figures", exist_ok=True)
    os.makedirs(f"{neuralODE_path}/../files/tables/", exist_ok=True)

    runs = query_runs_in_group("conference-2BP-redo-v6")

    selected_runs = select_best_runs_by_train_dataset(runs)
    if not selected_runs:
        raise RuntimeError("No runs selected for evaluation.")

    # plot_integrated_orbits("single_TBP_planar", None, None, None)
    ray.init()
    results_all = []
    futures_all = []
    for selected in selected_runs:
        config = selected["config"]
        model = selected["model"]
        run_id = selected["run_obj"]
        # Evaluate the model
        future = evaluate_model.remote(model, config, run_id.id)
        futures_all.append(future)

    for future in futures_all:
        results = ray.get(future)

        # Collect results from each run
        results_all.extend(results)

    os.makedirs(f"{neuralODE_path}/../files/results", exist_ok=True)
    with open(
        f"{neuralODE_path}/../files/results/2BP_results.pkl",
        "wb",
    ) as f:
        pickle.dump(results_all, f)

    make_tables(results_all)


if __name__ == "__main__":
    results = main()
