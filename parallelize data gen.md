``` python
import multiprocessing as mp
import os
from functools import partial

import numpy as np

from neuralODE import constants

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"


def _worker_init(threads_per_proc: int):
    # Must be set before any JAX import in this process
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")  # legacy alias
    os.environ.setdefault("JAX_DISABLE_JAX_PLUGIN_DISCOVERY", "true")

    # Set the correct XLA threading flags
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_cpu_multi_thread_eigen=true"
    # Use the correct flag syntax for thread count
    xla_flags += f" --xla_force_host_platform_device_count={threads_per_proc}"
    os.environ["XLA_FLAGS"] = xla_flags

    # Avoid BLAS/OpenMP oversubscription
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, str(threads_per_proc))


def generate_orbit_batch(params, dataset_name, batch_idx, batch_size):
    # Import after spawn + env setup (inside worker)
    from sampling_2BP import generate_datasets

    batch_params = params.copy()
    batch_params["num_trajs"] = batch_size
    batch_name = f"{dataset_name}_batch_{batch_idx}"
    data_dict = generate_datasets(batch_params, batch_name)

    # (Optional sanity check while debugging)
    # import jax
    # assert all(d.platform == "cpu" for d in jax.devices())

    return data_dict, batch_idx


def parallel_generate_datasets(params, dataset_name, n_processes=None):
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 2)

    total_trajs = int(params["num_trajs"])
    batch_size = max(1, total_trajs // n_processes)
    n_batches = (total_trajs + batch_size - 1) // batch_size

    print(f"Using {n_processes} processes")
    print(
        f"Generating {total_trajs} trajectories in {n_batches} batches of ~{batch_size}",
    )

    cpu_total = os.cpu_count() or 32
    threads_per_proc = max(1, cpu_total // n_processes)

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_processes,
        initializer=_worker_init,
        initargs=(threads_per_proc,),
        maxtasksperchild=1,
    ) as pool:
        batch_func = partial(generate_orbit_batch, params, dataset_name)
        batch_indices = range(n_batches)
        batch_sizes = [
            batch_size
            if i < n_batches - 1 or total_trajs % batch_size == 0
            else total_trajs % batch_size
            for i in batch_indices
        ]
        results = pool.starmap(batch_func, zip(batch_indices, batch_sizes))

    combined_data_dict = None
    for data_dict, _ in results:
        if combined_data_dict is None:
            combined_data_dict = {k: v.copy() for k, v in data_dict.items()}
        else:
            for k in combined_data_dict:
                if isinstance(combined_data_dict[k], np.ndarray):
                    combined_data_dict[k] = np.concatenate(
                        [combined_data_dict[k], data_dict[k]],
                        axis=0,
                    )

    from sampling_2BP import save_dataset_to_wandb

    save_dataset_to_wandb(combined_data_dict, dataset_name)
    return combined_data_dict


if __name__ == "__main__":
    # Set spawn globally as early as possible
    mp.set_start_method("spawn", force=True)

    N_PROCESSES = 24  # e.g., on a 96-core machine

    single_complex_params = {
        "a_range": [constants.RADIUS_LEO + 12000, constants.RADIUS_LEO + 12000],
        "e_range": [0.5, 0.5],
    }

    simple_params = {
        "a_range": [constants.RADIUS_LEO, constants.RADIUS_LEO + 2000],
        "e_range": [0, 0.1],
    }
    complex_params = {
        "a_range": [constants.RADIUS_EARTH + 200, constants.RADIUS_GEO],
        "e_range": [0, 0.99],
    }
    planar_params = {
        "i_range": np.radians([0, 0]),
        "raan_range": np.radians([0, 0]),
    }
    non_planar_params = {
        "i_range": np.radians([0, 180]),
        "raan_range": np.radians([0, 180]),
    }
    planet_params = {
        "body_radius": constants.RADIUS_EARTH,
        "mu": constants.MU_EARTH,
    }

    TBP_complex_planar_params = {
        **complex_params,
        **planar_params,
        "w_range": np.radians([0, 180]),
        "nu_range": np.radians([0, 360]),
        **planet_params,
        "num_trajs": 1024,
        "num_points": 360,
    }

    complex_TBP_planar_data_dict = parallel_generate_datasets(
        TBP_complex_planar_params,
        "complex_TBP_planar_1024",
        N_PROCESSES,
    )

    print("Dataset generation complete!")
```