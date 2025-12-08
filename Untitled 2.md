ray.init(
            num_cpus=16,  # or another cap below your physical core count
            runtime_env={
                "env_vars": {
                    "JAX_PLATFORM_NAME": "cpu",
                    "CUDA_VISIBLE_DEVICES": "",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false --xla_cpu_multi_thread_eigen_thread_count=1",
                },
            },
        )