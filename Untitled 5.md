```
/usr/local/python/3.11.13/lib/python3.11/site-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/python/3.11.13/lib/python3.11/site-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
ERROR:2025-12-09 17:47:15,015:jax._src.xla_bridge:487: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
Traceback (most recent call last):
  File "/usr/local/python/3.11.13/lib/python3.11/site-packages/jax/_src/xla_bridge.py", line 485, in discover_pjrt_plugins
    plugin_module.initialize()
  File "/usr/local/python/3.11.13/lib/python3.11/site-packages/jax_plugins/xla_cuda12/__init__.py", line 328, in initialize
    _check_cuda_versions(raise_on_first_error=True)
  File "/usr/local/python/3.11.13/lib/python3.11/site-packages/jax_plugins/xla_cuda12/__init__.py", line 285, in _check_cuda_versions
    local_device_count = cuda_versions.cuda_device_count()
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: jaxlib/cuda/versions_helpers.cc:113: operation cuInit(0) failed: CUDA_ERROR_NO_DEVICE
/usr/local/python/3.11.13/lib/python3.11/subprocess.py:1885: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = _fork_exec(
2025-12-09 17:47:16,480 WARNING services.py:2148 -- WARNING: The object store is using /tmp instead of /dev/shm because /dev/shm has only 67100672 bytes available. This will harm performance! You may be able to free up space by deleting files in /dev/shm. If you are inside a Docker container, you can increase /dev/shm size by passing '--shm-size=10.24gb' to 'docker run' (or add it to the run_options list in a Ray cluster config). Make sure to set this to more than 30% of available RAM.
2025-12-09 17:47:17,629 INFO worker.py:1951 -- Started a local Ray instance.
(pid=318023) /usr/local/python/3.11.13/lib/python3.11/site-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
(pid=318023)   warnings.warn(
(pid=318023) /usr/local/python/3.11.13/lib/python3.11/site-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
(pid=318023)   warnings.warn(
(pid=318023) ERROR:2025-12-09 17:47:19,199:jax._src.xla_bridge:487: Jax plugin configuration error: Exception when calling jax_plugins.xla_cuda12.initialize()
(pid=318023) Traceback (most recent call last):
(pid=318023)   File "/usr/local/python/3.11.13/lib/python3.11/site-packages/jax/_src/xla_bridge.py", line 485, in discover_pjrt_plugins
(pid=318023)     plugin_module.initialize()
(pid=318023)   File "/usr/local/python/3.11.13/lib/python3.11/site-packages/jax_plugins/xla_cuda12/__init__.py", line 328, in initialize
(pid=318023)     _check_cuda_versions(raise_on_first_error=True)
(pid=318023)   File "/usr/local/python/3.11.13/lib/python3.11/site-packages/jax_plugins/xla_cuda12/__init__.py", line 285, in _check_cuda_versions
(pid=318023)     local_device_count = cuda_versions.cuda_device_count()
(pid=318023)                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(pid=318023) RuntimeError: jaxlib/cuda/versions_helpers.cc:113: operation cuInit(0) failed: CUDA_ERROR_NO_DEVICE
(main pid=318023) wandb: Currently logged in as: swielgosz (mlds-lab) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
(main pid=318023) wandb: WARNING Using a boolean value for 'reinit' is deprecated. Use 'return_previous' or 'finish_previous' instead.
(main pid=318021) wandb: setting up run bz9v8vna
(main pid=318023) Wandb initialized with project: neuralODEs, entity: mlds-lab
(main pid=318023) wandb: Tracking run with wandb version 0.22.1
(main pid=318023) wandb: Run data is saved locally in /workspaces/neuralODEs/scripts/training/wandb/run-20251209_174719-26l6ykdh
(main pid=318023) wandb: Run `wandb offline` to turn off syncing.
(main pid=318023) wandb: Syncing run vague-dream-9879
(main pid=318023) wandb: ⭐️ View project at https://wandb.ai/mlds-lab/neuralODEs
(main pid=318023) wandb: 🚀 View run at https://wandb.ai/mlds-lab/neuralODEs/runs/26l6ykdh
(main pid=318023) wandb:   1 of 1 files downloaded.  
(main pid=318023) Only one orbit available, no split performed. ```