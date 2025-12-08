Training progress:  17%|█▋        | 433/2500 [00:25<01:46, 19.49step/s] [repeated 10x across cluster]
(main pid=2889586) wandb: Tracking run with wandb version 0.22.1
(main pid=2889586) wandb: Run data is saved locally in /workspaces/neuralODEs/scripts/training/wandb/run-20251208_212940-du56dide
(main pid=2889586) wandb: Run `wandb offline` to turn off syncing.
(main pid=2889586) wandb: Syncing run wild-butterfly-6637
(main pid=2889586) wandb: ⭐️ View project at https://wandb.ai/mlds-lab/neuralODEs
(main pid=2889586) wandb: 🚀 View run at https://wandb.ai/mlds-lab/neuralODEs/runs/du56dide
(main pid=2888759) Only one orbit available, no split performed.  [repeated 2x across cluster]
(main pid=2888759)  Validation set will be the same as training set! [repeated 2x across cluster]
(main pid=2889586) Wandb initialized with project: neuralODEs, entity: mlds-lab
(main pid=2889586) wandb:   1 of 1 files downloaded.  
Training progress:   0%|          | 0/2500 [00:00<?, ?step/s]
Training progress: 100%|██████████| 2500/2500 [02:33<00:00, 16.31step/s] [repeated 14x across cluster]
Training progress:  91%|█████████ | 2272/2500 [02:23<00:15, 14.77step/s] [repeated 1514x across cluster]
Training progress:  16%|█▌        | 390/2500 [00:21<01:56, 18.19step/s] [repeated 6x across cluster]
Training progress:  92%|█████████▏| 2303/2500 [02:29<00:14, 13.15step/s] [repeated 110x across cluster]
Training progress:  46%|████▌     | 1151/2500 [01:32<01:43, 13.03step/s] [repeated 1198x across cluster]
Training progress:  32%|███▏      | 796/2500 [01:02<03:03,  9.29step/s] [repeated 13x across cluster]
Training progress:  97%|█████████▋| 2414/2500 [02:34<00:06, 13.55step/s] [repeated 254x across cluster]
(main pid=2888965) Integration failed: Above is the stack outside of JIT. Below is the stack inside of JIT:
(main pid=2888965)   File "/usr/local/python/3.11.13/lib/python3.11/site-packages/ray/_private/workers/default_worker.py", line 323, in <module>
(main pid=2888965)     worker.main_loop()
(main pid=2888965)   File "/usr/local/python/3.11.13/lib/python3.11/site-packages/ray/_private/worker.py", line 984, in main_loop
(main pid=2888965)     self.core_worker.run_task_loop()
(main pid=2888965)   File "/workspaces/neuralODEs/scripts/training/sweep.py", line 71, in main
(main pid=2888965)     model = train_model(
(main pid=2888965)             ^^^^^^^^^^^^
(main pid=2888965)   File "/workspaces/neuralODEs/neuralODE/neuralODE.py", line 934, in train_model
(main pid=2888965)     preds = model(ref_ts, ref_y0)
(main pid=2888965)             ^^^^^^^^^^^^^^^^^^^^^
(main pid=2888965)   File "/workspaces/neuralODEs/neuralODE/neuralODE.py", line 136, in __call__
(main pid=2888965)     solution = diffrax.diffeqsolve(
(main pid=2888965)                ^^^^^^^^^^^^^^^^^^^^
(main pid=2888965)   File "/usr/local/python/3.11.13/lib/python3.11/site-packages/diffrax/_integrate.py", line 1502, in diffeqsolve
(main pid=2888965)     sol = result.error_if(sol, jnp.invert(is_okay(result)))
(main pid=2888965)           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(main pid=2888965)   File "/usr/local/python/3.11.13/lib/python3.11/site-packages/equinox/_module/_prebuilt.py", line 33, in __call__
(main pid=2888965)     return self.__func__(self.__self__, *args, **kwargs)
(main pid=2888965)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(main pid=2888965) 
(main pid=2888965) equinox.EquinoxRuntimeError: The maximum number of solver steps was reached. Try increasing `max_steps`.
(main pid=2888965) 
(main pid=2888965) -------------------
(main pid=2888965) 
(main pid=2888965) An error occurred during the runtime of your JAX program.
(main pid=2888965) 
(main pid=2888965) 1) Setting the environment variable `EQX_ON_ERROR=breakpoint` is usually the most useful
(main pid=2888965) way to debug such errors. This can be interacted with using most of the usual commands
(main pid=2888965) for the Python debugger: `u` and `d` to move up and down frames, the name of a variable
(main pid=2888965) to print its value, etc.
(main pid=2888965) 
(main pid=2888965) 2) You may also like to try setting `JAX_DISABLE_JIT=1`. This will mean that you can
(main pid=2888965) (mostly) inspect the state of your program as if it was normal Python.
(main pid=2888965) 
(main pid=2888965) 3) See `https://docs.kidger.site/equinox/api/debug/` for more suggestions.