``` 
parameters:
  # KAN capacity for 2BP. Start moderate and sweep around this.
  # Basis family is modular and configurable.
  # Options: gaussian, gaussian_plus_inverse_r,
  #          gaussian_plus_soft_inverse_r, gaussian_plus_shifted_inverse_r,
  #          cubic_bspline
  kan_basis_family: gaussian
  kan_num_basis: 5
  kan_basis_min: -1.5
  kan_basis_max: 1.5
  kan_basis_scale_mult: 1.0


  # kan_basis_eps: 1.0e-2
  # kan_inverse_basis_clip: 10.0
  # kan_inverse_shift: 0.1
  # Keep diagnostics on but avoid verbose Jacobian debug logging.
  solver_stats_jacobian_debug: false
  solver_stats_frequency: 100   # log every N global steps
  # solver_stats_samples: 5       
  solver_stats_jacobian: true
  solver_stats_jacobian_max_dim: 32
  solver_stats_use_all_samples: true
  # solver_stats_jacobian_eps: 1.0
  # solver_stats_jacobian_percentiles: [5.0, 95.0]
  # Keep full-window supervision each phase.
  length_strategy: [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]

  # Cubic B-spline basis can still induce stiff regions; use gentler LR.
  lr_strategy: [[0.001, .0005, .0001]]
  steps_strategy: [[500, 500, 1000]]

  # Curriculum from shorter to longer segments.
  segment_length_strategy: [[360,360,360]] #[[6, 12, 18]]


  width: [24]
  depth: [2]
  train_val_split: 1
  batch_size: [128]
  num_trajs: -1
  # Use multiple seeds to compare architectures robustly.
  seed: [2345]
  # loss_fcn: "mean_squared_error"
  loss_fcn: [percent_error_plus_nmse]
  # loss_fcn: multi_step_rollout_percent_error_plus_nmse
  # rollout_steps: 8        # optional, default = full segment
  # rollout_stride: 1       # optional, default = 1

  # activation: [leaky_relu]
  activation: [tanh]
  # activation: leaky_relu
  # activation: elu

  # feature_layer: [sph_4D_rinv_vel_loga_energy]
  # feature_layer: [sph_4D_rinv_vel_logra]
  feature_layer: [sph_4D_rinv_vel]
  # output_layer: [mlp_4D_signed, mlp_4D_unit, mlp_4D_unit_softplus, mlp_4D_logmag_unit_exp]
  # Stabilized KAN candidate: clipped log-mag + unit direction.
  output_layer: [mlp_4D_unit]
  planar_constraint: true
  

  # Slightly looser tolerances + larger step budget for solve stability.
  rtol: 0.00001
  atol: 0.0000001
```