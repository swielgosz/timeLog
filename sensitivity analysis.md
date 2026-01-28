We are testing the sensitivity of our model to changes in model and training pipeline parameters. 

## Seeding
The purpose of this experiment is to test the effect of changing the seeding of the model.  
Config:
``` python
wandb:
  group: "sweep-seeds-v2"  # Change this to your desired group name

data:
  # dataset_name : ["complex_TBP_planar_1_train", "complex_TBP_planar_10_train", "complex_TBP_planar_100_train"]
  dataset_name : ["complex_TBP_planar_10_train"]
  # dataset_name : ["simple_TBP_planar_1_train","complex_TBP_planar_1_train","simple_TBP_planar_10_train","complex_TBP_planar_10_train","simple_TBP_planar_100_train","complex_TBP_planar_100_train"]
  problem: '2BP'

parameters:



  # EXHAUSTIVE LENGTH STRATEGY
  length_strategy:    [
                        # [0.0, 0.1],
                        # [0.0,1.0],[0.0,1.0],],
                        [[0.0,1.0],]
                      ]

  lr_strategy: [[0.001, 0.0001]]
  steps_strategy: [[1000,500]]
  segment_length_strategy: [[4,]]


  width: [64]
  depth: [3]
  train_val_split: 1
  batch_size: [64]
  num_trajs: -1
  seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,]
  # seed: [1234]
  # loss_fcn: "mean_squared_error"
  loss_fcn: percent_error_plus_nmse

  activation: [leaky_relu]
  # activation: leaky_relu
  # activation: elu

  feature_layer: [sph_4D_rinv_vel]
  # output_layer: [mlp_4D_signed, mlp_4D_unit, mlp_4D_unit_softplus, mlp_4D_logmag_unit_exp]
  output_layer: [mlp_4D_unit_softplus]
  planar_constraint: true

  rtol: 0.000001
  atol: 0.00000001
```
We sweep over 20 initial seeds and observe the distribution of mean acceleration error as a violin plot. The name of the wandb group is "sweep-seeds-v2". We train on 10 complex orbits and test the performance of the differently seeded models on 10 independent validation orbits.

