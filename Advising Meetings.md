# November 18

Debugging this behavior:
![[Pasted image 20251112102231.png]]
config:
``` python
parameters:

  length_strategy:
                      [[ 
                        [0.0, 1.0],                        
                      ]]
  lr_strategy: [[0.001, ]]
  steps_strategy: [[1000, ]]
  segment_length_strategy: [[4,]]

  width: 16
  depth: 4
  train_val_split: 0.8
  batch_size: 32
  num_trajs: -1

  loss_fcn: "percent_error"

  activation: leaky_relu

  feature_layer: sph_4D_rinv_vel

  output_layer: mlp_4D

  planar_constraint: true

  rtol: 0.000001
  atol: 0.00000001
```


Trying to speed up training so I can debug better. Let's try longer segments first, then maybe lower depth.  

Changing segment_length to 18 instead of 4: 
![[Pasted image 20251118100548.png]]
![[Pasted image 20251118100633.png]]
![[Pasted image 20251118100654.png]]


![[Pasted image 20251118100558.png]]
![[Pasted image 20251118100614.png]]
![[Pasted image 20251118100620.png]]

We don't look totally converged. Let's run for 200 steps instead of 1000.
Time to run: 


Previously, 2 wasn't sufficient but maybe 3 is? Keeping batch size the same for now

# November 11
Is there a way we can view the acceleration magnitude and direction similar to how we applied the model and viewed the feature layer components?

To determine if acceleration is pointed in the correct direction, we know that $a_r \;=\; \hat{\mathbf r}\cdot \mathbf a \;=\; \frac{\mathbf r}{\|\mathbf r\|}\cdot \mathbf a$

- If a_r<0: acceleration points **toward** the primary → attractive.
- If a_r>0: acceleration points **away** from the primary → repulsive.
- If a_r\approx 0: no radial component (purely tangential).
We can also get the angle between the acceleration and the position, where we know that if our force is purely attractive we should have the angle = $\pi$ :
\theta = \operatorname{atan2}\!\left(\;\|\mathbf r \times \mathbf a\|\;,\;\mathbf r\cdot \mathbf a\;\right)

- \theta is the angle between **r** and **a** in [0,\pi].
    
- For a purely attractive **central** force (2BP), \mathbf a is antiparallel to \mathbf r ⇒ \theta \approx \pi (180°).

## Dataset
![[Pasted image 20251111133559.png]]
![[Pasted image 20251111133754.png]]

Possible errors:
may occur at apoapsis, periapsis, when we cross -1 or 1
repeat the plots for the acceleration components and magnitude 
if $\hat{a}_{true} = \hat{a}_{pred}$
return to og config with spikes - why are they there?
make the same plots that I have for input features, but for output features 

## Baseline:
``` python
data:
  dataset_name : "complex_TBP_planar_4"
  problem: '2BP'

parameters:

  ## MINIMAL LENGTH STRATEGY
  length_strategy:
                      [[ 
                        [0.0, 1.0],                        
                      ]]
  lr_strategy: [[0.001, ]]
  steps_strategy: [[1000, ]]
  segment_length_strategy: [[4,]]

  width: 16
  depth: 2
  train_val_split: 0.8
  batch_size: 32
  num_trajs: -1

  # loss_fcn: "mean_squared_error"
  loss_fcn: "percent_error"
  # loss_fcn: "percent_error_plus_nmse"

  activation: tanh
  # activation: leaky_relu
  # activation: elu

  feature_layer: sph_4D_rinv_vel
  # feature_layer: sph_4D_rinv_vinv
  output_layer: mlp_4D
  planar_constraint: true

  rtol: 0.000001
  atol: 0.00000001
  ```
loss:
![[Pasted image 20251111133931.png]]
![[Pasted image 20251111133936.png]]
![[Pasted image 20251111134252.png]]![[Pasted image 20251111134300.png]]
![[Pasted image 20251111134307.png]]

Our model may not be complex enough

## v2 - increase depth to 4
``` python
parameters:

  ## MINIMAL LENGTH STRATEGY
  length_strategy:
                      [[ 
                        [0.0, 1.0],                        
                      ]]
  lr_strategy: [[0.001, ]]
  steps_strategy: [[1000, ]]
  segment_length_strategy: [[4,]]

  width: 16
  depth: 4
  train_val_split: 0.8
  batch_size: 32
  num_trajs: -1

  # loss_fcn: "mean_squared_error"
  loss_fcn: "percent_error"
  # loss_fcn: "percent_error_plus_nmse"

  activation: tanh
  # activation: leaky_relu
  # activation: elu

  feature_layer: sph_4D_rinv_vel
  # feature_layer: sph_4D_rinv_vinv
  output_layer: mlp_4D
  planar_constraint: true

  rtol: 0.000001
  atol: 0.00000001
```
![[Pasted image 20251111134744.png]]
![[Pasted image 20251111134750.png]]
![[Pasted image 20251111134806.png]]
![[Pasted image 20251111134730.png]]
![[Pasted image 20251111134820.png]]
## v3 - relu

## v4 - leaky relu
![[Pasted image 20251111134856.png]]
![[Pasted image 20251111134907.png]]
![[Pasted image 20251111134948.png]]
![[Pasted image 20251111134957.png]]
## v5 - change input features
Previously, input features were 1/r, x/r, y/r, z/r, and velocity. replace velocity with velocity magnitude and direction.
