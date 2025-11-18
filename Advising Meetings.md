# November 18

## Dataset
![[Pasted image 20251111133754.png]]
![[Pasted image 20251111133559.png]]

What is our validation data now that we have segmented orbits?
Right now, we segment our data and _then_ we split it into training and validation. This is bad! This means we could end up with parts of the orbit that we do not see. We should split into training and validation orbits before we segment the data. Before that, let's see what our training and validation data currently looks like and see if that could explain the behavior that we're currently seeing at all.

![[Pasted image 20251118105057.png]]
eek!

update this:
![[Pasted image 20251118105809.png]]
## v0- baseline
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

Time to run: 
Trying to speed up training so I can debug better. Let's try longer segments first, then maybe lower depth.  

# v1 - longer segment length (trying to speed up training)
** Difference**: changing segment_length to 18 instead of 4: 
![[Pasted image 20251118100548.png]]
![[Pasted image 20251118100633.png]]
![[Pasted image 20251118100654.png]]


![[Pasted image 20251118100558.png]]
![[Pasted image 20251118100614.png]]
![[Pasted image 20251118100620.png]]

We don't look totally converged. Let's run for 2000 steps instead of 1000.

## v2 - increasing steps to improve convergence
Time to run: 1:25
Loss decreases more but validation gap gets worse:

![[Pasted image 20251118102102.png]]


![[Pasted image 20251118102117.png]]
![[Pasted image 20251118102055.png]]
![[Pasted image 20251118102149.png]]
![[Pasted image 20251118102156.png]]
![[Pasted image 20251118102207.png]]

We're looking real whacky around periapsis... let's look at our datasets

Turns out, we have very little data at periapsis for highly eccentric orbits!

# v3 - updating segmentation to occur after train/val split

**run_id**: 3gd0rl4o

Running for 5000 steps to be safe after 1000 was not converged.
Time to run: 2:47
![[Pasted image 20251118111714.png]]
![[Pasted image 20251118111722.png]]
![[Pasted image 20251118111729.png]]
![[Pasted image 20251118111747.png]]
![[Pasted image 20251118111814.png]]
![[Pasted image 20251118111828.png]]
![[Pasted image 20251118111800.png]]

Validation gap looks a lot better, but mlp output behavior is whacky. 
Looks like we plateau around 3000 steps. Let's lower our batch size to speed up training.

## v4 - lower batch size for faster training
run_id: 26d2zihk
Difference: lower batch size from 32 to 16
run time: 
config:
``` python
  length_strategy:
                      [[
                        [0.0, 1.0],
                      ]]
  lr_strategy: [[0.001]]
  steps_strategy: [[5000]]
  segment_length_strategy: [[4,]]

  width: 16
  depth: 4
  train_val_split: 0.8
  batch_size: 16
  num_trajs: -1

  # loss_fcn: "mean_squared_error"
  loss_fcn: "percent_error"
  # loss_fcn: "percent_error_plus_nmse"

  # activation: tanh
  activation: leaky_relu
  # activation: elu

  feature_layer: sph_4D_rinv_vel
  # feature_layer: sph_4D_rinv_vinv
  output_layer: mlp_4D
  # output_layer: mlp_4D_unit_scaled
  # output_layer: mlp_simple
  planar_constraint: true

  rtol: 0.000001
  atol: 0.00000001```


Time to run: 2:39. No improvement! In fact, this might not even be large enough to learn. Each batch is a set of segments of shape \[batch_size, segment_length, state_dim\]

![[Pasted image 20251118113124.png]]
![[Pasted image 20251118113133.png]]


![[Pasted image 20251118113141.png]]

![[Pasted image 20251118113149.png]]



![[Pasted image 20251118113156.png]]

![[Pasted image 20251118113211.png]]
![[Pasted image 20251118113220.png]]

The plot says the network captures the high-level pattern (both model and truth see an acceleration-magnitude spike near periapsis), but the direction components go unstable exactly where the physics is most demanding. The truth directions rotate rapidly as the spacecraft whips through periapsis; your model instead produces exaggerated swings (overshooting past ±1 and ringing afterward), which means the output head—or the part of the solver downstream of the feature layer—is not translating features into a consistent, normalized direction vector when the magnitude changes quickly. In other words, the headings are uncalibrated during the sharp acceleration spike: the model overreacts, so the vector points in the wrong orientation and then slowly recovers. That usually indicates the output layer can’t represent the combined magnitude/direction relationship under high curvature, or it hasn’t seen enough of those events (so it’s extrapolating). I’d target better regularization or capacity in the output head specifically around enforcing unit direction vectors and coupling them to the magnitude.

It looks like we may not have enough expressiveness in our model. Let's try increasing width without overfitting too extremely. Increase depth after if we need more expressiveness. 

## increasing width for expressiveness
``` python
wandb:
  group: "2BP-sensitivity"  # Change this to your desired group name

data:
  dataset_name : "complex_TBP_planar_4"
  problem: '2BP'

parameters:

  length_strategy:
                      [[
                        [0.0, 1.0],
                      ]]
  lr_strategy: [[0.001]]
  steps_strategy: [[3000]]
  segment_length_strategy: [[4,]]

  width: 32
  depth: 2
  train_val_split: 0.8
  batch_size: 32
  num_trajs: -1

  # loss_fcn: "mean_squared_error"
  loss_fcn: "percent_error"
  # loss_fcn: "percent_error_plus_nmse"

  # activation: tanh
  activation: leaky_relu
  # activation: elu

  feature_layer: sph_4D_rinv_vel
  # feature_layer: sph_4D_rinv_vinv
  output_layer: mlp_4D
  # output_layer: mlp_4D_unit_scaled
  # output_layer: mlp_simple
  planar_constraint: true

  rtol: 0.000001
  atol: 0.00000001
```

run_id: z0v2rmhl
time to run: 1:29


![[Pasted image 20251118115157.png]]
![[Pasted image 20251118115124.png]]
![[Pasted image 20251118115208.png]]
![[Pasted image 20251118115228.png]]
![[Pasted image 20251118115234.png]]
![[Pasted image 20251118115254.png]]

- output features are not tracking, but we are seeing smooth behavior now
- Let's try increasing depth to increase expressiveness

# v4 - 
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
