# November 12 
``` python

def sph_4D_rinv_vinv(y):
    pos = y[:3]
    vel = y[3:6]
    radius = jnp.sqrt(jnp.sum(pos**2))
    velocity = jnp.sqrt(jnp.sum(vel**2))
    return jnp.concatenate(
        [
            jnp.array(
                [
                    1 / radius,
                    pos[0] / radius,
                    pos[1] / radius,
                    pos[2] / radius,
                    vel[0] / radius,
                    vel[1] / radius,
                    vel[2] / radius,
                ],
            ),
        ],
    )

```

```
  ## NO LENGTH STRATEGY
  length_strategy: [[[0.0, 1.0],[0.0, 1.0]],
                    [[0.0, 0.1], [0.0, 1.0]],

                   ]
                    
  lr_strategy: [[0.001, 0.0001],[0.001,0.001]] # if we want
  steps_strategy: [[1000,2000],
                   [500, 1000],
                  ]
                  ```
                  
                  , "complex_TBP_planar_1024", "complex_TBP_planar_10000"]