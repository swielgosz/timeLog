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



