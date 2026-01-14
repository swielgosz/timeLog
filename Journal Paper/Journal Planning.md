The topic of the conference paper in the spring will most likely be forcing linearization in latent space and extending this to find periodic orbits and the offshooting trajectories. 
We need to heavily extend the results of the conference paper. This will include:
- [ ] sensitivity studies
- [ ] 3D orbits
	- [ ] extend 2BP to 3D
	- [ ] extend CR3BP to more than just Lyapunov orbits

## Datasets
Our TBP planar was named `<complex|simple>_TBP_planar_<num_orbits>_<train|test>`
Let's follow the same convention for nonplanar

## Sensitivities to study
What are the effects of:
- **training data quantity**
- segment length strategy
- learning rate strategy
- steps strategy
- batch size
- loss function
- feature layer
- output layer
- activation function
- segment then train/test split or vice versa

## Paper Goals
We need to perform sensitivity studies to find expected bounds on performance. This is more conference level.

For the novelty aspect, we want to use and compare neuralODEs and latentODEs for the purpose of modeling unknown force components:
- SRP
- 4BP from CR3BP approx
- drag
- for latent ODEs, can we exploit them to for learning dynamics say when an orbit is on the dark side of the moon?

This is similar to what was done in SALAMANDER. Let's give that 