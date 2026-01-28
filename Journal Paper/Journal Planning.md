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

This is similar to what was done in SALAMANDER. Let's give that a reread. I do know that they used Universal Differential Equations. I don't know the exact distinction between neural ODEs and UDEs. 

Given:
  - Known dynamics: f_known(x, t)
  - Unknown forcing: f_theta(x, t) (neural network, outputs acceleration/derivative)
  - Observed trajectories: { (t_i, x_obs(t_i)) }

Define UDE dynamics:
  dx/dt = f_known(x, t) + f_theta(x, t; θ)

Training loop:
  initialize θ
  for each epoch:
    for each trajectory batch:
      x0 = observed initial state
      t_eval = observation times
      # integrate UDE forward
      x_pred(t_eval) = ODESolve(dx/dt, x0, t_eval, θ)
      # loss compares predicted trajectory to observed data
      loss = MSE(x_pred(t_eval), x_obs(t_eval))
      # backprop through ODE solver to update θ
      θ ← θ - η * ∂loss/∂θ

Output:
  learned forcing model f_theta(x, t) embedded in the ODE

Abstract:

This work investigates the potential of neural ordinary differential equations (neural ODEs) for modeling unknown perturbations acting on a spacecraft. Specifically, the proposed approach formulates the dynamics as a universal differential equation, where known physical laws define the nominal dynamics and a neural ODE learns a corrective term capturing unknown forces. Learning such corrective dynamics is particularly important for autonomous spacecraft operations, where discrepancies between baseline physics models and true system behavior can accumulate and degrade long-term prediction. In addition, we explore whether learning perturbative dynamics in a latent state space using latent ODEs can provide improved representation efficiency and robustness compared to modeling directly in the physical state space. As a proof of concept, we demonstrate the ability of the neural ODE to learn perturbative dynamics from data. Results indicate that neural ODE-based models show promise as accurate, data-driven surrogates for traditional spacecraft dynamics models, offering a flexible framework for representing complex and partially unknown dynamics in support of autonomous mission analysis and design.