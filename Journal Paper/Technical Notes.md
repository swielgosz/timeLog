# February 10
## Reverse mode auto diff
[Reverse Mode Automatic Differentiation - 26 min](https://www.youtube.com/watch?v=EEbnprb_YTU)
[MIT OpenCourseWare - Adjoint Differentiation of ODE Solutions 58 min](https://www.youtube.com/watch?v=cvBHoCAUkD4)
[Parallel Computing and SciML - Chris Rackaukas 1 hr 36 min](https://www.youtube.com/watch?v=KCTfPyVIxpc)
[Neural ODE - Pullback/vJp/adjoint rule 1 hr 42 min](https://www.youtube.com/watch?v=u8NL6CwSoRg) - for ODE
[Adjoint State Method for an ODE | Adjoint Sensitivity Analysis 42 min - Machine Learning & Simulation](https://www.youtube.com/watch?v=k6s2G5MZv-I) - more introductory

Let's watch in the following order:
1. Overview: [Reverse Mode Automatic Differentiation - 26 min](https://www.youtube.com/watch?v=EEbnprb_YTU)
2. Adjoint Method: [Adjoint State Method for an ODE | Adjoint Sensitivity Analysis 42 min - Machine Learning & Simulation](https://www.youtube.com/watch?v=k6s2G5MZv-I) 
3. Adjoint method for ODEs: [Parallel Computing and SciML - Chris Rackaukas 1 hr 36 min](https://www.youtube.com/watch?v=KCTfPyVIxpc)
> **The adjoint method is the continuous-time formulation of reverse-mode automatic differentiation for differential equation solvers.**

### Reverse Mode Automatic Differentiation - Nathan Sprague
Side tangent: Reverse-mode autodiff is neither opt-disc nor disc-opt by itself. Opt-disc vs disc-opt are about what computational graph you differentiate through when ODE solvers are involved.
Reverse-mode autodiff is backprop through a computation graph: $\frac{\partial L}{\partial \theta}$ computed efficiently when you have many parameters and a scalar loss. It does not decide whether you differentiate through an ODE solver, use adjoints, whether solver is continuous or discrete.
Disc-opt pipelin
## Neural ODE training difficulty and improvement – code repositories
- [ ]  ANODE – Accurate gradients for Neural ODEs (2019)
Paper: ANODE: Unconditionally Accurate Memory-Efficient Gradients for Neural ODEs
Code: https://github.com/amirgholami/anode
Notes: Fixes issues with the adjoint method by using discretize-then-optimize and checkpointing for stable gradients.

- [ ]  Regularized Neural ODEs (RNODE)
Paper: How to Train Your Neural ODE: the World of Jacobian and Kinetic Regularization
Code: https://github.com/cfinlay/ffjord-rnode
Notes: Adds Jacobian norm and kinetic energy regularization to reduce stiffness and number of function evaluations during training.

- [ ]  Simulation-Free Training of Neural ODEs
Paper: Simulation-Free Training of Neural ODEs on Paired Data
Code: https://github.com/seminkim/simulation-free-node
Notes: Avoids backpropagating through the ODE solver by using flow matching to reduce training cost and instability.

## Additional Neural ODE codebases (general reference)
- [ ]  Easy Neural ODE (JAX + Haiku)
Code: https://github.com/jacobjinkelly/easy-neural-ode
Notes: JAX-based Neural ODE, Latent ODE, FFJORD examples with training scripts.
- [ ]  Awesome Neural ODE (curated list of repos and papers)
Code: https://github.com/Zymrael/awesome-neural-ode
Notes: Useful index of papers, tools, and implementations across Neural ODE research.
- [ ]  Neural ODE tutorial notebook (PyTorch)
Code: https://github.com/cagatayyildiz/neural-ode-tutorial
Notes: Simple tutorial-style implementation of Neural ODE training.

## **Related solver and training behavior repositories**
- [ ]  HeavyBallNODE
Paper: Heavy Ball Neural ODEs
Code: https://github.com/hedixia/HeavyBallNODE
Notes: Adds momentum to continuous-depth models to improve optimization and conditioning.

- [ ]  Lyapunov-based Neural ODE training
Paper: Learning stable dynamics with Lyapunov constraints
Code: https://github.com/ivandariojr/LyapunovLearning
Notes: Enforces stability constraints during Neural ODE training.

- [ ]  Adaptive step-size Neural ODE experiments
Paper: Training Neural ODEs with adaptive step size control
Code: https://github.com/Allauzen/adaptive-step-size-neural-ode
Notes: Studies how adaptive solvers interact with training dynamics and stability.

## Other papers
- [ ] A guide to neural ordinary differential equations: Machine learning for data-driven digital engineering (in Zotero)

## Other videos/resources
- https://www.youtube.com/watch?v=wTgYDg_zPcg[Neural ODE Code Walkthrough]
- https://www.uu.nl/sites/default/files/an_introduction_to_scientific_modelling_with_neural_ODEs.pdf Patrick Kidger walkthrough to neural ODEs (corresponding video at https://www.youtube.com/watch?v=7QBLRzkMi4c)
- 

# SALAMANDER
## Motivation
Systems of interest have partially known dynamics. Unknown forces can lead to simulation biases, which basically means our model results are skewed. As we move toward relying more on autonomous systems in complex missions, it is important to be able to model these unknown forces and correct the base level of dynamics. Neural networks have been used to perform orbit prediction, but they are completely data driven which is data inefficient and the models are longer to train. 

Deep symbolic regression has been used to uncover unmodeled forces, but the authors argue that Universal Differential Equations (UDEs) are more suitable because they are more expressive, robust, and accurate than symbolic regression. To this end, they combine symbolic regression and UDEs. 

Takeaways:
- Good motivation here for why we want models to predict unknown dynamics *on top* of the base dynamics - this is more data efficient and utilizes laws that we already know. 
- What are UDEs? 

## Background
### SciML
Mechanistic models are derived from known physical laws. You probably know the structure of the system and use dasta to estimate parameters. Non-mechanistic models are data driven and are black box predictive models which do not need knowledge of the system. However, they lack interpretability and are not strongly generalizable. Author proposes combining these

... Neural Networks, ResNET, Neural ODEs ...
## Universal Differential Equations
Domain models are augmented with data-driven techniques: $\frac{dh(t)}{dt}=f(h(t),t,\theta,\alpha,\beta)$. This makes training more data-efficient. 

Side tangent - this is not a very clear explanation. Let's look elsewhere. 

**Patrick Kidger Thesis**
"Endowing a model with any known structure of a problem is known as giving the model an inductive bias."

So when we say we want to learn the unknown portion of $f = f_{known} + f_{NODE}$, this is a neural ODE.

Most of the rest of the paper is numerical examples. For now, move on to learning UDEs

---
# Universal Differential Equations for Scientific  Machine Learning
Note to self - source code is included in this paper. It is written in Julia. 

## Introduction
- Mechanistic models are used in areas where small training datasets result in inaccurate models
	- Mechanistic models are constrained to be predictive
	- A mechanistic model in machine learning is a model whose structure is derived from first-principles knowledge of the system (e.g., physical laws), with learning used only to estimate parameters or augment missing components.
	- Pure physics<── Hybrid (physics + ML)──→ Pure ML

| **Type**               | **Example**                    | **What ML learns** |
| ---------------------- | ------------------------------ | ------------------ |
| Fully mechanistic      | Newton’s equations             | Nothing            |
| Parametric mechanistic | Mass, drag coefficients        | Parameters         |
| **Hybrid / UDE**       | Known dynamics + NN correction | Missing forces     |
| Black-box ML           | MLP mapping state → next state | Everything         |
- Mechanistic models vs standard ML models:
	- Standard ML (black box):
		`state_t  ──► neural network ──► state_{t+1}` 
		- no explicit notion of physics
		- learns correlations
		- poor extrapolation is common
		- limited interpretability
	- Mechanistic ML (hybrid):
		`dx/dt = known_physics(x, t) + learned_component(x, t)`
		- physics defines structure
		- ML fills gaps
		- strong inductive bias
		- better extrapolation and stability
		- Examples in ML:
			- Physics-informed parameter learning
				- Governing equations are fixed, ML estimates unknown parameters
				- Example: learning drag coefficient, thrust efficiency
			- Universal Differential Equations
				- Differential equations with learned terms
				- Neural networks represent unknown physics
				- Widely used with neural ODE solvers
			- Physics-informed neural networks (PINNs)
				- neural networks trained to satisfy PDEs/ODEs
				- Physics enforced through loss terms
			- Gray box models
				- Partial mechanistic structure
				- Data-driven submodels embedded inside
		- Advantages:
			- inductive biases - hypothesis space is restricted to physically plausible solutions
			- data efficiency
			- extrapolation. better generalization to new regimes, longer time horizons, changed parameters
			- interpretability - learned components can be inspected and reasoned about
		- But really, PINNs aren't mechanistic/ Pinn vs UDE/mechanistic neural ODE
			- PINN:
				- neural netwrok approximates the solution
				- physics enforced by penalizing residuals
				- dynamics are not explicitly integrated
				- `NN(x,t) → u(x,t) # loss enforces PDE/ODE`
			- UDE/mechanistic neural ODE
				- Physics defines the time evolution
				- Neural network augments dynamics
				- Solver enforces physics exactly
				- `dx/dt = physics(x,t) + NN(x,t) # integrate forward in time`

# Perturbations (Vallado Chapter 8)
We will consider drag and SRP in the 2BP. Atmospheric drag has the most influence on a satellite near Earth, and more distant satellites are affected by SRP and third-body effects moreso than oblateness and drag. 

## Atmospheric Drag
We want a model that is good enough to account for atmospheric density . Drag is nonconservative - energy is lost due to friction. Drag primarily changes the SMA and eccentricity of the orbit. 

Equation for aerodynamic drag:
$\vec{a}_{drag}=-\frac{1}{2}\frac{c_dA}{m}\rho\nu^2_{rel}\frac{\vec{v}_{rel}}{|\vec{v}_{rel}|}$  
- $c_D$ is the dimensionless coefficient of drag which reflects the satellite's susceptibility to drag forces. It is ~2.2 for flat plate modle in upper atmosphere. Spheres have $c_D$ of 2.0 to 2.1. It is satellite configuration specific and usually limited to three significant figures. 
- $\rho$ is the atmospheric density - most difficult parameter to determine
- $A$ is the exposed cross-sectional area which is the are normal to the satellites velocity vector. 
- $m$ is satellite's mass
- $v_{rel}$ is the velocity vector relative to the atmosphere. From Vallado: "In actuality, the Earth’s atmosphere has a mean motion due to the Earth’s rotation, and the winds are superim-
posed on this mean motion. Notice also that the force of drag opposes the velocity vector at all times. This is a primary use for the NTW coordinate system. For a nonspherical
satellite, we must also consider companion aerodynamic forces such as lift and side forces. Remember that although the atmosphere is rotating, it does so with a “profile”
that follows a little behind the Earth. Due to friction with the Earth, the atmosphere closest to the Earth rotates a little faster than higher altitudes. The velocity vector relative to
the rotating atmosphere is

![[Pasted image 20260121105029.png]]-