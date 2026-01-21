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
$\vec{a}_{drag}=-\frac{1}{2}$ 