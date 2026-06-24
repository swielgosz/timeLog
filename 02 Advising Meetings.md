# June 24

Work on speed up
Run best case overnight - do training properly for 8 hours

Architecture Changes

- Added sph_4D_rinv_vel feature layer mapping [x,y,vx,vy] to [1/r, sx, sy, vx, vy], mirroring the parent repo. Inverse-radius features improve generalization in Keplerian dynamics.
- Added mlp_4D_unit_scaled output layer that constrains the network to output physically-structured [vx, vy, ax, ay] with a unit-direction acceleration.
- Added HIDDEN and N_LAYERS to the config block so architecture can be varied without touching code.

Training Changes

- All orbit segments are batched into a single forward+adjoint pass. The key correctness fix is that backward_dynamics loops over each sample individually — a full-batch VJP would sum gradients across orbits, contaminating each orbit's adjoint with other orbits' Jacobians.
- Added NeuralODENormalized with three scaling modes (magnitude, componentwise_rel, componentwise_abs). Scaling happens only on dL/dθ, not on a(t) itself — two separate autograd.grad calls per RK4 step. Scaling a(t) corrupts the adjoint ODE and produces NaN losses.
- Added WARMUP_STEPS: normalized modes use a standard NeuralODE for the first N epochs so weights stabilize before scaling activates.
- split_dataset now shuffles orbits with a seeded RNG (8 train, 2 test). Split indices are stored in each checkpoint so analysis scripts use the exact same split.
- Added seeded reproducibility via torch.manual_seed and np.random.seed in main() and each Ray worker.
- Replaced sequential training with Ray remote workers, one per adjoint mode, all CPU-pinned. Progress is plain print lines every LOG_INTERVAL epochs to avoid tqdm flashing with parallel workers.

Analysis and Diagnostics

- Added adjoint_diagnostics.py which computes and plots the Frobenius norm of df/dtheta, the norm of a(t), and their product (the integrand of eq. 5 of Chen 2018) for each model and orbit, both as time series and on the XY plane. Separates train vs test orbits into subdirectories. The goal is to understand where along the orbit each adjoint method concentrates its gradient signal, since periapsis dominance is the core problem being studied.
- Updated analyze.py to include all 4 model labels and colors, added .squeeze(1) on batched model output to handle the extra batch dimension, and added a .get fallback for scalar in older checkpoints.
- Added neuralODEs.code-workspace so NeuralODE-ref appears as a second root in VS Code source control.



Added the stripped down neuralODE repo to my workspace. 
It's a fully independent git repo — its own `.git/`, its own remote pointing to the original GitHub repo. It doesn't import or depend on anything from `neuralODEs`. The only "relationship" is that it lives inside the same folder, which is just a convenience for your Explorer sidebar. Nothing is linked at the code level.

Within this repo, we are going to change the parameter gradient to 
$$$$$$\frac{dL}{d\theta} = \int_{t_1}^{t_0} \frac{1}{f(z(t),t,\theta)}a(t)^{T} \frac{\partial f(z(t), t, \theta)}{\partial \theta} \, dt$$

I want to update the parameter gradient from Chen eq. 5:$$\frac{dL}{d\theta} = \int_{t_1}^{t_0} a(t)^{T} \frac{\partial f(z(t), t, \theta)}{\partial \theta} \, dt$$ to a parameter gradient update which is normalized by the dynamics $$\frac{dL}{d\theta} = \int_{t_1}^{t_0} \frac{1}{f(z(t),t,\theta)}a(t)^{T} \frac{\partial f(z(t), t, \theta)}{\partial \theta} \, dt.$$
Can we do this, and if so why? 

**`NeuralODE_naive` ↔ `diffrax.RecursiveCheckpointAdjoint`** Both differentiate through the solver steps directly using standard autograd. `RecursiveCheckpointAdjoint` adds gradient checkpointing to reduce the O(T) memory cost, but the fundamental approach — autodiff through the RK steps — is the same.

**`NeuralODE` ↔ `diffrax.BacksolveAdjoint`** Both solve a second ODE backward in time using the adjoint equations (Chen 2018). `BacksolveAdjoint` is the production-quality version — it handles the time direction correctly, supports adaptive step sizes, and integrates cleanly with diffrax's solver infrastructure — but the mathematical object is identical to what `NeuralODE.py` implements.

One nuance worth noting: `BacksolveAdjoint` in diffrax also backsolves for `z(t)` (the state) simultaneously with the adjoint, rather than saving `z(t)` from the forward pass. This makes it truly O(1) memory but introduces numerical error if the dynamics are stiff or the backward solve diverges — which is exactly the kind of instability you're seeing near periapsis in 2BP. That's part of why your normalized gradient idea is interesting: it addresses a symptom of that instability at the gradient level.
# June 23
Our loss is a scalar-valued function:
$$L\big(z(t_1)\big)=L\left(\operatorname{ODESolve}\big(z(t_0), f, t_0, t_1, \theta\big)\right)$$
The adjoint describes how the gradient of the loss depends on the hidden state at each instant:
$$a(t) = \frac{\partial L}{\partial z(t)}$$
Its dynamics are governed by 
$$\frac{d a(t)}{dt} = - a(t)^{T} \frac{\partial f(z(t), t, \theta)}{\partial z}$$
We compute the parameter gradient as 
$$\frac{dL}{d\theta} = \int_{t_1}^{t_0} a(t)^{T} \frac{\partial f(z(t), t, \theta)}{\partial \theta} \, dt$$
The adjoint has an inherent time dependence because $a(t)=\partial L/\partial z(t)$ measures how the loss changes if the state at time t is perturbed. For a trajectory-level loss, earlier states can influence a larger portion of the subsequent trajectory, while later states have less remaining time over which their perturbations can affect the accumulated loss. Therefore, in some systems and loss formulations, $\|a(t)\|$ may decrease as we move forward in time. However, this decay is not guaranteed; it depends on the loss definition, the learned dynamics, and the state-transition sensitivity along the trajectory.

We have a trajecotry loss. The adjoint diagnostic should include sensitivity contributions from every time point where we evaluate the loss, not only the final state. 

Making the loss $L = \frac{1}{N+1}\sum_{k=0}^{N} \ell(z(t_k), z_k^\ast)$ already makes the objective values at all saved times equally weighted. But it does not make their gradient influence equal. Earlier states still influence more future residuals through the rollout dynamics. The diminishing adjoint is expected for a causal rollout loss, but it is undesirable for learning a globally accurate orbital vector field. A uniformly averaged trajectory loss weights all time samples equally in the objective, but not necessarily in the gradient.

So, our issue is that the loss values are all equally weighted in time, but the training signal is not.


---
Lev Semenovich Pontryagin, EF Mishchenko, VG Boltyanskii, and RV Gamkrelidze. The mathematical theory of optimal processes. 1962. This is the original adjoint method source.

Comes from older optimal-control idea - introduce a second variable (adjoint or costate - these typically refer to the same mathematical object in optimal control).

In optimal control, we care about this problem because we want to carry sensitivity information backward through our dynamics. 

Basic optimal-control problem:
$\min_{u(t)} J = \phi(x(t_f)) + \int_{t_0}^{t_f} \ell(x(t),u(t),t)\,dt$ 
subject to
$\dot x(t)=f(x(t),u(t),t), \qquad x(t_0)=x_0.$

This is saying that we have a system moving according to physics. We can influence it with some input. What input history should we choose so the system does the best thing? In our case, we choose the hwole control function to make the total cost $J$ as small as possible.The first part is the terminal cost. The second part is the running cost which accumulates cost along the entire trajectory. It is constrained by the system dynamics, and must evolve according to the differential equation $\dot x(t)=f(x(t),u(t),t)$.

Introduce a costate $\lambda(t)$ (also called adjoint variable) and form the Hamiltonian
$$H(x,u,\lambda,t) = \ell(x,u,t) + \lambda^\top f(x,u,t).$$

The state $x(t)$ tellsl you where the ststem is, the costate $\lambda(t)$ tells us how valuable or costly the state is from the perspective of the objective. This is analogous to $a(t)$ in our neuralODE - it tells us how much the final objective would change if we slightly changed the state at time $t$. 

The Hamiltonion tells you the cost you are accumulating right now, and the way your current dynamics affect future cost. First term is the immediate running cost (how expensive is it to be at this state and use this control right now?), second term connects the dynamics to the future objective. $f$ tells you how the state is currently moving, and the costate $\lambda$ tells you which state changes matter for the objective. $f$ might point some direction in state space which tells us how the state is changing, $\lambda$ might point in the direction where the loss is most snesitive. Their dot product asks if the system is currently moving in a direction that increases or decreases the objective?

The necessary conditions are then
$$\dot x = \frac{\partial H}{\partial \lambda} = f(x,u,t),$$
This just recovers the original dynamics, and says the state must still objey the phsyical dynamics.
$$\dot \lambda = - \frac{\partial H}{\partial x},$$
this tells us how the costate evolves. The state evolves forward according to the dynamics, and the costate evolves according to how the Hamiltonian changes iwth the state. “If changing the current state would strongly affect immediate cost or future dynamics, then the costate changes accordingly.”

and the optimal control satisfies a pointwise extremum condition
$$u^*(t) = \arg\min_u H(x^*(t),u,\lambda(t),t),$$ (or argmax depending on the Hamiltonian sign convention)

This says “If you already know the state x(t) and costate \lambda(t), then the best control at that instant is the one that optimizes the Hamiltonian.” This means “If you already know the state x(t) and costate \lambda(t), then the best control at that instant is the one that optimizes the Hamiltonian.”

The costate terminal condition is typically 
$\lambda(t_f) = \frac{\partial \phi}{\partial x(t_f)}.$ so $\lambda(t)$ is telling us how much the final objective would change if you perturbed the state at time $t$. 

This is essentially the same object as our neural ODE notation $a(t)=\frac{\partial L}{\partial z(t)}.$ 

Why did it matter. for control? It converted trajectory optimization into sensitivity propagation - answers questions like how does final miss distance change if I perturb the thrust at this time. 

The adjoint gives a compact answer. If a perturbation $\delta u(t)$ changes the dynamics through
$$\delta \dot x = f_x \delta x + f_u \delta u,$$
then the first-order variation of the objective can be written in terms of the adjoint as
$$\delta J = \int_{t_0}^{t_f} \left( \frac{\partial H}{\partial u} \right)^\top \delta u(t)\,dt$$

plus boundary terms, depending on the problem formulation.

  

So $\partial H/\partial u$ is the local gradient signal for the control. This is the same structural idea that appears in neural ODEs, except the “control” becomes the parameter vector $\theta$, and the gradient signal becomes

  

$\frac{dL}{d\theta} = \int_{t_0}^{t_f} a(t)^\top \frac{\partial f_\theta}{\partial \theta} \,dt.$


Do not break:
- relationship between the forward dynamics, loss, adjoint equation, and the parameter-gradient accumulation. Math may become inconsistent if we change one piece but still interprest the result as the exact gradient of the original loss
	
	For our neural ODE, the parameter gradient is
	$$\frac{dL}{d\theta} = \int_{t_0}^{t_f} a(t)^\top \frac{\partial f_\theta}{\partial \theta} \,dt.$$ and let's say we want to make the gradient less dominated by the dynamics Conceptually we want something like
	$$\frac{dL}{d\theta}_{\text{modified}} = \int_{t_0}^{t_f} \rho(z,t,\theta) \, a(t)^\top \frac{\partial f_\theta}{\partial \theta} \,dt,$$
	where $\rho$ is a normalization factor like
	$$\rho = \frac{1}{\|f_\theta(z,t)\|+\epsilon}.$$
	
	The issue - we can modify the gradient, but then it may no longer be the exact gradient of the original loss. Not automatically bad - optimization methods often use modified gradients like clipping, normalization, preconditioning, etc.
	
	Safest thing to modify is the parameter-gradient accumulation term, not the state adjoint.
	E.g. change the standard
	$$g_\theta = \int_{t_0}^{t_f} a(t)^\top \frac{\partial f_\theta}{\partial \theta} \,dt.$$
	to the modified
	$$g_\theta^{\text{norm}} = \int_{t_0}^{t_f} \rho(t) a(t)^\top \frac{\partial f_\theta}{\partial \theta} \,dt.$$
	which keeps $a(t)$ as the correct sensitivity of the loss wrt the state. It just changes how much each time contributes to the parameter update. 
- forward dynamics
- terminal condition
- adjoint has jumps for losses at multiple observation times - don't change these. We can reweight the loss jumps by changing the loss, not by changing the jump terms. Example: $L=\sum_k w_k \ell_k$ would give $a(t_k^-) = a(t_k^+) + w_k \frac{\partial \ell_k}{\partial z(t_k)}.$ which is mathematically valid becuase it corresponds to a modified loss
- don't break sign convention

repo uses h(t) instead of z(t)


Before tomorrow:
- [ ] take repo, generate orbital data and test idea and see where to implement it
- [ ] 
# June 22
- Can we jsut add this function to the loss function? if not, we need to dig into diffrax
- look at the dynamics and diff eq 
- We want to know how to actualy navigate what is happening in diffrax
- check parallelization of adjoint analysis
- script for first order test tomorrow 
- thursday friday - we can talk about swath foe xperiments and possibly do some refactoring
- clone diffrax - have claude strip it down to bare bones. no fancy adjoint methods, controlled odes, etc. this could become unit test that needs to be preserved. 
- lead analytics 
- do some tidying 
- begin writing out the readme


The scripts are too long. There is a lot of formatting. We mainly just want to see what the methods etc are. Things like formatting should be hidden from user
We have a lot of if statements - these should be methods
in particular, `train_model` is very long
everything is very dense
do unit testing. make dummy data to run things very quickly and see if the plots look how we want
harnesses - workflow you want executred every time you run your code. this would 
come up with unit tests for things in repository. claude - set line limits for scripts, if there are enough private methods in a script it should be a class, set line scripts for methods, etc. 
these tools value volume - we want the simplest experiments 
look up ponytail - use fewest number of lines. 
separate user facing front end from nitty gritty getting formatting correct. we don'tj need these for the academic side of things 

Our goal is to normalize the loss function by the model dynamics such that we are not placing importance only on the high dynamics areas. We also w

BacksolveAdjoint vs RecursiveAdjoint (aka opt-disc vs disc-opt)
`BacksolveAdjoint` solves the continuous adjoint ODE backwards, while `RecursiveCheckpointAdjoint` tells JAX to differentiate directly through the numerical solver steps, i.e. backprop through the for-loop of the integrator (what does this mean?).

In optimise-then-discretize, we treat the adjoint equation as a continuous mathematical object. We derive our adjoint equations analytically, and then solve those numerically. The gradient we get is an approximation to the true continuous sensitivity, and numerical is introduced from the backward solve. Conversely, in discretise-then-optimise, we treat the solver as a fixed computational graph and we diffferentiate that graph using reverse-mode AD.

Example of computational graph. Let's say the solver (Tsit5 and PID controller) takes a seuqnece of adaptive steps, and it takes N steps to go from t0 to t1. Our forward pass looks somethig like:
```
y0  →  [Tsit5 stage evals + error estimate + PID step accept/reject]  →  y1
y1  →  [Tsit5 stage evals + error estimate + PID step accept/reject]  →  y2
...
y_{N-1}  →  [...]  →  y_N ≈ y(t1)
```

`RecusriveCheckpointAdjoint` tells JAX to differentiate this ocmposition using the chain rule with no approximations. 

```
dL/dθ = (dL/dy_N) · (dy_N/dy_{N-1}) · ... · (dy_1/dy_0) · (dy_0/dθ)
```

Each factor `dy_k/dy_{k-1}` is the Jacobian of one solver step, computed exactly by JAX's AD through Tsit5's arithmetic.




Can we return to the 2BP
# June 17/18

Issues in github can become context for the agent once we have 

- reproduce on a different dynamical system (exponential growth? spring-mass damper might be too similar)
- can we desensitize the adjoint to time? 
- do we want a model that focuses on periapsis or can we weight this differently? we can probably hack this - take this term, normalize by r^2. maybe divide by 1/dynamics(t) we is agnositic of the dynamics
- what if we tried changing the loss to something like mse/dynamics?
- go back to most vanilla case - probably try full orbit but don't discount segments
- training strat - maybe you have a uniform distribution that kinda defines what is acceptable 
# June 11
We want gradient of loss wrt each individual training arc
next week - srtnog understandigm of what each term in the integrand means
we think second term is where the modeling energy is getting placed 

test wit claude - does this relate to other systems like lspring-mass system 


It stands for **Frobenius norm**. For a matrix $A$, it's defined as:

$$|A|_F = \sqrt{\sum_{i,j} A_{ij}^2}$$

It's just the square root of the sum of all squared entries — effectively treating the matrix as a flat vector and taking its Euclidean norm. Here $A = \partial f_\theta / \partial \theta$ is a $6 \times P$ matrix (6 output dimensions, $P$ parameters), so $|\cdot|_F$ gives a single scalar summarizing the overall magnitude of parameter sensitivity at that point

Our adjoint represent how sensitive we are 

**Orbit gradient plots (Figures 1 & 2)** Colored trajectory showing $|\partial L_t / \partial \theta|$ at each orbital position, for 4 different ICs. Tells you _where on the orbit_ the model receives gradient signal. If apoapsis is consistently dark, the parameters active there are undertrained regardless of how many epochs you run. Separate train/test figures show whether the pattern generalizes.

**Signal vs. radius/time/acceleration (Figure 3)** Same gradient norm plotted against $|r|$, time, and $|a_\text{true}|$ as x-axes, overlaid with accumulated error and instantaneous error. The point is to identify the _causal variable_ — does gradient signal drop because radius is large, because acceleration is small, or both? Separating accumulated error (drift) from instantaneous error (local model fidelity) tells you whether apoapsis is bad because the model can't learn it, or because error drifts into it from elsewhere.

---

**`analyze_adjoint.py`**

**$|\partial f / \partial \theta|_F$ (row 1)** Frobenius norm of the Jacobian of the vector field with respect to parameters. This measures how sensitive the model _output_ is to its weights at each orbital location, independent of any loss. Low at apoapsis means gradient descent literally cannot move those weights much regardless of how large the loss gradient $a(t)$ is — the two multiply together, so if either is zero, $dL/d\theta$ is zero there.

**$|a(t)|$ (row 2)** The adjoint norm — how much the loss changes if you perturb the state at time $t$. This is set by the loss function and the dynamics Jacobian $\partial f / \partial x$ propagating sensitivity backward. Tells you whether the loss is even "aware" of what the model does at each orbital location. If this is small at apoapsis, changing the loss function (e.g. `percent_error_weighted`) is the right lever.

**Integrand $|a(t)^\top (\partial f / \partial \theta)|$ (row 3)** The actual per-location contribution to $dL/d\theta = \int a(t)^\top (\partial f/\partial \theta), dt$. This is the bottom line — if it's small somewhere, that location contributes nothing to the parameter update. Rows 1 and 2 diagnose _why_ it's small (model insensitivity vs. loss insensitivity), but row 3 is the direct consequence.

**$\text{tr}(\partial f_\theta / \partial x)$ divergence (row 4)** For true 2BP this is exactly zero everywhere — the flow is Hamiltonian and volume-preserving. If the learned model has a consistently nonzero trace, it has learned a dissipative or expansive vector field even if its trajectory rollout looks fine. Negative divergence directly causes adjoint decay (phase space is being contracted), which suppresses $|a(t)|$ in row 2. This is the diagnostic for whether adjoint decay is a _model quality problem_ rather than a loss or dynamics property.

---

**Does post-training matter?**

For diagnosis, no — you want to understand what happened, not intervene. But it does mean these plots can't guide individual gradient steps. Their value is in informing _structural_ decisions: which loss function to use, whether the model architecture can even represent the right Jacobian, and whether adding more training data or epochs will help at all (if $|\partial f/\partial \theta|_F$ is structurally zero at apoapsis, no loss change fixes it).

# June 5
![[Pasted image 20260605131822.png]]

# May 27

can we look at the jacobi drift between specific parts of the orbit? like if we break into 10 segments, maybe we track well in some of the segments but not others.
color bar for the drift relative to the orbits?

June 22-26 working with John we will iterate on all of the visualizations and the deliverables and experiments that need ot be done and then write for the next month
Before then:
- Refine error metrics
- We need error metrics for local regions - if we just have 4 lyapunov orbits, why are they not being modeled well? where is the boundary at which point our model begins to break? distance from reference trajectory? almount of training data? training mechanics? how do we define boundaries where it breaks - wualitative and quanititative metrics. where does this happen and why. Ideally, we can see this in multiple locations. then what do we ened to do to model, training process, etc to overcome this? 
Target July 22 for journal paper submitted deadline

adversarial agentic question asking

Worst case - if journal paper is not full
Abstract by first of new semester




I want to discuss a CR3BP error metric and a schedule for the summer.

To bring you back into the neural ODE headspace, we are returning to  working on improving our vanilla neural ODEs for CR3BP. The first question to answer is what makes a good model? In order to know if neural ODEs can recover CR3BP dynamics well, we need to have an idea of what this actually means.

For the metrics, my goals is to have a few different pieces of information available at a glance for model diagnostics while keeping required compute time low. We want both qualitative and quantitative diagnostic measures. 

My thought was that for a model to be good, it should model scientifically important regions well. Errors in these regions will indicate poor model behavior. To determine what is scientifically important, I would like to consider proximity to Lagrange points, and proximity to stable and unstable manifolds.

I ranked the Lagrange points in order of scientific significance in the Earth-Moon system. L1 and L2 are the most important being near the moon, L4 and L5 are symmetric and the only stable lagrange points so I have ranked those lower since they could still be used for observation, and L3 is generally not used much in space missions.

We also need to know what proximity to the Lagrange points means - 



![[Pasted image 20260527115825.png]]

# Performance Review

Adam Grant - psychology guy

For agentic coding - build plans first before any coding (something like two page brain dump. Problem to solve, where I'm at. This is what I have, this is kinda where I want to go with it. Want to build a diagnostic - can you ask me questions and give a couple suggestions. We'll iterate before they write any code. You can tell it to be adversarial and call you out if something seems like I don't know. If you highlight a certain part, refine that). Use wiht plots - think they're useful, want a strict metric., look at literature etc. 
Distill weekly updates into three minute updates. can read off a script. Consolidate up front. 

skills:
- question formulation
- experiment design
- journal paper
- plan!!
- construct another research project of choosing for space flight mechanics in 2027
- git!!!
- some form of metric persisting beyond journal


# May 8 - Group meeting
- Performance reviews next Wednesday - fill out by noon on Tuesday
- Think about suggestions for change
- Summer:
	- everyone is here
	- John unavailable May 18-22 (vacation)
	- John available remotely but out of country:
		- July 27-31 ASC 
		- August 3-14
	- Keep group meeting time available for summer but we might not use it
	- Program meetings expected to continue over the summer 
	- Research weeks will be held again - for one week, spend four hours per day together
		- candidate schedule: June 22-26 me
	- Let John know when week off is, remote week is fine within reason but let him know
	- Send John figures to showcase if we want

Journal paper:
- long term, construct the linear latent model of the CR3BP but that is step c or d.
- latent ODE - encoding energy level is not necessarily the way to go. John agrees that it makes sense for the long term linear latent goal. 
- we should be able to poorly learn full CR3BP without prior information encoded, then if we include more info we should be able to better learn the dynamics. 
- can we use vanilla neural ODEs to represent the CR3BP and how do we assess if we have successfully recovered those dynamics. We have not yet shown if we can or cannot do this with the vanilla neuralODE. Definitions of the metrics we're using, plots that we are using. 

- in notes, we should know what question we are specifically asking
- what metrics define high quality tools across the different systems we have developed
	- using these metrics - how can we quantitatively assess if a design improved or degraded model quality
	- from there, vary training data etc to determine what is needed for a certain level of performance
	- how much data do we need to do something interesting downstream (discover periodic orbits)
- work on doing weekly checkins pre meeting with John to make sure thoughts are well organized and update will result in productive conversation. A lot of times I am getting lost in the weeds and we end up waxing poetic. 
- summer - defining questions leading to paper
- what metrics are useful for evaluating CR3BP model metrics? where does it matter that these are performing well? we need to decide which are the top ones, not just a laundry list of choices. 




Goals for today: the latent ODE models are losing information about the physical structure of the 2BP dynamics. The latent space is not structured well, so we consequently cannot accurately learn latent dynamics. Today I want to focus on systematically implementing changes to improve the latent structure and overall model. And taking good notes!

## The Problem
The latent ODE models are not preserving physical structure of the 2BP dynamics. 
![[Pasted image 20260505113357.png]]
![[Pasted image 20260505113443.png|500]]

![[Pasted image 20260505113454.png]]
![[Pasted image 20260505113429.png]]

==What am I looking at, why do I care==
## Possible Solutions
1. More intelligent design of training data representation:
Recall that when we were focused on improving the neuralODE results, in addition to scaling the data as is standard such that $\mu^*=1$, we also spent time designing feature layers that allowed the model to more easily learn the dynamics of the problem. Typically, our feature layer is $[1/r,x/r/y/r,z/r,v_x,v_y,v_z]$ . This encodes the inverse relationship between gravity and distance. The direction components tell the network which way to point the acceleration. 

The encoder acts on the training data and converts it to some latent representation before passing the latent representation through the MLP which learns latent dynamics. Thus, the notion of a feature layer within the dynamics MLP no longer exists in the same way as in the vanilla neural ODE because we do not have a representation of the training data in physical space at this point. This isn't a big deal for the latent ODE - we can just transform the training data before passing it through the model. Feature layers are just more convenient.

So, we want to represent the data in a way that gives it a more meaningful latent structure. How do we want our orbits to be structured? Ideally, the model should learn to group orbits in a way that groups like energy levels, SMA, and eccentricity. Recall that when we were designing the vanilla neural ODE, we did not want to encode energy because we wanted a proof of concept showing that the models can learn dynamics without it knowing that energy is conserved. However, I don't think we need to maintain this restriction anymore because we are moving towards the goal of encoding CR3BP dynamics using a linear latent model to perform downstream analysis, rather than learning dynamics from scratch. So now, we can include energy in our training data and/or in the loss function. Is there a way that we can encourage further intelligent grouping in our model?

2. A hyperparameter search, particularly of variables like the latent_size and hidden_size, may show if the latent space is not expressive enough or alternatively too large to the point that we are unable to generalize well. I don't necessarily expect this to solve everything, but I would like to gain a better understanding of how to size the different parts of the architecture and how important this is or is not.
3. The encoder itself may need improvement/adjustment when we use it (but recall we are also testing an MLP to learn the initial latent condition since our data is not noisy). One known failure mode here is KL collapse, where the encoder learns to ignore its input entirely and outputs the same distribution regardless of the observed trajectory. This would explain the lack of physical structure in the latent space. A quick diagnostic we can check is whether the KL term in the loss goes to zero during training, and whether the posterior mean and variance are roughly the same across all inputs.
4. Can we design loss functions that encourage physical structure in latent space? With vanilla neural ODEs, we avoided penalties like energy drift because we wanted to see if the model could learn the dynamics without this knowledge as a proof of concept. However, we are now going a different direction with the goal of developing a linear latent model eventually for the CR3BP. I think it is now appropriate to bake in knowledge like energy being conserved for 2BP, or JC being conserved in the CR3BP. Perhaps including this in the loss will encourage a better structured latent space.


## Implementing latent model changes
### 1 - Testing feature layers
Let us begin with our most stripped back latent ODE model. This includes an mlp encoder to directly map y0 to a latent state, which is suitable since we have clean data with no noise. The latent dynamics are parameterized nolinearly via a free MLP. The decoder is linear. So, our full pipeline is currently `y[0] → MLP → (μ, log σ) → sample z_0 ~ N(μ, σ) → MLP (latent_to_hidden) → h_0 → MLP ODE → h(t) → linear decode → ŷ(t)`

I understand why this came to be since it was based on Kidger's model, but looking at it now I have a couple issues with it. If we are using an MLP in place of a GRU as an encoder in the case of noiseless data, there is no reason to sample z_0 from a distribution then have *another* MLP transforming from latent to hidden (both of these design choices were artifacts of the GRU encoder version). Let's take a closer look at some of the encoder related design choices.

---
#### *Tangent - double checking some of the encoder related code*
*Question - I currently have the encoding mlp map the first observed state to (mean, logstd) of z0. Does this make sense?* 

*Answer - We shouldn't need a distribution since it's a 1:1 mapping. This seems unlikely to be hurting anything significantly, but still is unnecessary. Also, our full pipeline is more complicated than it needs to be if we are just using an MLP to map y\[0\] to an initial hidden state. Is there a good reason for this, or is this just redundant? In fact, is it just redundant or is it harmful? How would we have a mean and std for our mlp encoder when we don't even have noise for which we need to estimate a posterior. We already have a flag to turn this off an dI believe it was only included to match the formatting of the GRU pipeline, but we should probably just remove the option to use an MLP encoder to estimate a posterior.*
*<details>**
**Prior — your belief about `z_0` before seeing any data. In the latent ODE this is N(0,I) — a default assumption that latent initial conditions are roughly centered near zero with unit variance.**
**Posterior — your updated belief about `z_0` after seeing the observation `y[0]` (or a trajectory segment). This is what the encoder outputs: given this specific observation, where in latent space did this trajectory probably start?**
**The KL term penalizes the posterior for deviating from the prior. The tension is:**
**- Reconstruction loss pushes the posterior to be **specific** — encode as much information as possible about this particular trajectory**
**- KL term pushes the posterior back toward the prior — don't deviate too far from N(0,I)**
**</details>*
*Question - Kidger's pipeline is `ys + ts → GRU → hidden → **Linear** (hidden_to_latent) → (μ, log σ) → sample z_0 → **MLP** (latent_to_hidden) → h_0 → MLP ODE → h(t) → **Linear** (hidden_to_data) → ŷ(t)` - why is there both an encoder (GRU) and latent to hidden mlp?*

 *Answer - In Kidger's design there are actually three separate spaces:*
- ***hidden space** (`hidden_size`) — the GRU's internal state, sized to give the RNN enough capacity to process the sequence*
- ***latent space** (`latent_size`) — the VAE bottleneck, intentionally small to force compression*
- ***hidden space again** (`hidden_size`) — the ODE state space, sized to give the dynamics enough capacity*
*So the flow compresses down and then expands back up:*
*`GRU hidden (hidden_size) → linear → z (latent_size) → MLP → h_0 (hidden_size) → ODE`*
*The bottleneck through `z` is what forces the model to learn a compact, meaningful representation. If you went directly from the GRU hidden state to the ODE initial condition you'd skip the compression entirely and lose the structured latent space.*
*In your case with `y0_mlp`, the GRU is replaced by an MLP that goes directly from `y[0]` to `(μ, log σ)` in latent space, so the first compression step is handled by that MLP instead. The `latent_to_hidden` expansion back up to ODE state space is the same.*

*Question - why is there a linear layer from hidden to latent but an MLP from latent to hidden?*

*Answer -*
***Hidden → latent (Linear):** The GRU hidden state already contains a rich nonlinear representation of the trajectory — the GRU cells themselves are nonlinear. So a simple linear projection down to `(μ, log σ)` is sufficient. Adding an MLP here would be redundant nonlinearity on top of what the GRU already did.*
***Latent → hidden (MLP):** Going the other direction, `z_0` is a small compressed vector and you need to expand it into a rich enough initial condition to drive the ODE dynamics. A linear map from a small latent space to a larger hidden space is severely limited — it can only produce initial conditions that lie on a low-dimensional linear subspace of `h` space. An MLP can produce initial conditions anywhere in `h` space, giving the ODE dynamics much more to work with.*

*In short: compression can be linear because the nonlinearity already happened in the GRU. Expansion needs to be nonlinear because you're asking a small vector to seed a high-dimensional dynamical system.*

*Question - due to how we followed this logic, when we use an MLP to encode , we have `y[0] → MLP → z_0 → MLP (latent_to_hidden) → h_0 → MLP ODE → h(t) → linear decode → ŷ(t)`. The MLP between z_0 and h_0 seems redundant, doesn't it?*

*Answer - yes, that is redundant. You could collapse the two into a single MLP going directly from `y[0]` to `h_0` and nothing would be lost. The `latent_to_hidden` only has a distinct role when there's a meaningful bottleneck at `z` — either because the stochastic encoder forces compression via the KL term, or because you want to sample from the prior at test time (drawing `z ~ N(0,I)` and decoding). If you're doing neither, `z_0` is just an intermediate activation with no special meaning and the two MLPs are redundant. Single MLP now maps encoder features directly to `h0` (bypassing the redundant `z → h` bottleneck) when `encoder_type=y0_mlp` and `use_stochastic_encoder=false`, and the encoder feature layer now includes specific orbital energy as an input feature.*

*Question - do we need to include time in our encoder? Our dynamics are autonomous, but I'm not sure if/how it affects the* 

*Answer - it isn't necessary to include if the timesteps are equal since timesteps are processed sequentially, e.g. the recurrence itself is the temporal mechanism. However, if dt is not fixed, we need to pass time to the encoder. This is only true for the GRU though, not the MLP since the MLP will directly encode y\[0\] to h\[0\]. Recall that when we generate our 2BP training data, we sample at equal true anomalies so we do not have equal dt.*

*Follow up - does it make sense to directly encode from y\[0\] to h\[0\] in the case where we have noiseless data? We don't need to use a GRU encoder, but should we still include some notion of time history of data when we get our initial hidden state to pass into the dynamics mlp?*

*Answer - y[0] is theoretically sufficient since the initial state fully determines the entire trajectory, so we do have mathematical justification. However, there may be a practical advantage to using a short prefix of observations. The MLP has to learn the appropriate mapping from the initial condition which may be difficulty. If we have a GRU, it could make it easier to identify the orbit's energy and angular momentum.*

*Follow up - can we use a GRU that does not sample form a distribution? e.g. we maintain the idea of observing a few time samples in our encoder to bake in information about the orbital energy, but we do not bake in stochasticity?*

*Answer - we can do this. We don't need to learn a distribution using the GRU - we can just directly learn the mapping from physical space to hidden space and skip the sampling and `latent_to_hidden` MLP.  Now all three encoder types (`y0_mlp`, `prefix_gru`, `backward_gru`) output `h0` directly when `use_stochastic_encoder: false` — no `(μ, log σ)` split,  no `latent_to_hidden` step. The stochastic path is unchanged.*

---

Important details from the tangent:
- there is no reason for deterministic encoders (whether it be MLP, GRU) to be mapping to a distribution from which to sample z_0
- Similarly, there is no reason to have a deterministic encoder followed by a latent_to_hidden mlp
- Our updated pipelines

**Stochastic (`use_stochastic_encoder: true`):** `y[0] → feature layer → MLP or GRU → (μ, log σ) → sample z_0 ~ N(μ, σ) → MLP (latent_to_hidden) → h_0 → MLP ODE → h(t) → linear decode → ŷ(t)`

**Deterministic, old (`use_stochastic_encoder: false`, before today):** `y[0] → feature layer → MLP or GRU → (μ, log σ) → take μ as z_0 → MLP (latent_to_hidden) → h_0 → MLP ODE → h(t) → linear decode → ŷ(t)`

**Deterministic, new (`use_stochastic_encoder: false`, after today):** `y[0] → feature layer → MLP or GRU → h_0 → MLP ODE → h(t) → linear decode → ŷ(t)`

Everything is stripped back and we are ready to begin.

---

**Baseline - no feature layer (cartesian_only)**
> [!note]- config.yaml
> 
> ```python
> wandb:
>   group: "test-latent-v2"
> 
> data:
>   dataset_name: ["complex_TBP_planar_100_train"]
>   problem: "2BP"
> 
> parameters:
>   # Train on full trajectories from the start — no curriculum.
>   length_strategy: [[[0.0, 1.0],[0.0, 1.0],[0.0, 1.0],[0.0, 1.0],[0.0, 1.0]]]
>   lr_strategy: [[1e-2,1e-2,1e-2,1e-2,1e-2]]
>   steps_strategy: [[500, 500, 500, 500, 1000]]
>   segment_length_strategy: [[18, 36, 90, 180, 360]]
> 
>   width: [16]
>   depth: [2]
>   train_val_split: 0.8
>   batch_size: [128]
>   num_trajs: -1
>   seed: [2345]
> 
>   rtol: 0.000001
>   atol: 0.00000001
> 
>   latent_size: [8]           # VAE bottleneck (z space)
>   latent_hidden_size: [16]   # ODE hidden space (h space), matches Kidger hidden_size
>   latent_width: [16]
>   latent_depth: [2]
>   latent_activation: [softplus]       # Kidger uses softplus in MLP body
>   latent_final_activation: [tanh]      # drift MLP final activation, matches Kidger
>   latent_encoder_final_activation: [linear]  # encoder head is linear, no squashing of (μ, log σ)
>   # latent_consistency_weight: [1e-1]
> 
>   latent_dynamics_type: [nonlinear]
>   # latent_linear_init_scale: [0.02]
>   latent_drift_scale: [1.0]           # Kidger initializes scale to 1.0
>   latent_residual_scale: [1.0]
> 
>   latent_segment_stride_ratio: [1.0]
> 
>   latent_encoder_type: [y0_mlp] # options are "prefix_gru", "backward_gru", "y0_mlp"
>   latent_encoder_use_time: [true]     # concatenate dt with features in GRU
>   latent_prefix_length: [18]
>   latent_encoder_feature_layer: [cartesian_only]
> 
>   # Stochastic encoder with KL — matches Kidger's VAE setup.
>   # latent_reconstruction_type: [kidger]
>   latent_use_stochastic_encoder: [false]
>   latent_kl_weight: [1]
>   latent_kl_schedule: [linear_warmup]
>   latent_kl_weight_start: [0.0]
>   latent_kl_warmup_steps: [-1]   # auto: clamp(fraction * total_steps, min, max)
>   latent_kl_warmup_fraction: [0.2]
>   latent_train_use_mean: [false]
>   latent_eval_use_mean: [true]
> 
>   # Minimal regularization — let the model learn freely first.
>   latent_initial_state_weight: [0.0]
>   latent_orbit_closure_weight: [0.0]
>   latent_state_norm_weight: [0.0]
>   latent_drift_norm_weight: [0.0]
>   latent_linear_stability_weight: [0.0]
> 
>   # Phase dynamics off.
>   latent_use_phase_dynamics: [false]
>   latent_phase_supervision_weight: [0.0]
> 
>   # Aux orbital head off.
>   latent_aux_orbital_weight: [0.0]
>   latent_aux_orbital_mu: [1.0]
>   latent_aux_orbital_use_head: [false]
> 
>   latent_log_gradient_stats: [false]
>   latent_log_dimension_mpe: [false]
>   latent_grad_clip_total_norm: [0.0]
>   latent_grad_clip_encoder_norm: [0.0]
>   latent_grad_clip_dynamics_norm: [0.0]
>   latent_grad_clip_decoder_norm: [0.0]
>   latent_grad_clip_aux_norm: [0.0]
> 
>   latent_solver_max_steps: [8192]
>   latent_log_every: [50]
>   latent_plot_orbits: [[0, 1]]
>   latent_checkpoint_dir: ["files/models"]
>   latent_energy_drift_weight: 0.0   # tune this
>   latent_energy_drift_mu: 1.0       # normalized 2BP
> ```

The feature layer is `cartesian_only`, and our encoder is `y0_mlp`. 
run_id: edymbm5d

Generalization gap:
![[Pasted image 20260506120534.png|500]]


![[Pasted image 20260506114313.png]]
![[Pasted image 20260506113341.png]]
![[Pasted image 20260506113412.png]]
![[Pasted image 20260506113423.png]]

Takeaways: 
- The encoder and decoder are working well in that reconstruction loss is low- we find a good h0 and the reconstruction is mostly correct. However, this does not mean that the hidden representation is a good one - they have no notion of the dynamics of the problem
- The ODE dynamics diverge significantly in past test cases, and the residual acceleration field is largely random
- The bottleneck is the dynamics MLP. We learn a resonable mapping between state space and hidden space, but the hidden space ODE is not learning to propagate h(t) in a way corresponding to physical orbital motion.
- The hidden space has no particular structure due to the cartesian coordinates (and maybe the small hidden size?)

---
**v1 - cartesian_only features, prefix_gru deterministic encoder**
run_id: ![[Pasted image 20260506121512.png]]
Let's try still using cartesian features, but use a prefix GRU encoder (deterministic) to see if/how this changes the structure of the hidden space

> [!note]- config.yaml
> ``` python
> wandb:
>   group: "test-latent-v2"
> 
> data:
>   dataset_name: ["complex_TBP_planar_100_train"]
>   problem: "2BP"
> 
> parameters:
>   # Train on full trajectories from the start — no curriculum.
>   length_strategy: [[[0.0, 1.0],[0.0, 1.0],[0.0, 1.0],[0.0, 1.0],[0.0, 1.0]]]
>   lr_strategy: [[1e-2,1e-2,1e-2,1e-2,1e-2]]
>   steps_strategy: [[500, 500, 500, 500, 1000]]
>   segment_length_strategy: [[18, 36, 90, 180, 360]]
> 
>   width: [16]
>   depth: [2]
>   train_val_split: 0.8
>   batch_size: [128]
>   num_trajs: -1
>   seed: [2345]
> 
>   rtol: 0.000001
>   atol: 0.00000001
> 
>   latent_size: [8]           # VAE bottleneck (z space)
>   latent_hidden_size: [16]   # ODE hidden space (h space), matches Kidger hidden_size
>   latent_width: [16]
>   latent_depth: [2]
>   latent_activation: [softplus]       # Kidger uses softplus in MLP body
>   latent_final_activation: [tanh]      # drift MLP final activation, matches Kidger
>   latent_encoder_final_activation: [linear]  # encoder head is linear, no squashing of (μ, log σ)
>   # latent_consistency_weight: [1e-1]
> 
>   latent_dynamics_type: [nonlinear]
>   # latent_linear_init_scale: [0.02]
>   latent_drift_scale: [1.0]           # Kidger initializes scale to 1.0
>   latent_residual_scale: [1.0]
> 
>   latent_segment_stride_ratio: [1.0]
> 
>   latent_encoder_type: [prefix_gru] # options are "prefix_gru", "backward_gru", "y0_mlp"
>   latent_encoder_use_time: [true]     # concatenate dt with features in GRU
>   latent_prefix_length: [18]
>   latent_encoder_feature_layer: [cartesian_only]
> 
>   # Stochastic encoder with KL — matches Kidger's VAE setup.
>   # latent_reconstruction_type: [kidger]
>   latent_use_stochastic_encoder: [false]
>   latent_kl_weight: [1]
>   latent_kl_schedule: [linear_warmup]
>   latent_kl_weight_start: [0.0]
>   latent_kl_warmup_steps: [-1]   # auto: clamp(fraction * total_steps, min, max)
>   latent_kl_warmup_fraction: [0.2]
>   latent_train_use_mean: [false]
>   latent_eval_use_mean: [true]
> 
>   # Minimal regularization — let the model learn freely first.
>   latent_initial_state_weight: [0.0]
>   latent_orbit_closure_weight: [0.0]
>   latent_state_norm_weight: [0.0]
>   latent_drift_norm_weight: [0.0]
>   latent_linear_stability_weight: [0.0]
> 
>   # Phase dynamics off.
>   latent_use_phase_dynamics: [false]
>   latent_phase_supervision_weight: [0.0]
> 
>   # Aux orbital head off.
>   latent_aux_orbital_weight: [0.0]
>   latent_aux_orbital_mu: [1.0]
>   latent_aux_orbital_use_head: [false]
> 
>   latent_log_gradient_stats: [false]
>   latent_log_dimension_mpe: [false]
>   latent_grad_clip_total_norm: [0.0]
>   latent_grad_clip_encoder_norm: [0.0]
>   latent_grad_clip_dynamics_norm: [0.0]
>   latent_grad_clip_decoder_norm: [0.0]
>   latent_grad_clip_aux_norm: [0.0]
> 
>   latent_solver_max_steps: [8192]
>   latent_log_every: [50]
>   latent_plot_orbits: [[0, 1]]
>   latent_checkpoint_dir: ["files/models"]
>   latent_energy_drift_weight: 0.0   # tune this
>   latent_energy_drift_mu: 1.0       # normalized 2BP
> 
> ```

![[Pasted image 20260506120507.png|500]]
![[Pasted image 20260506121517.png]]
![[Pasted image 20260506120947.png]]

![[Pasted image 20260506120958.png]]
![[Pasted image 20260506121108.png]]
![[Pasted image 20260506121050.png|500]]
![[Pasted image 20260506124538.png]]

Takeaways:
- let's stick with the y0_mlp encoder for now
- 

---
**v2 - sph_4D_rinv_vel, y0_mlp encoder**
We also want a baseline of our behavior when using the same feature layer that we did in the conference paper vanilla neuralODE.
run_id: kvb9n4f7
> [!note]- config.yaml
> wandb:
  group: "test-latent-v2"
> ``` python
> data:
>   dataset_name: ["complex_TBP_planar_100_train"]
>   problem: "2BP"
> 
> parameters:
>   # Train on full trajectories from the start — no curriculum.
>   length_strategy: [[[0.0, 1.0],[0.0, 1.0],[0.0, 1.0],[0.0, 1.0],[0.0, 1.0]]]
>   lr_strategy: [[1e-2,1e-2,1e-2,1e-2,1e-2]]
>   steps_strategy: [[500, 500, 500, 500, 1000]]
>   segment_length_strategy: [[18, 36, 90, 180, 360]]
> 
>   width: [16]
>   depth: [2]
>   train_val_split: 0.8
>   batch_size: [128]
>   num_trajs: -1
>   seed: [2345]
> 
>   rtol: 0.000001
>   atol: 0.00000001
> 
>   latent_size: [8]           # VAE bottleneck (z space)
>   latent_hidden_size: [16]   # ODE hidden space (h space), matches Kidger hidden_size
>   latent_width: [16]
>   latent_depth: [2]
>   latent_activation: [softplus]       # Kidger uses softplus in MLP body
>   latent_final_activation: [tanh]      # drift MLP final activation, matches Kidger
>   latent_encoder_final_activation: [linear]  # encoder head is linear, no squashing of (μ, log σ)
>   # latent_consistency_weight: [1e-1]
> 
>   latent_dynamics_type: [nonlinear]
>   # latent_linear_init_scale: [0.02]
>   latent_drift_scale: [1.0]           # Kidger initializes scale to 1.0
>   latent_residual_scale: [1.0]
> 
>   latent_segment_stride_ratio: [1.0]
> 
>   latent_encoder_type: [y0_mlp] # options are "prefix_gru", "backward_gru", "y0_mlp"
>   latent_encoder_use_time: [true]     # concatenate dt with features in GRU
>   latent_prefix_length: [18]
>   latent_encoder_feature_layer: [sph_4D_rinv_vel]
> 
>   # Stochastic encoder with KL — matches Kidger's VAE setup.
>   # latent_reconstruction_type: [kidger]
>   latent_use_stochastic_encoder: [false]
>   latent_kl_weight: [1]
>   latent_kl_schedule: [linear_warmup]
>   latent_kl_weight_start: [0.0]
>   latent_kl_warmup_steps: [-1]   # auto: clamp(fraction * total_steps, min, max)
>   latent_kl_warmup_fraction: [0.2]
>   latent_train_use_mean: [false]
>   latent_eval_use_mean: [true]
> 
>   # Minimal regularization — let the model learn freely first.
>   latent_initial_state_weight: [0.0]
>   latent_orbit_closure_weight: [0.0]
>   latent_state_norm_weight: [0.0]
>   latent_drift_norm_weight: [0.0]
>   latent_linear_stability_weight: [0.0]
> 
>   # Phase dynamics off.
>   latent_use_phase_dynamics: [false]
>   latent_phase_supervision_weight: [0.0]
> 
>   # Aux orbital head off.
>   latent_aux_orbital_weight: [0.0]
>   latent_aux_orbital_mu: [1.0]
>   latent_aux_orbital_use_head: [false]
> 
>   latent_log_gradient_stats: [false]
>   latent_log_dimension_mpe: [false]
>   latent_grad_clip_total_norm: [0.0]
>   latent_grad_clip_encoder_norm: [0.0]
>   latent_grad_clip_dynamics_norm: [0.0]
>   latent_grad_clip_decoder_norm: [0.0]
>   latent_grad_clip_aux_norm: [0.0]
> 
>   latent_solver_max_steps: [8192]
>   latent_log_every: [50]
>   latent_plot_orbits: [[0, 1]]
>   latent_checkpoint_dir: ["files/models"]
>   latent_energy_drift_weight: 0.0   # tune this
>   latent_energy_drift_mu: 1.0       # normalized 2BP
> ```
> 
> 

![[Pasted image 20260506125226.png|500]]

State error:
![[Pasted image 20260506125305.png]]

Test orbits:
![[Pasted image 20260506125322.png]]

Dynamics v training orbits:
![[Pasted image 20260506125341.png]]

True vs model implied acceleration:
![[Pasted image 20260506125401.png]]

Latent PCA:
![[Pasted image 20260506125417.png]]

Takeaways:
- The encoder/decoder is not as accurate once we change the feature layer
- PCA shows some structure, but not meaningful
- How can we encourage a meaningful latent structure? Do we need to change the model architecture? The feature layer? The loss function? I would imagine that for the 2BP, a meaningful structure would encode a notion of energy and angular momentum. Can we encourage this?

Possible solutions:
- Include orbital energy in training data. We are no longer trying to learn dynamics completely from scratch, so let's use the information we have available to us.
- Acceleration consistency loss. Add a loss term that penalizes `||J_dec(h) @ f_theta(h) - a_true(y)||` at decoded points along the trajectory. This directly forces the dynamics to produce the right acceleration when decoded.
- Metric loss on latent codes. Penalize `||z0_i - z0_j||` being large when the orbits are physically similar (same energy/SMA), and small when dissimilar. This directly shapes the latent geometry. But it only fixes the encoder structure — the dynamics can still be wrong.
- Penalize energy drift. 

---

Running questions:
- are there guidelines to sizing latent vs hidden size when we use a VAE?























# April 23
I had more plots than ended up in the report 
Section 4 
# April 15

be able to speak to transformers - don't need to process sequentially
do presentation at high level first. don't need full derivations
good for datasets with uncertainty - used for RL to construct datasets with some belief. connect to estimation and kalman filtering

**Metrics**
Encoder/decoder
- **RMSE position / RMSE velocity**: For each orbit, compute the per-timestep position error norm, then take RMSE **over timesteps**. The reported number is then the **mean of that per-orbit RMSE over all orbits**. So it's mean-over-orbits of RMSE-over-time.
- **Mean position MAPE**: For each orbit, compute `|pos_err| / |pos_true|` at each timestep, average **over timesteps**. Reported number is the **mean of that over all orbits**.
- **Mean accum pos error**: For each orbit, sum `pos_err * dt` **over timesteps** (a trapezoid-style integral over the arc). Reported number is the **mean of that integral over all orbits**.
Pointwise Dynamics
- **Mean accel error**: For each orbit, compute the % acceleration error at each of `n_grid` evenly-spaced arc points, average **over those timesteps** → one scalar per orbit (`mean_pct_errors`). Reported number is `np.mean(mean_pct_errors)` — **mean-over-orbits of mean-over-timesteps**.
- **Median accel error**: Same per-orbit scalars, but `np.median` over orbits instead of mean.
- **State-mismatch accel**: For each orbit, evaluate `a(ŷ_hat)` vs `a(y_true)` at each timestep, average the % error **over timesteps**. Reported number is the **mean of that over all orbits**.
Propagation
- **Mean final pos/vel error**: Take the error at the **last timestep only** for each orbit, then average **over orbits**. There is no time-averaging here — it's a single endpoint measurement per orbit.
- **Mean accum pos error**: Same cumulative integral as Section 1 but in physical units (km·s), taken at the final timestep, then averaged over orbits.
# April 1
We can beat down the uncertainty with GRU - especially good with noisy trajectories. We want an informative initial belief distribution 
Double check intuition with kidger and otherwise - he was talking about sparse dataset with unknown state. Determine if there is value in doing this 
Is there a reason why we would need to do this when we have perfect knowledge of the state? 
Diagnose the koopman stuff - this doesn't totally make sense
What do we need to understand why things are not behaving the way that they should - 
We want to evaluate specific parts of model performance:
- reconstruction (encoding/decoding)
- propagation
- dynamics
- mix of qualitative and quantitative
- focus on what errors exist - dynamics, encoder, etc. 
- define experiment which can quantify error, and then we can choose what metrics we want and fiddle with it
- what is broken, and what circumstances? circular orbit, etc. 
- can we decouple the encoder and decoder with the dynamics - what happens if we just sample from latent space (no encoder/decoder)
	- hold latent var 2 at 0, vary var 1. integrate, decode etc
	- is there an intuitive structure
	- every metric will have figure etc
Latent ODE:
![[Pasted image 20260401125841.png]]
Koopman (no segmentation):
![[Pasted image 20260401130205.png]]
run_id: b1s2lngc
![[Pasted image 20260401130424.png]]


# March 4
- Emphasize end application rather than nitty gritty details
- We are not locked into perturbation analysis
- confirm that initial conditions are the same for the latentODE wandb upload and postprocessing

Continuation methods:
- most intuitive is jacobian, but commonly we do pseudo-arc length. Whatever latent trajectory we're on, we have some associated arc length and we can only perturb in some direcitons - unstable and stable manifolds, or the purely periodic. Compute monodromy matrix, perturb toward periodic mode, constrain to some pseudo-arc length.
- There is probably a way to transfer from one type of periodic orbit to another
- figure out reconstruction issues since we will need pseudo-arc length
- how to do latent orbit stuff well uses branching logic
- challenges converting CR3BP to latent space - timescales and linearity. Different families take far longer than others. Would like to be able to gracefully handle this (Sundman transfer ?). Can we normalize such that all orbits can be propagated in the same fashion agnostic to geometric length. How to encourage linearity? How can we decrease extensive iterations, etc. 
- Journal - show that models can accurately represent large portions of CR3BP phase space. how well can neuralODEs represent the CR3BP system? So far we have explored a narrow space of this? Can we generalize to 3D? How well can we learn existing dynamics? If we can define metrics to define how covered our phase space is in 
- How much phase space is covered in test sets and how can we test that? 
- could we characterize the differences between the learned latent space vs original space
- add physics informed constraint and just say it's linear and see what the consequences are. it wouldj be must cheaper than integrating nonlinear. Get long orbits using linear form rather than integration
- Both will need us to know how good they are in critical regions of phases space
- come up with backup metrics, tests for qhat the quality is 
- do continuation and linearization in parallel, two or three studies and see what falls out

TODO:
- [ ] Review Jacobian stablity info
- [ ] Redo sensitivity analyses from conference
- [ ] Write up lessons learned and debugging info
- [ ] Better understanding of latent ode vs ode-rnn. How does ode-rnn relate to CDEs?

## Housekeeping
- Goddard Symposium discount code no longer valid; not attending
- Offered to be on the student panel for visit day
- SSPI abstract accepted
	- Abstract: This work investigates the potential of neural ordinary differential equations (neural ODEs) for modeling unknown perturbations acting on a spacecraft. Specifically, the proposed approach formulates the dynamics as a universal differential equation, where known physical laws define the nominal dynamics and a neural ODE learns a corrective term capturing unknown forces. Learning such corrective dynamics is particularly important for autonomous spacecraft operations, where discrepancies between baseline physics models and true system behavior can accumulate and degrade long-term prediction. In addition, we explore whether learning perturbative dynamics in a latent state space using latent ODEs can provide improved representation efficiency and robustness compared to modeling directly in the physical state space. As a proof of concept, we demonstrate the ability of the neural ODE to learn perturbative dynamics from data. Results indicate that neural ODE-based models show promise as accurate, data-driven surrogates for traditional spacecraft dynamics models, offering a flexible framework for representing complex and partially unknown dynamics in support of autonomous mission analysis and design.
	  - Report due by March 27th. There are no details?
- ASC - can Simon and I coauthor a paper?
	- Maybe we can use KANs for perturbation analysis?

## Sensitivity Studies

# February 24
Periodic orbits - continuation, differential correction, multiple shooting, etc. Can we apply those tools to the phase space learned by latentODEs. 
What if we lift dynamics to latent space that is larger, candidate transfers maybe. 
Example - doing transfer from L1 to L2 using intersection of Poincare maps. 
Latent ODEs might break - for example we only change delta v irl but there is coupling between where you are and where to actuate. 
Look for periodic orbits. 
Continue known periodic orbits in latent space and see what happens.
After generating a large number of these trajectories (all families, not just lyapunov). see if we can connect outputs with differential correct and enforce periodicity in the latent space. 
/github/copilot/instructions
include in agents file that agent is responsible for writing its own research notes. Use asked me for X, I tried this, it didn't work so I did this. e.g. transfers aren't conecting and I want to know why. tell it area where you think issue is, 
train latent ODE
look at eigenvectors - positive eigenvalue will have tangential trajectory?
confirm lessons learned on ASC dataset - WRITE THIS DOWN

Tested on separate datasets:
![[Pasted image 20260225131101.png]]
![[Pasted image 20260225131228.png]]

Testing on training dataset:
![[Pasted image 20260225132349.png]]
![[Pasted image 20260225132359.png]]


# February 18
new training losses:
- `percent _error_plus_nmse` 7amlq8af
- `steer_temporal` 9ox59bvo
- `optimal_transport` csxlmnxv
- `jacobian_norm` e68r5zxc
- `l2_weight_norm` jph6m5ep

## Seeding followup

## Stability
steer_temporal
![[Pasted image 20260218132155.png]]
percent_error_plus_nmse![[accel_field_with_orbits_7amlq8af.png]]

![[jacobian_orbit_eig_hist_7amlq8af.png]]

![[jacobian_orbit_xy_heatmap_7amlq8af.png]]

![[solver_dt_vs_jac_7amlq8af.png]]

optimal_transport
![[accel_field_with_orbits_csxlmnxv.png]]

![[jacobian_orbit_eig_hist_csxlmnxv.png]]

![[jacobian_orbit_xy_heatmap_csxlmnxv.png]]

![[solver_dt_orbit_grid_csxlmnxv.png]]

![[solver_dt_vs_jac_csxlmnxv.png]]
jacobian_norm
![[accel_field_with_orbits_e68r5zxc.png]]

![[jacobian_orbit_eig_hist_e68r5zxc.png]]

![[jacobian_orbit_xy_heatmap_e68r5zxc.png]]

![[solver_dt_orbit_grid_e68r5zxc.png]]

![[solver_dt_vs_jac_e68r5zxc.png]]
l2_weight_norm
![[accel_field_with_orbits_jph6m5ep.png|500]]

![[jacobian_orbit_eig_hist_jph6m5ep.png|750]]

![[jacobian_orbit_xy_heatmap_jph6m5ep.png|1000]]

![[solver_dt_orbit_grid_jph6m5ep.png|750]]

![[solver_dt_vs_jac_jph6m5ep.png|750]]


run id lr8kkoly 
Prescriptive config, segmentation:
``` 
        "length_strategy": [
            [0.0, 0.1],
            [0.0, 0.25],
            [0.00, 0.5],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        "segment_length_strategy": [40],
        "steps_strategy": [500, 500, 500, 500, 1000],  # None,
        "lr_strategy": [1e-3, 1e-3, 1e-3, 1e-3, 1e-4],
        
        kl = 1e-3
```

![[media_images_reconstruction_2999_e3139021f37575ad467e.png|500]]

Same as above but not segmentation:
![[Pasted image 20260218124017.png|500]]

![[W&B Chart 2_18_2026, 12_41_37 PM.png|500]]

![[W&B Chart 2_18_2026, 12_43_25 PM.png|500]]

![[W&B Chart 2_18_2026, 12_43_41 PM.png|500]]


# February 11
Look at baseline for reject steps - we should also compare tolerances 
work on latent linear ODE - identify chaos and penalize things in terms of jacobian
pivot to latent, regularized forms of neural ODEs

Run id jrq509xt
Segment length 18. no length strat:
``` config
wandb:
  group: "segment-18-2-11"  # Change this to your desired group name

data:
  # dataset_name : ["complex_TBP_planar_1_train", "complex_TBP_planar_10_train", "complex_TBP_planar_100_train"]
  dataset_name : ["complex_TBP_planar_10_train"]
  # dataset_name : ["simple_TBP_planar_1_train","complex_TBP_planar_1_train","simple_TBP_planar_10_train","complex_TBP_planar_10_train","simple_TBP_planar_100_train","complex_TBP_planar_100_train"]
  problem: '2BP'

parameters:
  # solver_stats_jacobian_debug: true
  # solver_stats_frequency: 100   # log every N global steps
  # solver_stats_samples: 5       
  # solver_stats_jacobian: true
  # solver_stats_jacobian_max_dim: 32
  # # solver_stats_use_all_samples: true
  # solver_stats_jacobian_eps: 1.0
  # solver_stats_jacobian_percentiles: [5.0, 95.0]
  # EXHAUSTIVE LENGTH STRATEGY
  length_strategy:    [
                        # [0.0, 0.1],
                        # [0.0,1.0],[0.0,1.0],],
                        [[0.0,1.0],]
                      ]

  lr_strategy: [[0.001, 0.0001]]
  steps_strategy: [[500,500]]
  segment_length_strategy: [[18,]]


  width: [64]
  depth: [2]
  train_val_split: 1
  batch_size: [256]
  num_trajs: -1
  seed: [2345]
  # loss_fcn: "mean_squared_error"
  loss_fcn: percent_error_plus_nmse
  # loss_fcn: multi_step_rollout_percent_error_plus_nmse
  # rollout_steps: 8        # optional, default = full segment
  # rollout_stride: 1       # optional, default = 1

  activation: [leaky_relu]
  # activation: leaky_relu
  # activation: elu

  # feature_layer: [sph_4D_rinv_vel_loga_energy]
  # feature_layer: [sph_4D_rinv_vel_logra]
  feature_layer: [sph_4D_rinv_vel]
  # output_layer: [mlp_4D_signed, mlp_4D_unit, mlp_4D_unit_softplus, mlp_4D_logmag_unit_exp]
  output_layer: [mlp_4D_signed]
  planar_constraint: true
  

  rtol: 0.000001
  atol: 0.00000001
```

![[Pasted image 20260211123237.png]]
![[Pasted image 20260211123308.png]]
![[Pasted image 20260211130433.png]]
![[Pasted image 20260211130409.png]]

![[Pasted image 20260211125425.png]]
![[Pasted image 20260211130444.png]]

--- 


With length strat:
7bwn7zzx
```
  length_strategy:    [
                        # [0.0, 0.1],
                        # [0.0,1.0],[0.0,1.0],],
                        [[0.0,0.1],[0.0,0.5],[0.0,1.0],]
                      ]

  lr_strategy: [[0.001, 0.001, 0.0001]]
  steps_strategy: [[200,200, 500 ]]
```
![[Pasted image 20260211130545.png]]
![[Pasted image 20260211130525.png]]
![[Pasted image 20260211130028.png]]
![[Pasted image 20260211130008.png]]
![[Pasted image 20260211130056.png]]
![[Pasted image 20260211130134.png]]
 
 ---

  

No segmentation, using length strat:
run id aa9qd97n
```
config
wandb:
  group: "segment-18-w-length-2-11"  # Change this to your desired group name

data:
  # dataset_name : ["complex_TBP_planar_1_train", "complex_TBP_planar_10_train", "complex_TBP_planar_100_train"]
  dataset_name : ["complex_TBP_planar_10_train"]
  # dataset_name : ["simple_TBP_planar_1_train","complex_TBP_planar_1_train","simple_TBP_planar_10_train","complex_TBP_planar_10_train","simple_TBP_planar_100_train","complex_TBP_planar_100_train"]
  problem: '2BP'

parameters:
  solver_stats_jacobian_debug: true
  solver_stats_frequency: 100   # log every N global steps
  # solver_stats_samples: 5       
  solver_stats_jacobian: true
  solver_stats_jacobian_max_dim: 32
  solver_stats_use_all_samples: true
  # solver_stats_jacobian_eps: 1.0
  # solver_stats_jacobian_percentiles: [5.0, 95.0]
  # EXHAUSTIVE LENGTH STRATEGY
  length_strategy:    [
                        # [0.0, 0.1],
                        # [0.0,1.0],[0.0,1.0],],
                        [[0.0,0.1],[0.0,0.5],[0.0,1.0],]
                      ]

  lr_strategy: [[0.001, 0.001, 0.0001]]
  steps_strategy: [[200,200, 500 ]]
  segment_length_strategy: [[360,]]


  width: [64]
  depth: [2]
  train_val_split: 1
  batch_size: [256]
  num_trajs: -1
  # seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,]
  # seed: [0,1,2,3,4]
  seed: [2345]
  # loss_fcn: "mean_squared_error"
  loss_fcn: percent_error_plus_nmse
  # loss_fcn: multi_step_rollout_percent_error_plus_nmse
  # rollout_steps: 8        # optional, default = full segment
  # rollout_stride: 1       # optional, default = 1

  activation: [leaky_relu]
  # activation: leaky_relu
  # activation: elu

  # feature_layer: [sph_4D_rinv_vel_loga_energy]
  # feature_layer: [sph_4D_rinv_vel_logra]
  feature_layer: [sph_4D_rinv_vel]
  # output_layer: [mlp_4D_signed, mlp_4D_unit, mlp_4D_unit_softplus, mlp_4D_logmag_unit_exp]
  output_layer: [mlp_4D_signed]
  planar_constraint: true
  

  rtol: 0.000001
  atol: 0.00000001
```
![[Pasted image 20260211125026.png]]
![[Pasted image 20260211125046.png]]

---

segment length 4, no length strat
run id: 86ab7651

 --- 
 

# February 4
Program with Simon:
- astrodynamics package - all relevant classes of orbits. Lyapunov, NRHO, resonant, maybe QPO and manifold. Don't need to write it from scratch - look at open source materials and make sure we have access to orbits we're interested in. Corresponding dynamics. Visualization. Simon will probably have orbit theory, I have more ML. Corresponding training pipelines. 
- Include undergrad for 30 min of program meeting


Justification for the $\delta \dot{x} = A(t)\delta X$ if lambda is >= 0 maybe our Jacobian is larger and pushes the loss, and 
tensorboard you can look at distirbution of gradient updates across batches 
study over batch sizes - how do the gradients change. do we have tighter distributions with more orbtis
Look for more uses for neural ODEs. Look at related articles - journal articles top tier universities. Berkley, stanford MIT and Europe. ESA.
Look for repositories 
Referenced the paper "The Vanishing Gradient Problem for Stiff Neural Differential Equations". Neural ODEs can become stiff during training, even if true data dynamics are non-stiff, which can cause vanishing gradients. If the network learns sharp, highly curved vector fields then Jacobian eigenvalues blow up, the system becomes stiff, solver takes tiny steps, and gradients collapse.

For neural ODEs
$\frac{d \lambda}{dt} = - \left( \frac{\partial f_\theta}{\partial x} \right)^T \lambda$
If the Jacobian has large negative eigenvalues (stiff decay modes), $\lambda(t) \sim e^{-\alpha t}$ so gradients decay exponentially backward in time. 

Adaptive solvers will reject unstable steps and prevent divergence in early training. They can be dangerous if they mask model errors. If the NN learns a stiff, wildly curved vector field then the solver will take tiny steps, reject steps, and compensate numerically. When the solver shrinks its steps to handle stiffness, the adjoint becomes ill-conditioned, gradients vanish, and long-horizon behavior plateaus. 

This supports the finding that increasing depth does not necessarily improve results: "When the neural network is deep and the dynamics are stiff, gradients may vanish due to both repeated nonlinear transformations and the numerical damping imposed by the integrator’s stability function, making some parameters effectively untrainable." (from aforementioned paper). And from "Improving Neural ODE Training with Temporal Adaptive Batch Normalization": "Previous studies on Neural ODEs parameterize the learnable temporal derivatives using a shallow neural network with a limited number of parameters [7, 30]. Without special treatment, merely stacking additional layers in the temporal derivatives does not necessarily enhance Neural ODE performance. Furthermore, deeper networks might increase the stiffness of the ODE system, leading to challenges with the ODE solver, such as excessively small step sizes or even failures due to infinite state values, as shown in Figure 1". Depth of the MLP controls nonlinearity of the vector field, so increasing depth too much results in sharper, more twisted vector fields. Increasing width can help expressivity and optimization more reliably than increasing depth, but can still result in stiffer learned vector fields. 

Width also does not always improve results. Width results in smoother Jacobians, fewer extreme eigenvalues, and adjoint gradients decay slower. However, if we increase width too much, stiffness again increases and solver steps decrease and training becomes unstable since it can represent sharp vector fields. We can combat this through solver-step penalties, Lipschitz regularization, spectral norm constraints, or Jacobian regularization. The Jacobians in question:
1. State Jacobian of the learned vector field: 
	$J_f(x) \;=\; \frac{\partial f_\theta(x)}{\partial x}$
	where 
	$\dot{x}(t) = f_\theta(x(t))$.
	This Jacobian controls local stability, stiffness, and gradient flow through time. The sensitivity of the trajectory to perturbations evolves as $\frac{d}{dt} \delta x(t) = J_f(x(t)) \, \delta x(t)$, so the spectrum of $J_f$ controls:
	- stiffness (large negative eigenvalues)
	- chaotic behavior (large positive eigenvalues)
	- vanishing gradients (contractive flows)
	- exploding gradients (expansive flows)
	- solver step size (adaptive solvers shrink dt if eigenvalues are large)
	Deeper MLPs induce Jacobians with larger operator norm and worse conditioning. Wider but shallow networks tned to give smoother $f_\theta(x)$, smaller, better-conditioned $J_f(x)$, less stiffness, more stable adjoints. 
	Each eigenvalue $\lambda_i$ corresponds to a local exponential rate in some direction of state space:
	- $\Re(\lambda_i) < 0$: contraction (stable / decaying mode)
    - $\Re(\lambda_i) > 0$: expansion (unstable / diverging mode)
	- $\Re(\lambda_i) \approx 0$: neutral direction
2. Jacobian of the flow map, e.g. the trajectory snesitivity/STM: $\partial \Phi_t / \partial x_0$.
	$J_\Phi(t) = \frac{\partial x(t)}{\partial x_0}$
	This is what explodes/vanishes in the adjoint method. It evolves via $\frac{d}{dt} J_\Phi(t) = J_f(x(t)) \, J_\Phi(t)$.
	Bad $J_f$ = bad $J_\Phi$. Stiff vector field means ill-conditioned STM, and vanishing gradients in time. Vanishing gradients are usually due to $J_\Phi(t)$ collapsing due to large negative eigenvalues of $J_f$. 
3. Jacobian wrt parameters $\partial f_\theta / \partial \theta$.
	This affects optimization conditioning, how noisy gradients are, how easy it is for SGD to move through the vector field. Solver pathologies are driven by the state Jacobian, not this one. 
We record the state Jacobian of the learned vector field. We the max absolute real part of the eigenvalues because large negative real parts indicate stiff, fast-decaying modes (solver shrinks dt) and large positive real parts indicate unstable, exploding modes. Pure imaginary parts (like in the spectral radius) just indicate oscillations, which does not give us info about stiffness just oscillatory behavior (which we have in 2BP anyway). We also record the spread ratio = p95/p05. We have eigenvalues of the state Jacobian, take their real parts and absolute values so we have a vector of nonnegative scalars. p05 is the 5th percentile of that vector, e.g. it is the value such that 5% of the directions in state space have local exponential rates at or below this number and 95% are at or above it. This means 5% of eigenmodes are slower than or equal to p05. So p05 is the slow mode, and p95 is the fast mode. The spread_ratio can tell us about conditioning. 

I wanted to check to make sure the neural ODE itself is not becoming stiff. Here is a an example of training on complex_TBP_planar_10_train. Run id 3frpgcw1
Replot without trajectories 
![[Pasted image 20260204130243.png]]
![[Pasted image 20260204130139.png]]

![[Pasted image 20260204123704.png]]


--- 
# January 28
Inspect why trends are behaving "weirdly"
Consistent behavior ndicates "good model" - hanspeterschaub.info/Papers/grads/JohnMartin.pdf

We are not particularly sensitive to architecture
If architecture isn't producing the treds that are expected, 
Run different architectures across 4 or 5 seeds, reuse visualizations and scripts to perform that study. Want to find a quanitatively informed best model architecture of the ones we've tried. If configs are not "doing the job", 1. maybe neuralODEs are just underpowered during 

maybe churn some latent odes in the background and perform these studies
read about other use cases for neuralODE - are there any other comprehensive studies that use them or are we alone in this
repeat with different input/output layers with different sees. try this with latent odes as well. maybe start with model size like we did for neuralODEs. 
# January 21
# Learning perturbations
---
try putting neural ODE around an asteroid with other little masses around. 

tangential - 


put vis in dictionary and pickle it for later visualization
## Step 1 - minimum working example
Forced harmonic oscillator:
 $\ddot{x}(t) \;+\; \omega^{2} x(t) \;=\; \epsilon\,\bigl(x(t) - x_{\text{shift}}\bigr)^{3}$
get sensitivity studies done 
- seeding test is top priority 
- larger network may be better
- should be 30 min experiments
Have a report for sensitivity study by next week

Training data:
![[Pasted image 20260121120448.png]]

## Vanilla neural ODE
Training procedure:
```
initialize neural network parameters θ
initialize optimizer (e.g. Adam)

for each training step:

    # 1. Enforce known physics
    define ODE RHS:
        ẋ = v
        v̇ = −ω² x  +  fθ(x, v, t)

    # 2. Forward solve with enforced physics
    integrate the ODE from initial state y[0]
    using the fixed known term −ω² x
    and the learned residual fθ

    # 3. Trajectory-level supervision
    compare predicted trajectory ŷ(t[i])
    against observed data y(t[i])

    # 4. Loss construction
    data_loss = trajectory error
    regularization = ||fθ||²
    total_loss = data_loss + λ · regularization

    # 5. Update residual only
    backpropagate total_loss
    update θ (parameters of fθ only)
```  

Loss:
![[Pasted image 20260121120833.png]]

Results:

![[Pasted image 20260121115232.png]]

## Latent neural ODE
![[Pasted image 20260121125045.png]]
![[Pasted image 20260121125029.png]]
## Drag
Using simplified drag model from Vallado. At a reference height of 300 km, area=10 m^2 and mass = 1000 kg for one revolution:
![[Pasted image 20260121123412.png]]
For 100:
![[Pasted image 20260121123500.png]]
Training on 1 orbit:
![[Pasted image 20260121124449.png]]

SRP:
![[Pasted image 20260121132014.png]]
# January 9
To do before meeting:
- [ ] Paper draft
- [ ] Clean code
- [ ] Plan visualizations and create

Training on nonplanar dataset, with length strategy: ![[Pasted image 20260109060514.png]]

Without length strat:
![[Pasted image 20260109062448.png]]

## Storyboard
Extending conference results. We were originally planning to include latent ODEs. We want to show that we can force linearization in latent space 

Increase network size and perform more rigorous experiments on seeding results. 
Can we learn dynamics that are not obvious?
- 4BP - if we have CRTBP, can we compensate for those pieces. Use latent ODEs- how more powerful are these
- Can latent ODEs learn time varying forces?

Neural ODEs, latent ODEs learn underlying dynamics. SRP, drag, 4 body effects affect our motion. 
If you're observing a trajectory, can we learn the dynamics models?
Sensitivity study to get an idea of what errors we should actually be expecting. Where is dangerous and helpful.
Study 3 forces, see how well we can recover these. 
May need to write and validate 
What if we only observe part of the trajectory? Dark side fo the moon. What if we have state estimates? What if we want this to become more.
NRHO would be cool

loop timer- go through activities
# December 2

Submit paper top of the year - 6 months until review
Expect John to be asking for plots soonish for program applciations
No group meetings during break
No official 1:1s during break -  restart 1/20. He's still available. We can set a time and waive if not necesary
for Zaratan:
	test different core counts, memory configs, gpu if using. 
# November 25
Dataset:
![[Pasted image 20251125121057.png]]
and orbit 2 is our validation orbit
![[Pasted image 20251111133754.png]]
Questions for John:
- Is putting an activation directly in the output layer "normal"?
Recap:
- We discovered that our model was having toruble training because our output layer introduced an unexpected failure mode. The model was predicting acceleration (signed) and its unit vector (also signed), but only cares about the _product_ of these quantities, e.g. it only "sees" the final acceleration vector which gets integrated to compute loss. Because of this, we were seeing randomly flip flopping acceleration direction vector
``` python
def mlp_4D_signed(mlp_output, state, scalar=1.0):
    r_mag = mlp_output[0:1] # This is signed!
    r_dir = mlp_output[1:4]
    acc_pred = r_mag * r_dir
    return jnp.concatenate((state[3:6], acc_pred), axis=0)
```
- Example of direction randomly flip flopping:![[Pasted image 20251118100654.png]]
I fixed this by enforcing that acceleration and position are always opposite each other. If $\theta$ is the angle from the acceleration to the position vector, then $\theta = 180^\circ$ must always be true for the force to be attractive. To enforce this, I check dot product of the position and acceleration vectors and penalize positive values (this assumes that they align). However, I didn't actually check that the force is attractive. Because of this, I could get results where acceleration and position were in the opposite directions of what they should be:
![[Pasted image 20251118115254.png]]
To fix this, we can just enforce that the acceleration magnitude is truly a magnitude, i.e. it should always be positive and the model just learns its direction.  We updated the model to use the following output:
``` python
def mlp_4D(mlp_output, state, scalar=1.0):
    r_mag = jnp.abs(mlp_output[0:1])
    r_dir = mlp_output[1:4]
    acc_pred = r_mag * r_dir
    return jnp.concatenate((state[3:6], acc_pred), axis=0)```
I stripped back the model (returned to leaky relu, percent error). I will refer to this as the baseline for the day

## v0 (baseline)
run_id: nx9r8tau
runtime: 1:34 
config:
``` python
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

  loss_fcn: "percent_error"

  activation: leaky_relu

  feature_layer: sph_4D_rinv_vel
  output_layer: mlp_4D
  planar_constraint: true

  rtol: 0.000001
  atol: 0.00000001
  ```
![[Pasted image 20251125103904.png]]
![[Pasted image 20251125103920.png]]
![[Pasted image 20251125103943.png]]
![[Pasted image 20251125111642.png]]

![[Pasted image 20251125104007.png]]

Notes:
- direction is fixed!
- acceleration is turning radial near periapsis
- we could enforce that acceleration is radial but it would be nice to keep the loss fairly generalized. to see if we can get a bit closer, let's try a loss which is a combination of percent error and rmse to enforce

## v1 - RMSE + percent error loss
Differences: 
- change loss function from percent error to RMSE + percent error. RMSE is scaled so that it is a similar scale to the percent error component of loss
- decrease to 2000 steps
run_id: vhrfc9z7
runtime: 1:17
![[Pasted image 20251125105938.png]]
![[Pasted image 20251125110105.png]]
![[Pasted image 20251125110040.png]]
![[Pasted image 20251125111415.png]]
![[Pasted image 20251125105950.png]]
Results are comparable. I'll return to percent error since it's lighter weight, but keep this in mind as back up.
Our network is fairly shallow. I want to experiment with increasing expressiveness, so let's 1) switch from 32x2 model to 16x4 or 32x4, then 2) add a final activation. I'm also wondering if the segments could be too short? Let's decrease from 90 to 36 segments

## v2 - update architecture from 32x2 to 16x4
Differences: 
- return to percent error loss
- update from width 32 depth 2 to width 16 depth 4.
run_id: wy4pijq6
runtime: 1:11
![[Pasted image 20251125115339.png]]

![[Pasted image 20251125115404.png]]

Not looking as good as 32x2. Let's try 32x4 for fun

## v3 - 32x4
run_id: 5qetg6ll
runtime: 1:12
config:
``` python
  length_strategy:
                      [[
                        [0.0, 1.0],
                      ]]
  lr_strategy: [[0.001]]
  steps_strategy: [[2000]]
  segment_length_strategy: [[4,]]

  width: 32
  depth: 4
  train_val_split: 0.8
  batch_size: 32
  num_trajs: -1

  # loss_fcn: "mean_squared_error"
  # loss_fcn: "percent_error_with_attraction"
  loss_fcn: "percent_error"
  # loss_fcn: "percent_error_plus_nmse"

  # activation: tanh
  activation: leaky_relu
  # activation: elu

  feature_layer: sph_4D_rinv_vel
  # feature_layer: sph_4D_rinv_vinv
  # output_layer: mlp_4D_activation
  # output_layer: mlp_4D_unit_scaled
  # output_layer: mlp_simple_hybrid
  output_layer: mlp_4D
  planar_constraint: true

  rtol: 0.000001
  atol: 0.00000001
  ```

![[Pasted image 20251125115931.png]]
![[Pasted image 20251125115948.png]]
About comparable with 32x2. 
![[Pasted image 20251125120811.png]]
![[Pasted image 20251125120803.png]]
![[Pasted image 20251125120750.png]]
## v4 - 64x2
run_id: zjiht8pb
runtime: 1:10
config:
``` python
parameters:

  length_strategy:
                      [[
                        [0.0, 1.0],
                      ]]
  lr_strategy: [[0.001]]
  steps_strategy: [[2000]]
  segment_length_strategy: [[4,]]

  width: 64
  depth: 2
  train_val_split: 0.8
  batch_size: 32
  num_trajs: -1

  # loss_fcn: "mean_squared_error"
  # loss_fcn: "percent_error_with_attraction"
  loss_fcn: "percent_error"
  # loss_fcn: "percent_error_plus_nmse"

  # activation: tanh
  activation: leaky_relu
  # activation: elu

  feature_layer: sph_4D_rinv_vel
  # feature_layer: sph_4D_rinv_vinv
  # output_layer: mlp_4D_activation
  # output_layer: mlp_4D_unit_scaled
  # output_layer: mlp_simple_hybrid
  output_layer: mlp_4D
  planar_constraint: true

  rtol: 0.000001
  atol: 0.00000001```

![[Pasted image 20251125121721.png]]
![[Pasted image 20251125121713.png]]

![[Pasted image 20251125120852.png]]
This looks great! We're not getting the radial acceleration that was apparent in narrower networks. Behavior looks good across most of the orbits:
![[Pasted image 20251125121354.png]]![[Pasted image 20251125121405.png]]
but notably had a tough time had trouble when the orbit IC is near periapsis:
![[Pasted image 20251125121556.png]]

Outputs for healthy orbit 0:
![[Pasted image 20251125121751.png]]
vs orbit where we start near periapsis:
![[Pasted image 20251125121824.png]]
# v5 - final activation
Difference: apply softplus to acc mag and tanh to direction (I don't take abs of r_mag because we use softplus)
``` python
def mlp_4D_activation(mlp_output, state, scalar=1.0):
    r_mag = mlp_output[0:1]
    r_dir = mlp_output[1:4]
    r_mag_activated = jnn.softplus(r_mag)
    r_dir_activated = jnn.tanh(r_dir)
    acc_pred = r_mag_activated * r_dir_activated
    return jnp.concatenate((state[3:6], acc_pred), axis=0)
```
run_id: ke3mcj2t
runtime: 1:13
config:
``` python
parameters:

  length_strategy:
                      [[
                        [0.0, 1.0],
                      ]]
  lr_strategy: [[0.001]]
  steps_strategy: [[2000]]
  segment_length_strategy: [[4,]]

  width: 64
  depth: 2
  train_val_split: 0.8
  batch_size: 32
  num_trajs: -1

  # loss_fcn: "mean_squared_error"
  # loss_fcn: "percent_error_with_attraction"
  loss_fcn: "percent_error"
  # loss_fcn: "percent_error_plus_nmse"

  # activation: tanh
  activation: leaky_relu
  # activation: elu

  feature_layer: sph_4D_rinv_vel
  # feature_layer: sph_4D_rinv_vinv
  output_layer: mlp_4D_activation
  # output_layer: mlp_4D_unit_scaled
  # output_layer: mlp_simple_hybrid
  # output_layer: mlp_4D
  planar_constraint: true

  rtol: 0.000001
  atol: 0.00000001
```
![[Pasted image 20251125122417.png]]
![[Pasted image 20251125122426.png]]
![[Pasted image 20251125124843.png]]
![[Pasted image 20251125124851.png]]
![[Pasted image 20251125124859.png]]



## v6 - final activation w/ sigmoid instead of softplus
run_id: 4actbiif
runtime: 1:08
![[Pasted image 20251125123122.png]]
![[Pasted image 20251125123131.png]]
![[Pasted image 20251125124550.png]]
![[Pasted image 20251125123206.png]]
![[Pasted image 20251125123212.png]]
## v7 - final activation, loss is position + rmse
Differences: wjith final activaiton, go back to percent error + rmse loss
run_id: ftx7brwd
runtime: 1:17
config:
``` python
parameters:

  length_strategy:
                      [[
                        [0.0, 1.0],
                      ]]
  lr_strategy: [[0.001]]
  steps_strategy: [[2000]]
  segment_length_strategy: [[4,]]

  width: 64
  depth: 2
  train_val_split: 0.8
  batch_size: 32
  num_trajs: -1

  # loss_fcn: "mean_squared_error"
  # loss_fcn: "percent_error_with_attraction"
  # loss_fcn: "percent_error"
  loss_fcn: "percent_error_plus_nmse"

  # activation: tanh
  activation: leaky_relu
  # activation: elu

  feature_layer: sph_4D_rinv_vel
  # feature_layer: sph_4D_rinv_vinv
  output_layer: mlp_4D_activation
  # output_layer: mlp_4D_unit_scaled
  # output_layer: mlp_simple_hybrid
  # output_layer: mlp_4D
  planar_constraint: true

  rtol: 0.000001
  atol: 0.00000001
  note: final activaiton is sigmoid for acc, tanh for direction vector
  ```
``
![[Pasted image 20251125123636.png]]
![[Pasted image 20251125123845.png]]
![[Pasted image 20251125123856.png]]
Notabl;y,  tracks better than previously in the case that IC is near periapsis: ![[Pasted image 20251125123943.png]]
orbit 0:![[Pasted image 20251125124113.png]]
![[Pasted image 20251125124435.png]]


\begin{table}
\caption{Mean Acceleration Error for Different Datasets}
\label{tab:mean_acc_err}
\begin{tabular}{lccc}
\toprule
\diagbox{Train}{Test} & Single & Simple & Complex \\
\midrule
Single & 14.454852 & 14.194813 & 20.099958 \\
Simple & 15.344963 & 14.996629 & 16.371222 \\
Complex & 2.1322267 & 2.1185184 & 2.0743387 \\
\bottomrule
\end{tabular}
\end{table}

Best results:
Single & 45.34824 & 46.23542 & 38.12336 \\
Simple & 12.4346895 & 12.926439 & 8.13189 \\
Complex & 1.8970726 & 1.9041611 & 1.8519293

# November 18
 next week:
 - direciton fixed
 - redo conference stuff
 - what else needs to be done?
- pending sufficient data, wer could have a segmentaiton where we separate by segments and orbits - segmentation validation vs orbit validation?
- try with both and then decide

John's work enforced $\Lambda ^2 u = 0$
Take gradient with repect to output ($\grad \dot \vec{a}$) (equivalent to \vec{grad}u = -\vec{a})

re:validation data - pick a random seed and find distribution that is closer 
domain of output is way smaller when we choose the bounds - i.e. the directions are bounded vs the cartesian accelerations
John argues that smaller domain is more learnable because we are far better conditioned 
rapid interpretability is also useful
this is based on John's heurisitics and whatnot

- not done with characterization - we need to run sensitivity studies with the fresh setup
	- this should be done within the next week
	- we should have everything we need but in latent ODEs
	- John is optimistic that we can do this by the end of the month
	- gives me December for latent ODEs 
- end this month with very good to have all the differential correction stuff done
- if we have all pieces in place by January, that is good
- get scripts ready for latent odes
- studies shouldn't take more than one night to complete - should be 6 hoursish
- john isn't convinced that dot product. pass features through sigmoid for the magnitude (0 to 1), and then tanh for the rest
- minimize number of constraints included for the baseline
- could include a section for other knobs to turn
- first try enforcing magnitude 
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
[[![[Pasted image 20251118115228.png]]]]
![[Pasted image 20251118115234.png]]
![[Pasted image 20251118115254.png]]

- output features are not tracking, but we are seeing smooth behavior now
- Let's try increasing depth to increase expressiveness

# v4 - increasing depth
run_id: v0cve1s8
time to run: 1:51
config:
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
  depth: 4
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
![[Pasted image 20251118115653.png]]
![[Pasted image 20251118115706.png]]
![[Pasted image 20251118115716.png]]
![[Pasted image 20251118115729.png]]
![[Pasted image 20251118115734.png]]
![[Pasted image 20251118115840.png]]
"width tends to give immediate gains, lwhile depth is powerful but riskier in neural ODEs due to solver stability" what does this mean exactly? seems to be true - Wider layers give each time step more expressive power without changing the ODE’s depth, so you usually see quick improvements and training stays stable. Adding depth stacks more transformations into the dynamics function, which can represent richer behavior, but it makes the differential equation stiffer/harder to solve—optimizers and solvers have to cope with deeper Jacobians, which can cause instability or require tiny step sizes. So width is the safer knob for immediate gains, while depth can help but carries a higher risk of solver issues.

What if we add a final activation to increase expressiveness?

## adding final activation - jnn.tanh
run_id: 4uronz2w
time to run: 1:49
config is same as before, `final_activation` was implemented into `self.mlp` in `Func` class in `neuralODE`. For now, we are starting with tanh since it is smooth, bounded, and symmetric. 

![[Pasted image 20251118120711.png]]
![[Pasted image 20251118120719.png]]
![[Pasted image 20251118120747.png]]
![[Pasted image 20251118120734.png]]
![[Pasted image 20251118120758.png]]

- This wasn't fruitful
- Decreasing rtol and atol may help - Lowering rtol/atol makes the ODE solver track the dynamics more tightly, so it can capture sharp spikes (like periapsis) more faithfully during both training solves and feature captures. That often helps, because the model sees cleaner gradients and the solver doesn’t smooth over the rapid acceleration change. The trade-off is higher compute cost and potential stiffness: tighter tolerances mean smaller step sizes, so training slows down and might become unstable if the model still can’t represent the spike. I’d consider reducing tolerances moderately once the model capacity/output head is in good shape; pair it with monitoring solver statistics to ensure the integration remains stable.

# decreasing rtol and atol
run_id: amnrug9d
run time: 2:30
config:
``` python
parameters:

  length_strategy:
                      [[
                        [0.0, 1.0],
                      ]]
  lr_strategy: [[0.001]]
  steps_strategy: [[3000]]
  segment_length_strategy: [[4,]]

  width: 32
  depth: 4
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

  rtol: 0.00000001
  atol: 0.0000000001
```
This also still has final_activation in it
![[Pasted image 20251118121531.png]]
![[Pasted image 20251118121556.png]]
![[Pasted image 20251118121612.png]]
![[Pasted image 20251118121707.png]]
![[Pasted image 20251118121714.png]]
![[Pasted image 20251118121721.png]]
- This didn't have a significant difference (which is beneficial to some extent, train time is longer with lower tolerances). We will revert to previous tolerance and remove tanh
- just in case, let's try removing final activation but keep these tight tolerances (ETA - this is run n4tqvi4l and result basically mirrored those above. not shown for lack of relevance)
- things we still want to consider - the points where we get 0 acceleration are because ||acc_dir||~= 0. we should enforce a unit vector or test different output layers. Before we try this, let's decrease depth and increase width

## decrease depth, increase width
run_id: 48r3r35l
runtime: 1:54
config:
``` python
parameters:

  length_strategy:
                      [[
                        [0.0, 1.0],
                      ]]
  lr_strategy: [[0.001]]
  steps_strategy: [[3000]]
  segment_length_strategy: [[4,]]

  width: 64
  depth: 3
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
![[Pasted image 20251118123212.png]]
![[Pasted image 20251118123222.png]]
![[Pasted image 20251118123240.png]]
![[Pasted image 20251118123249.png]]
![[Pasted image 20251118123306.png]]

## returning to previous model
Conclusions so far:
- we need to enforce ||dir_acc|| = 1 to avoid acceleration dips previously seen
- There is a balance of model architecture needed to keep behavior smooth
- one model looks correct but the acceleration magnitude and directions are mirrored?
look at run_id = z0v2rmhl again:

![[Pasted image 20251118115157.png]]
![[Pasted image 20251118115124.png]]
![[Pasted image 20251118115208.png]]
![[Pasted image 20251118115228.png]]
![[Pasted image 20251118115234.png]]
![[Pasted image 20251118115254.png]]

![[Pasted image 20251118131059.png]]

we need to enforce that the force is always attractive
let's do this

## adding attraction component to loss:

![[Pasted image 20251118134317.png]]
![[Pasted image 20251118134324.png]]
![[Pasted image 20251118134332.png]]
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
