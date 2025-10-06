# September 24
## Recap
Yesterday in my 1;1, John and I agreed that it would make sense to not focus too. much on the education portion of my planning for the future phase. I haven't done much development in a while, so I want to get back to that.

I will build out capabilities for differential correction as needed. I won't focus on building out an extensive repository before I have a need for it. Rather, I'll 
# September 23
## Recap
Last week, I gave a literature review about periodic orbit discovery. I didn't get to multiple shooting methods or collocation, but will need to implement these in my own work. I want to go back and add these methods to the colab I shared out. I have a few other updates to make that I noted within the colab to work through.

Ultimately, I want to have a dynamics repo hopefully eventually extending to PO and QPO discovery in CR3BP, 3BP, 4BP (maybe include other trajectories?). I have a general outline of what I want this to look like, but John and I are going to discuss this further today.

As I continue to work on this, I also need to finalize my schedule and ultimate goal for the semester. We have an idea of where to go but not how to do it.

As a jumping off point, John recommended "Spacecraft Relative Motion Dynamics and Control Using Fundamental Modal Solution Constants" and "Geometric perspectives on fundamental solutions in the linearized satellite relative motion problem" from HP Schaub's group. I started working through those yesterday as well as some of my remaining Shane Ross videos. I want to review these in more depth today to be able to identify if/where they will be useful for us.

## Goals
- [x]  Second pass through papers by HP Schaub
	- [ ] Be able to explain them and identify if/where they will relate to our problem
- [x] Solidify goal for the semester in words
	- [x] I have notes on my plan, but I should summarize them into a more concrete problem statement and identify areas that will require the mose research/effort
	- [ ] Similarly, we started a semester schedule but I still need to fill in some details and dates
- [ ] Populate colab notebook with multiple shooting theory
- [ ] Populate colab notebook with collocation theory
- [ ] implement multiple shooting
- [ ] implement collocation
- [ ] transfer code from colab notebook to a more modular format in a manner that would be used for future work

## Problem Statement
We aim to develop a neural ODE framework for discovering periodic orbits in increasingly complex dynamical systems. This amounts to solving challenging boundary value problems in highly nonlinear settings.

Traditionally, periodic orbits are identified using numerical methods such as differential correction. These techniques often fail in strongly nonlinear regimes or when long integration times result in large accumulated errors.

To mitigate these limitations, our approach will use neural ODEs to parameterize the dynamics in a latent space, where they can be encouraged towards smooth and near-linear forms through training. This representation is more amenable to tools from numerical dynamical systems theory. We hypothesize that this approach will enable the discovery of periodic orbits that are otherwise inaccessible as the system of interest grows in complexity and nonlinearity. The resulting framework should increase the probability of converging to solutions of the boundary value problems that underlie periodic orbit discovery, even in complex nonlinear systems.

## Geometric perspectives on fundamental solutions in the linearized satellite motion problem

![[Pasted image 20250923131108.png]]
Modal decomposition for linear systems usually means diagonalizing a constant matrix. If you have a linear time-invariant (LTI) system
$$

\dot{x} = A x,

$$
you can diagonalize $A = V \Lambda V^{-1}$, and the solutions are combinations of exponential modes
$$

x(t) = \sum_i c_i v_i e^{\lambda_i t}.

$$
That is the standard modal decomposition.

For relative motion problems, only the Clohessy–Wiltshire (CW) case is LTI. Most other formulations (eccentric reference orbits, CR3BP, etc.) are linear time-varying (LTV):
$$

\dot{x} = A(t)x, \quad A(t+T) = A(t).

$$
Here the coefficients are periodic in time, and you cannot simply diagonalize one constant matrix.

Lyapunov–Floquet theory says there exists a periodic change of variables
$$

x(t) = P(t) z(t), \quad P(t+T)=P(t),

$$
such that the new coordinates satisfy

$$

\dot{z} = \Lambda z,

$$
with constant $\Lambda$. This shows any LTV system with periodic coefficients is equivalent to an LTI system.

  
Section IIA explains this in more detail:
Equation (2) says $x = P(t) z$, $z = P(t)^{-1}x$. 
Equation (3) shows that in the new coordinates the dynamics reduce to $\dot{z} = \Lambda z$. 
Equation (4) is the compatibility condition
$$

P(t)^{-1} A(t) P(t) - P(t)^{-1}\dot{P}(t) = \Lambda.

$$
Equation (5) constructs $P(t)$ using the state transition matrix $\Phi(t;t_0)$:
$$

P(t) = \Phi(t;t_0) e^{-\Lambda(t-t_0)}.

$$

Equation (6) enforces that $P(t)$ is the identity at the epoch time. Equation (7) shows that
$$

\Lambda = \tfrac{1}{T}\ln \Phi(t_0+T;t_0),

$$
which connects the constant matrix $\Lambda$ to the monodromy matrix.
Once you have $\Lambda$, and assuming it is diagonalizable, the solution takes the form

$$

x(t) = \sum_i c_i P(t) v_i e^{\lambda_i t},

$$

where the $v_i$ are eigenvectors and the $\lambda_i$ are eigenvalues of $\Lambda$. The exponential terms govern growth, decay, or oscillation, and the periodic modulation $P(t)$ shapes the time dependence.

For CR3BP or eccentric relative motion, the system is LTV with periodic coefficients. Lyapunov–Floquet theory lets you still write solutions as modal decompositions, but with periodic modulation. The eigenvalues $\lambda_i$ are obtained from $\Lambda = \frac{1}{T}\ln M$, with $M$ the monodromy matrix. Stability is determined by these eigenvalues, or equivalently by the multipliers of the monodromy matrix. The periodic modulation $P(t)$ does not affect stability but is needed if you want the actual time-domain solution.

In summary, modal decomposition is straightforward in the CW problem because the system is LTI. In more general periodic systems, Lyapunov–Floquet theory extends the idea: solutions look like exponential modes, but with periodic modulation. Stability comes from the exponential part, which is computed from the monodromy matrix.

 --- 
# September 17
## Recap
I am preparing for my literature review about numerical techniques to discover periodic orbits in the CR3BP. I originally began my introduction of shooting methods with a simple example akin to the classic cannon problem, but as I've gone on I don't think that it's worth including this with the time that I have.

I collected many resources regarding PO discovery in the CR3BP (primarily papers, theses, repos from Kathleen Howell's group) but I realized my recollection of the fundamentals of stability analysis was shaky and the resources that I was referencing were a step beyond where my understanding was at.

To this end, I am working through Shane Ross's [CR3BP course](https://www.youtube.com/playlist?list=PLUeHTafWecAXDF9vWi7PuE2ZQQ2hXyYt_), which provides a good primer on the content and introduces numerical differentiation methods for orbit discovery. I brushed up on some fundamental concepts from linear algebra/numerical methods/astrodynamics, most notably in the area of stability analysis. Now, I'm ready to take a deeper dive into the actual implementation side of numerical differential correction. This will involve:
- STM
- Single Shooting
- Multiple Shooting
- Collocation

## Goals for the day
At a high level, I want to cement my understanding of differential correction, with a focus on single and multiple shooting methods today, and collocation tomorrow. I want to end the day with an better understanding of collocation and the monodromy matrix, but not necessarily do anything with this yet.
- [ ] Refer to Grebow's thesis and https://bluescarni.github.io/heyoka.py/notebooks/Periodic%20orbits%20in%20the%20CR3BP.html for numerical methods guidance
- [ ] Before coding, outline the steps required in detail
- [ ] At a minimum, Lyapunov orbit discovery
- [ ] Preferably halo or some type of 3D orbit discovery as well

# September 9

## Recap
I am currently working on understanding how QPOs are generated for my lit review/deepening of understanding of dynamics models. To guide my understanding, I am dissecting the [CR3BP repo](https://github.com/DhruvJ22/Astrodynamics_Research) that I previously used to generate training data for my neural ODEs.

Relevant terms to learn and/or implement:
- Shooting methods
	- Single shooting methods
	- Multiple shooting methods
- Differential correction
	- STMs needed for anything involving sensitivities, corrections, or stability
- Continuation methods
- Constraints
- Crossing events

I also need to finalize a research topic for the semester. I am generally interested in looking at n-body dynamics and exploiting the latent nature of neural ODEs. What can we improve using this?

## Goals
- [ ]  Study and understand all the terms above at a medium level
- [ ]  Fully understand and implement shooting methods for a simple example
- [x]  Finalize semester research topic - what are we doing that is novel? How can we exploit the latent nature of neural ODEs? What system will we study? What methods are currently used?
- [ ] Improve understanding of adjoint method - I need to know this in my sleep

## Future plans - neural ODEs for nBP
Latent space representation - a latent space is an embedding of a A latent space, also known as a latent feature space or embedding space, is an [embedding](https://en.wikipedia.org/wiki/Embedding "Embedding") of a set of items within a [manifold](https://en.wikipedia.org/wiki/Manifold "Manifold") in which items resembling each other are positioned closer to one another. Position within the latent space can be viewed as being defined by a set of [latent variables](https://en.wikipedia.org/wiki/Latent_variable "Latent variable") that emerge from the resemblances from the objects.

In most cases, the [dimensionality](https://en.wikipedia.org/wiki/Dimensionality "Dimensionality") of the latent space is chosen to be lower than the dimensionality of the [feature space](https://en.wikipedia.org/wiki/Feature_space "Feature space") from which the data points are drawn, making the construction of a latent space an example of [dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction "Dimensionality reduction"), which can also be viewed as a form of [data compression](https://en.wikipedia.org/wiki/Data_compression "Data compression").[[1]](https://en.wikipedia.org/wiki/Latent_space#cite_note-1)

Ideas:
- Encode trajectories in latent space to reveal stable/unstable manifolds for trajectory design in latent coordinates. Use to search for heteroclinic connections, design initial guesses for transfers
- Multi-fidelity residual modeling - with NBP dynamics as our baseline, let a latent ODE parameterize residuals (unmodeled forces like SRP, outgassing, drag)
- Fast surrogate propagation across regimes - encode initial pconditions to z0 svolve \dot{z}
	- Use for Monte Carlo, reachability analysis, coverage, design loops
	- Benchmark - test speedup vs high-fidelity integrator
	- Latent dynamics much simpler than raw n-body dynamics, so we get quicker propagation for Monte Carlo, optimization, design tasks
	- Use cases:
		- Trajectory Design and exploration - once trained, latent ODE can generate trajectories in milliseconds letting us sweep wide design spaces. Example - explore families of NRHOs 
		- Reachability and coverage studies cheaply
		- No: Flight computers are resource-constrained — they can’t run ephemeris-driven N-body propagation in real time.

## Summary
Today I continued research for literature review regarding periodic orbit discovery. Technical notes for relevant topics are found in Technical Notes/Periodic Orbit Discovery. I began implementing shooting methods in a Jupyter notebook. I will first present a simple problem where we only need to use root-finding methods, then advance to the more complex Kepler's problem and CR3BP where STMs are required. 

In my 1:1, John and I agreed on a research idea which will require more fine tuning, but at a high level we want to design neural ODEs to point us to POs or QPOs in complex environments (CR3BP -> 3BP -> 4BP).  We are hoping these become observable in latent space. We're not sure how to do this yet and it will require heavy research/brainstorming, but we are using Ethan Bernet's work as a jumping off point. 

We also began a semester schedule which I need to complete and fill out dates and dependencies for. All the tasks currently listed are not realistic for one semester.

# June 2025
## 6/03
Having trouble loading in and applying the saved equinox model (`.eqx`). When I load in the model, I am able to access it and view its structure but when I integrate reach the maximum numner of solver steps. I should be integrating a `Func` class term, so I load in the model as a Func and deserialize leaves accordingly. 
## 6/02
Goal: create experiment and visualization to modularize code. Begin with base class. What is the best place to organize classes/code? For now, I am going to create an `experiments` package and a `visualization` package. Each package will contain a module defining the base class, as well as separate modules for related subclasses. Do modules need to be packaged? Not sure (last time I was heavily using and creating classes was in C++ which has some differences), but it is the simplest way for me right now so I am sticking with that.

### Experiment base class
- `<model>.eqx` &rarr; experiment(model) &rarr; results (e.g. data to be plotted)
# May 2025
## 5/28
Sick
## 5/27
Primary goal for the morning is to work on visualizations. First task is saving model results appropriately so I can access all results.

I haven't used Curiosity in a while (since before it got moved). I got logged in all fine and dandy but had some permission issues related to having mlds-ml repo in my workspace. Marked the directory as safe using `git config --global --add safe.directory /workspaces/neuralODEs/mlds-ml` and everything works.

To verify that everything is working on Curiosity, I tried running the same Weights & Biases sweep that is working on my local machine. I get errors related to reading in the data - I was probably using an old dataset/format when testing on my local machine. Fix this to make the data and code consistent. 

Troubleshooting: I set the number of timesteps to propagate the orbit. When I check the number of timesteps, some have 999 timesteps while some have 1000. This is not correct. This is probably an unintended consequence of the work I was doing to create datasets with fixed timesteps. My suspicion was correct. I intend to improve the integration by fixed true anomaly functionality anyway, so for the time being I am leaving that functionality isolated on its own branch and removing those commits from downstream work. Update - using "drop" in interactive rebasing worked fine in a branch, but when I tried to sync my changes I got a lot of weird behavior related to rebasing. This happens frequently and makes it pretty frustrating to use - not sure if I am doing something wrong. This might not be advisable, but what I ended up doing to make things simpler is I rebased the branch onto main, then dropped commits directly from main rather than dropping from the branch, pushing, and merging/rebasing. This avoided some issues with conflicts between main and branch (I think, but don't quote me on that). After dropping appropriate commits, force push with `git push --force origin main`

Curiosity CPUs are slow compared to local machine. Now that I know everything is up and running correctly on relocated Curiosity, I am going to do visualization development on my local machine and then run curiosity once it's more worthwhile.
## 5/23
Code is now set up to set up to run scripts in parallel and log in wandb. It would be nice to have a way to run the script directly still to test one off scenarios so that I can quickly check if visualizations, etc are working. We don't necessarily want to run a sweep every time, especially when doing sandboxing. Tried for a little bit and it was more work than expected so I don't think it's worth pursuing this for the time being.

Went through stash and saved useful snippets, cleaned up the rest

Having issues with Ruff again where formatting is not occurring on save. 

## 5/20
Worked on same stuff as day before. Forgot to rubber duck with notes on here, but in general I have been having more success getting through pomodoro sessions. Getting back in a heavier coding flow has been a little tough since I left things in a messy state post abstract submission... don't do that again!
## 5/19
Today I am starting by cleaning up some of my code/repo that got a bit jumbled during the extended abstract crunch. I specifically want to review and keep or toss everything currently in my stash, as well as go through issues on github and archive, create PRs, close issues etc as appropriate. My git history for unpushed work is also messy right now - going to clean it up to the best of my ability in a reasonable amount of time. I have been using my local computer for development recently, but after a nice spring cleaning I will start using Curiosity again so I can run more experiments in parallel. 

Side note - my visualization script is very messy right now because I have individual functions for each visualization. I should consider implementing a visualization class at some point, but this is not a key issue right now. I am putting this on my running to do list. 

Confirming that training script can be run either directly as a `.py` file, or run multiple at once via parallelizing `sweep.sh` file. Confirmed that training script can be run directly (no images logged, only numeric metrics) with wandb sweep incorporated so that the scripts can run X number of times on one CPU (these runs occur sequentially). Confirmed that parallelized training script can be run from a shell script after some troubleshooting. This allows us to run trials on multiple CPUs at the same time, and we can still run more than one trial seqentially on each CPU (i.e. if we have 16 trials and 8 CPUs, we can run two runs sequentially on each CPU, and run all 8 CPUs in parallel). 

Now testing ability to log images. I previously scripted a minimum working example to confirm that I can plot multiple images when doing parallel runs. The minimum working example is functioning and is being used as reference. Logging multiple images is working properly.

Note about parallelized sequential runs - with the way that runs are being distributed, there are cases where one CPU is done with all tasks while others may have a backlog of tasks. This is because the hyperparamaters may just be defined such that some CPUs don't have as much work load as others. I don't think this matters for now, but it may be worth it in the future to distribute work to whichever CPUs are ready in a more dynamics fashion.