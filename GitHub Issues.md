Github issues:
1. Write and validate the mathematical explanation of the adjoint-method diagnostics for the neural ODE training process.

	Expected output:
	
	- Double-check adjoint calculations for correctness.
	- Explain the intuitive role of each term in the integrand, especially the adjoint, dynamics-to-parameters term, and their combined effect.
	- Connect the forward/backward integration view to how model updates occur.
	- Record any remaining confusion or open questions as comments so the discussion can continue asynchronously.

	Follow up: validate the adjoint/vanishing-gradient interpretation on a simple dynamical system such as a spring-mass-damper or exponential-growth model. Compare segmented and full-horizon calculations, document whether the same sensitivity pattern appears, and close with a short state/resolution write-up suitable for the repository or devlog.

	Follow up: 
	Connect the adjoint-method verification work to the current questions about Sundman and Levi-Civita transformations.
	- [ ] Document the role of the Sundman transform in the current model and why small time increments may destabilize the solver.
	- [ ] Write a short technical note or issue comment explaining the Levi-Civita transform at the level needed for debugging discussions.

2. Implement and document the local diagnostic method for CR3BP neural ODE models using Floquet analysis.

	Expected output:
	
	- Compare Floquet multipliers for true and learned dynamics where applicable.
	- Integrate perturbed trajectories and summarize whether local behavior matches the expected dynamics.
	- Identify which orbit families or regions expose model failure most clearly.
	- Include plots and written interpretation suitable for the journal-paper diagnostic package.
3. Evaluate two training modifications intended to reduce local-dynamics bias and segment-boundary compounding error.
	
	Expected output:
	
	- Implement a loss normalization based on the magnitude of the local dynamics, with safeguards near small denominators.
	- Test a multiple-shooting-style endpoint tolerance or curriculum that begins leniently and increases precision during training.
	- Compare both modifications against the current baseline using training stability and long-term rollout metrics.
	- Include plots and a concise interpretation of whether either method addresses periapsis bias or segment-boundary error.
	- Record a recommendation for follow-on experiments.Sprint update from advising notes: run a longer training session, approximately 8 hours, with a larger dataset and without speed optimizations to test whether the gradient/dynamics-normalized method improves over a longer horizon. Include the current baseline, training curves, rollout behavior, and a short interpretation of whether the normalization is actually helping.
	
	Note: This has been done but figures attached to issue would be good.