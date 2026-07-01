Extending training. 100 orbits, no normalization, 5000 steps, percent error loss. run_id: ukpkpr3r

100 orbits, normalize by dynamics, 5000 steps, percent error loss. run_id: vfyzpmbl. Behavior was very noisy. Put in a warmup period.

100 orbits, normalize by dynamics, 750 stpe warmup, percent error loss. run_id: 8s1eh0af


That's a cleaner question than the one I answered — let me address it directly.

Would you want ∂f/∂θ itself to have a lower range (i.e., less parametric sensitivity concentrated at periapsis)?

I'd say no, and here's the reasoning: ∂f/∂θ being large near periapsis isn't "excess effort" being wasted there — it's a reflection of the fact that fitting a 1/r²-type acceleration field requires more representational sensitivity exactly where that field is steepest. If you suppressed ∂f/∂θ at periapsis, you'd be asking the network to become less able to adjust its periapsis-region output in response to weight changes — which would make it harder, not easier, to fit accurately there. The true dynamics genuinely vary more per unit time near periapsis; a network that fits the whole orbit well necessarily needs more "adjustability" there. Flattening ∂f/∂θ would be actively fighting the shape of the physics, not just rebalancing training emphasis.

This is the core distinction: "how much the network can change its output when weights change" (∂f/∂θ) is not the same knob as "how much training signal periapsis contributes to weight updates" (the eq. 5 integrand, a(t)ᵀ ∂f/∂θ). You can flatten the second without touching the first — that's exactly what dividing the integrand by ||f_accel|| already does: it leaves the network free to have high sensitivity at periapsis (as it should, to fit the steep dynamics there), while preventing that local sensitivity from dominating the accumulated parameter gradient over the whole orbit.

So to directly answer: no, I don't think you'd want to lower ∂f/∂θ's range near periapsis as a goal in itself. The goal should stay at the level you were already working at — flattening the effective integrand (what actually drives parameter updates) — while letting ∂f/∂θ be whatever it needs to be for the network to represent the true dynamics well at each point.