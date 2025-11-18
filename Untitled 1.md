So width tends to give immediate gains, while depth is powerful but riskier in neural ODEs due to solver stability.

---

what did you mean by this: . I’d target better regularization or capacity in the output head specifically around enforcing unit direction vectors and coupling them to the magnitude.

I meant that the part of the network producing the acceleration vector (its magnitude plus direction components) should be guided to output unit-length directions tightly coupled to the magnitude. Two concrete ideas:

- **Capacity:** Add neurons/layers or stronger nonlinearities in the output head so it can model sharp changes near periapsis without overshooting. Right now it seems underpowered.
- **Regularization/constraints:** Encourage the direction outputs to stay unit-normalized and aligned with the magnitude, e.g., by normalizing the direction vector, adding a penalty when ‖dir‖≠1, or structuring the head so it predicts an unconstrained vector that you explicitly normalize before multiplying by acc_mag. That keeps the direction smooth even when magnitude spikes.

Together these adjustments give the head enough expressiveness while forcing physically consistent outputs—better matching the truth during those periapsis spikes.

--- 
