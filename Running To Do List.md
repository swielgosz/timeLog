# End of December:
- [ ] FIX COMMITS
- [ ] Figure out journal paper visualizations
- [ ] Figure out journal paper "plot" - I think the latent ODE linearization will be in the conference paper but will we have latent ODEs in the journal?
- [ ] Figure out when mlp_4D is worse

# December 11
Before meeting:
- [ ] Understand latent ODEs more. Am I using the "correct" kind? How do we aim to use it?
- [ ] pretty stats visualizations
- [ ] compile thoughts into powerpoint
- [ ] Why does mlp_4D_signed work the best?? Does it?
- [ ] Redo the heat map

Questions:
- Does the segment length affect generalization?
- Should we try this again with validation data? This shouldn't really be necessary - this is what postprocessing is for. When we see how models generalize to unseen datasets, we are basically measuring generalization gap. However, we save the best model based on the lowest training loss and I can't say that the best training loss corresponds to the best validation loss for all of these models.

# November 18:
- [ ] Speed up run - we are running diffrax solver twice in the case that we save off intermediate details
- [ ] unify teature_layers and output_layers formatting
- [ ] before meeting, get interactive plots working for inspection
- [ ] refresh on what batch size 

- [x] parallelize data generation
	- [ ] if datasets are very large - will we have to break them up to save/upload them for memory purposes? for the time being, curiosity should be fine. ETA: the size limit on artifact uploads to wandb is 5gb per file, so we shouldn't have to worry about this
- [ ] new acceleration metric - time averaged acceleration error?
- [ ] add a config for cpu vs gpu
- [ ] 