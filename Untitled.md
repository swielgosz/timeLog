we work on astro and ML. we are looking at trajectory design, and in particular these useful orbits 

john liked when I said spawn transfer from

don't say: neural odes are better at learning smoother represenations of the dynamics
don't say the proof of concept is useful
cushion myself - these are preliminary results. we want to prove that we can reproduce these results 


Today I continued research for literature review regarding periodic orbit discovery. Technical notes for relevant topics are found in Technical Notes/Periodic Orbit Discovery. I began implementing shooting methods in a Jupyter notebook. I will first present a simple problem where we only need to use root-finding methods, then advance to the more complex Kepler's problem and CR3BP where STMs are required. 

In my 1:1, John and I agreed on a research idea which will require more fine tuning, but at a high level we want to design neural ODEs to point us to POs or QPOs in complex environments (CR3BP -> 3BP -> 4BP).  We are hoping these become observable in latent space. We're not sure how to do this yet and it will require heavy research/brainstorming, but we are using Ethan Bernet's work as a jumping off point. 

We also began a semester schedule which I need to complete and fill out dates and dependencies for. All the tasks currently listed are not realistic for one semester.