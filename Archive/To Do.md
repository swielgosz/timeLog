# February
## 2/10 - 2/14
### 2/12
- Integrated Obsidian with git for reliable syncing and backup
- Relocating Zotero papers and improving backup
- Troubleshooting git interactive rebasing `reword` functionality
	- Can't edit commit message:
		1) Change to topic branch
		2) `git rebase -i HEAD~1`
		3) Change `pick` to `reword` for relevant commit
		4) Edit COMMIT_MSG, save and accept commit message 
		5) Error:
			```fatal: cannot lock ref 'HEAD': is at 7cc8818c8e25dbc3f89c61c6e244f07cb99288bc but expected e296f35c5ab4895eaabd109b197631bbc7067145 Could not apply e296f35... Fixed fps```
- Documenting and practicing git interactive rebasing
- Explored mlds repo

### 2/13
Morning:
- Emails/admin
- Addressing `neuralODE` issues \#1-\#3 and familiarizing myself with necessary concepts such as creating packages for installation, `pyproject.toml`, etc
- Learning more about rebasing for the purpose of continuing development from an unapproved merge request branch

Afternoon:
- Advising meeting
- Read through of "AN UNSUPERVISED PHYSICS-INFORMED NEURAL NETWORK FOR FINDING OPTIMAL LOW-THRUST TRANSFER TRAJECTORIES WITH THE DIRECT METHOD" and related research

### 2/14
Forgot to record - worked on some neuralODE coding stuff, tried to read but wasn't able to focus well - not a very productive day :(

## 2/17-2/21
Must dos for the week:
- [x] Port code over from my FaultDetection repo to MLDS neuralODe repo 
- [ ] Implement CR3BP
- [x] Implement spherical coordinates
- [ ] NeuralODE trained on multiple orbits (currently just one)
- [x] Read!! I've been having trouble focusing while reading. Suck a labmate in to listen to my ramblings 
	- [x] Put together lit review for group meeting

Other things that would be useful if I have time:
- [ ] Add classes to code. Looking at toy gitBoarding repo, I've realized there are a few structural changes I should make to my code (functionality is fine for what I've been doing, but I should future proof it more for when code gets more complex).
- [ ] Generalize neuralODE training - I want to be able to define runs using a config file
### 2/17
- Ported code over and reworked packaging structure/relevant related imports
	- This was the main task of the day - sorting through, restructuring, and patching code was time consuming (it was admittedly a refresher for me and required some fix ups. I'm getting heavily back into my code for the first time in a while so). Had some silly packaging issues but they are resolved and everything is working when the correct workspace is set
- Spent some time trying to get plots to show on local machine. I'm having trouble with VcXsrv and whatnot but don't want to waste more time on this until I get my new computer. For the time being, I'm going to save plots as .png files

### 2/18
Goals of the day:
- [x] Spherical coordinates
- [ ] CR3BP
- [ ] NeuralODE trained on multiple orbits
- [x] Work on lit review and finalize paper for Friday

Things I actually did/accomplished:
- Spherical coordinates/review of dynamics
- General code improvements/putting out associated fires/learning more about related git and python stuff 
	- The coding stuff always takes longer than I expect and I should probably be less ambitious for the time being (or be okay with not completing all goals)
- Research direct and indirect OCP
- Advising meetings

### 2/19
Goals of the day:
- [ ] !! Neural ODE trained on multiple orbits
- [x] Set up new computer :)
- [x] Work on lit review
- [ ] CR3BP

Things I actually did/accomplished:
- Computer handoff and set up - took about half the day between the pickup and set up
- X-lab group meeting
- OCP research

Reflection of the day:
- Good things that happened:
	- Computer is completely set up to my knowledge! Hopefully won't have any more issues
- Neutral things:
	- So much walking between meetings (brr). Good for health, bad for taking away useful time
- Less good:
	- Didn't get to work on neuralODE code
	- Very tired

Today's rating: 5.5/10

### 2/20
Goals of the day:
- [ ] neural ODE trained on multiple orbits!!!!! Today is the day!!!
	- [ ] doesn't need to be perfect, I just want to get the infrastructure set up
- [x] Finalize lit review presentation and write down talking points
- [x] Get visualization working
- [x] verify that spherical coordinates are working (I previously coded them in but didn't have a chance to check if it was correct)

Things I worked on:
- visualization working
	- update - earlier I (think I) had it working where if I ran code in devcontainer in vscode, plot would appear locally. Now, when I run code from devcontainer,  plots do not appear. It works fine if I run code from curiosity, it's just inconvenient to have vscode open where I'm actually working and a terminal just for running the code for visualization. what did I break?
- Spherical coordinate conversion and 2BP in spherical coordinates not working properly
	- Conversions are incorrect - I was referencing online resources because I didn't want to make silly typos or mistakes in my conversions. Those aren't correct, so I am deriving them by hand
- Put together lit review

Today's rating: 7/10. Not many tangible results, but I was very focused
### 2/21
- Put together lit review talking points
	- Reviewed OCP resources until meeting to guide talking points
	- Post meeting reflection: 
		- I didn't focus much on HJB equation for indirect method, but should review this for my own sake
		- PMP:
			- We understand the equations and *roughly* how they are analogous to physics based Lagrangian, Hamiltonian, etc. But, we should have a more solid understanding of exactly where the controls verison of these equations come from, and what the costate equation is representing (i.e. is it significant that costates and adjoints are interchangeable terms? are these the same or different than lagrange multipliers and how? using lagrange multipliers, we try to minimize energy in a physical system. what is the control analogy of this?)
- MLDS Group meeting

Today's rating: 6.5/10. Group meeting made me realize I knew less than I thought I knew and I was feeling pretty brain foggy by presentation time, but I like lab meetings and people's involvement in lit review discussions. 
## 2/24 - 2/31

Top priorities:
- [ ] Get spherical coordinates (both conversion and 2BP dynamics) working properly
	- [ ] Derive coordinate conversions and dynamics by hand - references have not been reliable
- [ ] Train neural ODEs on multiple orbits
	- I have been waiting for spherical coordinates to be working to try this out, but this keeps delaying me and I should just go ahead and at least set it up with Cartesian and/or orbital elements
- [ ] Get CR3BP working
	- [ ] Add CR3BP dynamics to code
	- [ ] Train neural networks to learn CR3BP

Other things that would be useful if I have time:
- [ ] Add classes to code. Looking at toy gitBoarding repo, I've realized there are a few structural changes I should make to my code (functionality is fine for what I've been doing, but I should future proof it more for when code gets more complex).
- [ ] Generalize neuralODE training - I want to be able to define runs using a config file
- [ ] Review some of the OCP concepts to solidify some muddy points that came up in group meeting

Intentions for the week to be successful!:
 - I keep setting goals ambitiously and then operate slower than necessary to achieve all of them (and/or more fires come up to put out than I would like). No problem - this week I will try setting fewer "must do" goals and a couple stretch goals each day
 - Weather is nice - go for some walks!
 - I'll probably be working from home more this week and over the next month due to very inconveniently placed construction. Make sure to maintain good habits at home (take breaks, eat lunch, wear real pants, etc)
 - Limit news reading to ~20 minutes a day. Want to stay informed but not wallow in an unproductive manner. Similarly, read more novels instead.
 - Set daily to do lists for the next day at the end of the day. Lists are fantastic! Currently, I do this at the beginning of the day but I should use my fresh brain power on more challenging tasks, and make lists at the end of the day when I'm tired.
### 2/24
Realistic goals:
- [x] Fix spherical coordinates
- [ ] Train neural ODEs on multiple orbits

Things I worked on:
- Went down a rabbit hole refreshing on coordinate frames, conversions, etc. My progression of thoughts:
	- Original plan: 
		1) Derive 2BP in spherical coordinates. Why did I want to derive it in spherical coordinates? 2BP is typically expressed in cartesian coordinates which is analytically convenient, but orbits would be more naturally described in terms of spherical coordinates. I wanted to be able to directly integrate in terms of spherical coordinates, rather than integrating in cartesian and converting to spherical in order to minimize numerical error. Is this necessary/ a good idea? Maybe not
		2) Since we would realistically describe our orbit's initial conditions in terms of spherical coordinates or orbital elements, write conversion script to convert from cartesian to spherical coordinates (I already have Cartesian to orbital elements and vice versa)
	- I was able to derive 2BP dynamics (not yet verified to be correct)
	- Great! Now we need to convert initial conditions to spherical coordinates. It is simple to convert *coordinates*, but we also need *velocities* in spherical components. No worries - consult internet resources/notes from dynamics.
	- Oh no! We have never explicitly discussed conversions directly between cartesian and spherical velocities. Rather, we begin with $\vec{r} = r\hat{e}_r$ and differentiate from there to get velocity and acceleration in spherical coordinates
	- That's okay. I'll refresh on my dynamics and try to search around/derive this myself
	- Some of the online discussions that basically identified my confusions:
		- https://physics.stackexchange.com/questions/763908/direct-conversion-of-cartesian-velocity-to-spherical-velocity-and-vice-versa
		- https://physics.stackexchange.com/questions/546479/conversion-of-cartesian-position-and-velocity-to-spherical-velocity
	- Is any of this even worth it? Let's say we ignore all these shenanigans and don't bother converting anything into spherical velocities. We can integrate in cartesian coordinates and convert the cartesian positions at each time step to spherical coordinates. However, we wouldn't have full state information in spherical coordinates since we would be missing velocities. But does this matter in terms of our ultimate goal of training neural ODEs? We can train the neural network to learn coordinates at each time step - we don't actually need velocities for training, just for generating data (which we can do with cartesian velocities) - right? 
	- After taking a break and resting my dynamics brain, velocity conversions have been derived, and acceleration equations were verified
	Overall: spend a lot of time reviewing dynamics, bounced back and forth between "this isn't worth it" and "no it's not that bad". Got a little sidetracked/intimidated by online discussion of GR and nonholonomic coordinate systems, but we should be good for now. 

Today's rating: 3/10. Brain super foggy and I was so tired I couldn't think clearly until ~6 pm. Not sure why - I've been sleeping ~8 hrs, eating well, etc. But on the bright side, once I reviewed notes and was thinking more clearly, everything made sense pretty rapidly.

### 2/25
Realistic goals:
- [ ] TRAIN NEURAL ODE ON MULTIPLE ORBITS!!!!! 
	- [ ] This keeps being at the end of my list and I get sidetracked. Just do it!!!!!!!!!!!!!!!!!!!!!!!!!! Will feel shame if I don't do it today
- [x] Put new spherical coordinate conversions in code
- [ ] Address comments from previous PR

- Coordinate conversions have been fully derived and are working (working as in when I convert from spherical to cartesian and vice versa, I get my original input. However, this does not necessarily mean these results are correct)
- Integrating 2BP via spherical coordinates is functional, but giving garbage results
- So, what is wrong? Is it my conversions from spherical to cartesian coordinates/velocities? Is it my 2BP dynamics? Do I have a typo somewhere? Did I not consider angle quadrant somehwere? Hmm... I can just propagate using classic cartesian 2BP propagator and then convert to spherical coordinates, but I don't like this for reasons stated in yesterday's notes

### 2/26:
- [ ] Pull request for spherical *coordinates* (not velocities/dynamics)
- [ ] Address comments from PR #2
- [x] Draft of introduction - doesn't have to be good! Hard limit - 2 hrs
- [x] After intro draft is completed, determine:
	- [x] what my plot(s) for next week should be
	- [x] next coding priority
- [x] Put together intro slides for X-Lab group meting:
	- [x] First conference paper and journal paper timeline
	- [x] Answer Heilmeier questions

Things I worked on/did:
- Paper reading/collection for about 1.5 hours - I have a list of some other 
- Introduction draft
- Paper planning
- Meeting prep
- Meetings

Reflection of the day:
7/10 - happy to overcome the hurdle of beginning writing, and I got tacos in group meeting. Downside - didn't work on much coding. I would like to be better at switching between literature tasks and coding tasks

### 2/27:
- [x] Address comments from PR #2
- [x] PR for spherical coordinates
- [ ] Train using spherical coordinates with keras.sequential model to verify spherical coordinates are working properly
- [ ] Train using multiple orbits
- [ ] Update neuralODE code to accept different coordinate systems - try this first just with cartesian single orbit vs spherical single orbit. Then move to cartesian multiple vs spherical multiple orbits.

Got first two things done, having some gnarly headaches so I'm operating at half capacity but hoping to make up some time over the weekend and get some good results! I want at least one good plot by Monday, then tackle CR3BP on Monday

Linters not working? I am working in a virtual environment in a dev container. I might just need to make some adjustments to make linters work in the venv. Might want to rethink workflow at some point, but not bothering with it now. 

### 2/28:
Still having bad headaches, aiming for fourish solid hours of work and then do some work over the weekend to compensate since it's time to get things done
- [ ] Train using spherical coordinates with keras.sequential model to verify spherical coordinates are working properly
- [ ] Train using multiple orbits
- [ ] Update neuralODE code to accept different coordinate systems - try this first just with cartesian single orbit vs spherical single orbit. Then move to cartesian multiple vs spherical multiple orbits.

# March
## 03/03 - 03/07
Top priorities:
- [ ] Start compiling plots for paper
	- [ ] Show that neural ODEs are effective at learning known dynamics by comparing true state to estimated state after training
		- [ ] CR3BP has not yet been implemented - we can start with 2BP to get a plot but CR3BP should be done this week
		- [ ] Train on multiple orbits to get more accurate results
- [ ] Begin methodology
- [ ] Work on cleaning up introduction

### 03/03
- [x] Verify spherical coordinates are working - train vanilla neural network on spherical coordinates for multiple orbits
- [ ] Train using multiple orbits with neuralODE. Steps:
	- [ ] Begin by training on spherical coordinates. Write out data structure - if we have multiple orbits, data about the initial condition and time of flight needs to be considered (as opposed to training on a single orbit where we had the same initial condition for each tof)
	- [ ] For now, if it's easier just create separate scripts for different coordinate systems and then later we can co back in and add better configuration set up
	- [ ] Do we want training use vanilla neural network for comparison?
- [x] Figure out why linter isn't working? I am using dev container 
	- [ ] Oops! It is working, I was just overlooking error messages


- All approved pull requests should now be on main. Took some trial/error but I think it is okay for now
	- Do we always want to rebase onto main and then force push that branch? Makes history more linear, but you also lack seeing which issues were addressed in certain spots
- Met with Pitt student
- Yapped with Logan
- Updated orbit propagation and data generation codes so datasets (both for single and multiple orbit datasets) include spherical coordinates
- Fixed formatting 
- To verify spherical coordinates are behaving as expected, I trained a neural network on 1000 different orbits in spherical coordinates. Code runs but results are garbage (interestingly, predicted results look very similar to my propagated orbit when I tried implementing 2BP in spherical coordinates. Likely not a coincidence). Go through and debug. I suspect one of my data structures is being referenced incorrectly. Reason to spend time on this - it will serve as baseline for comparison against neuralODEs. It isn't novel though, so switch focus to training neural ODE and come back to this if it's not resolved relatively quickly. 
- Simplify data format for training to make feature and label extraction simpler

### 03/04
- Restructured data generation code for simpler data structure
- Worked on implementing training on multiple orbits for vanilla neural network
	- Results with spherical coordinates are bad - may not have enough training data (used 1000 orbits)? Debugged some issues but didn't find anything immediately to fix issue
- Advising meetings
- Restored code and setting up dev container after accidentally deleting my entire workspace. I think only lost one fairly minor commit
- Worked on neural ODE code trained on multiple orbits - I am starting by going through the code in finer detail than I did before. Previously, I was mainly focused on implementation but I need to understand what is happening better before adjusting it for training on multiple orbits

Overall: great day! no major results, but I'm happy with the work that was done behind the scenes and I was super focused

### 03/05
- Spent a couple hours on expense report for travel 
- Group meeting
- Researching how to implement multiple orbits/initial attempts

### 03/06
- re: multiple orbits: realized I am being a dingus! Everything makes sense now! Worked on implementation
- Read Salamander - per their warning about instability in spherical coordinates for neural ODEs, I am going back to Cartesian for the time being

### 03/07
Accepted students day

## 03/10 - 03/14
### 03/10
- Implemented neural ODE with different (but same length) time series 
- Loss is converging
- When testing the learned ODE, the solver cannot numerically integrate. I think the problem is too stiff an we need more training points around periapsis and/or smaller dt

### 03/11
- Put together outline with some plot sketches
- Advising meetings
- Sick, slept a lot

### 03/12
Per 03/10 notes, after neural ODE finishes training, the solver fails to integrate given initial conditions. I suspect this is due to a lack of training data around periapsis.
Goals:
- [ ] Prove if this assumption is correct
- [ ] If yes, integrate by true anomaly rather than time and retrain

Other goals:
- [ ] Address PR comments

## 03/17-03/21

### 03/17
- Addressed PR comments and merged all PRs into main. Had a good deal of trouble with various conflicts when I would try to rebase my topic branch onto main before merging. I fixed and cleaned up what I could, but git history isn't exactly what I want it to be
- Improved multiple data generation code so that orbital elements were stepped through in a defined manner, rather than creating orbits using random orbital elements within a defined range. I used this dataset for testing and results are garbage. Possible reasons:
	- Through debugging, I noticed that features are not well scaled and show little variation. I'm not exactly sure why this is coming up now even though the datasets generated are the same as previously, and  the ranges of orbital elements from which those datasets are created are the same? Options to fix: 
		- Scale based on characteristic length and time (this is probably the best option long term)
		- Fix scaling such that we see more reasonable results like we did previously. I suspect this is related to how my data is reshaped before scaling
	- Feature scaling seems highly likely, but it is also worth noting that before, although orbital elements were randomized and we didn't have as fine control, this did result in a higher variation of training data. For example, every sample would have had a different eccentricity, while now we have 20% of our samples with the same eccentricity. This shouldn't matter since the neuralODE's strength should be that it can learn on less data, but worth taking note of and perhaps may inspire a different dataset generation method in the future (i.e. instead of nested for loops, utilize the zip function)

Previous example of the scaling applied to the first orbital sample (good): ![[Pasted image 20250318104758.png]]
Now (bad):
![[Pasted image 20250318104613.png]]
- A lot of the day was debugging and doing manual runs

## 03/24 - 03/28
### 03/24
Today and last Thursday were unfortunately slow due to travel/unstable wifi. Overall things I was working on:
- Performance still hasn't been great when I do manual runs. I wanted to go back to training on small datasets very close to my test orbit to see if results would be better. Through this and previous tests, I identified that data generation is absurdly slow which only began after PR #35. I haven't been able to identify the bottleneck
- Added eval loss to plots

### 03/25
- There was a bug where mu was being overwritten with mu_char=1 in the data generation script each loop, resulting in calculation of much higher periods and longer propagation times for each orbit. Fixed this and results are looking good again
- Started testing on multiple orbits instead of just one
- Wrote for an hour
- Advising meetings

### 03/26
Goals for the rest of the week:
- [x] Now that code is functional, do some more manual runs to get a feel how training performs using different parameters/architectures. 
- [ ] Set up config file and experiment file. 
- [ ] Set up wandb
- [ ] Figure out multiprocessing
- [ ] Improve multiorbit visualization

03/26 and 03/27, made progress toward above goals ^ I don't think my loss function is ideal right now. 
### 3/28
- Learning inkscape
- Elevator pitch prep
- Potential student lunch/talk for Mumu's lab
- Group meeting


# April
## 03/31 - 04/04
Goals for the week:
- Extended abstract draft!
	- Concept figure
	- Preliminary results

### 03/31
- Created loss module
- Implemented sharding so training can be parallelized! This was basically the whole day



Note: switching to using Research Notes rather than To Do  since this is redundant from personal notes