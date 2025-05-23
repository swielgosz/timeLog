# 5/23
Code is now set up to set up to run scripts in parallel and log in wandb. It would be nice to have a way to run the script directly still to test one off scenarios so that I can quickly check if visualizations, etc are working. We don't necessarily want to run a sweep every time, especially when doing sandboxing. Tried for a little bit and it was more work than expected so I don't think it's worth pursuing this for the time being.

Having issues with Ruff again.

# 5/20
Worked on same stuff as day before. Forgot to rubber duck with notes on here, but in general I have been having more success getting through pomodoro sessions. Getting back in a heavier coding flow has been a little tough since I left things in a messy state post abstract submission... don't do that again!
# 5/19
Today I am starting by cleaning up some of my code/repo that got a bit jumbled during the extended abstract crunch. I specifically want to review and keep or toss everything currently in my stash, as well as go through issues on github and archive, create PRs, close issues etc as appropriate. My git history for unpushed work is also messy right now - going to clean it up to the best of my ability in a reasonable amount of time. I have been using my local computer for development recently, but after a nice spring cleaning I will start using Curiosity again so I can run more experiments in parallel. 

Side note - my visualization script is very messy right now because I have individual functions for each visualization. I should consider implementing a visualization class at some point, but this is not a key issue right now. I am putting this on my running to do list. 

Confirming that training script can be run either directly as a `.py` file, or run multiple at once via parallelizing `sweep.sh` file. Confirmed that training script can be run directly (no images logged, only numeric metrics) with wandb sweep incorporated so that the scripts can run X number of times on one CPU (these runs occur sequentially). Confirmed that parallelized training script can be run from a shell script after some troubleshooting. This allows us to run trials on multiple CPUs at the same time, and we can still run more than one trial seqentially on each CPU (i.e. if we have 16 trials and 8 CPUs, we can run two runs sequentially on each CPU, and run all 8 CPUs in parallel). 

Now testing ability to log images. I previously scripted a minimum working example to confirm that I can plot multiple images when doing parallel runs. The minimum working example is functioning and is being used as reference. Logging multiple images is working properly.

Note about parallelized sequential runs - with the way that runs are being distributed, there are cases where one CPU is done with all tasks while others may have a backlog of tasks. This is because the hyperparamaters may just be defined such that some CPUs don't have as much work load as others. I don't think this matters for now, but it may be worth it in the future to distribute work to whichever CPUs are ready in a more dynamics fashion.