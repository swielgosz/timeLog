# RL Overview
## Markov decision process (MDP)
MDP $(\mathcal S, \mathcal A, P, r, \gamma)$
where $\mathcal{S}$ is the state space, $\mathcal{A}$ is the action space, $P(s_{t+1} \mid s_t, a_t)$ is the transition kernel, $r(s_t, a_t, s_{t+1})$ or $r(s_t, a_t)$ is the reward function, and $\gamma$ is the discount factor.

**Transition kernel**
A transition kernel is the conditional probability law of the next state given the current state and action. In the context of an MDP, it tells you how the next state is distributed given the current state and action.
It is written as $P(s \mid s, a)$, and means $P(s \mid s, a) = \text{Pr}(S_{t+1}=s \mid S_t=s, A_t=a)$, e.g. if you are in state $s$ and take action $a$, the transition kernel gives the probability of landing in next state $s$.



# PPO
Policy-gradient reinforcement - learning method designed to improve a policy reliably without letting each update aggressively change the policy. This is basically to avoid taking a gradient step which is too large and results in a worse new policy. To fix this, we soft limit how far the updated policy can move from the old one.

**Stochastic policy**
Stochastic policy: $\pi_\theta(a \mid s)$
gives the probability of taking action $a$ in state $s$. PPO updates this policy using trajectories collected from the current policy

