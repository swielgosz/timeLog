# Resources
Patrick Kidger example: https://docs.kidger.site/diffrax/examples/latent_ode/
Patrick Kidger thesis: https://arxiv.org/pdf/2202.02435
Latent Stochastic Differential Equations for Irregularly-Sampled Time Series - David Duvenaud https://www.youtube.com/watch?v=93wr5vhI0F4&t=346s
	This probably isn't the best for now since there is a stochastic component
A Cookbook for Deep Continuous-Time Predictive Models https://www.youtube.com/watch?v=aYwDBKIkKY4&t=964s
Latent ODEs for Irregularly-Sampled Time Series (Reading Papers) huyjytgttps://www.youtube.com/watch?v=tOkH339Wucs
	Associated paper is in Zotero
FFJORD: FREE-FORM CONTINUOUS DYNAMICS FOR  SCALABLE REVERSIBLE GENERATIVE MODELS
	I can't remember where I originally saw this referenced - might not be the most crucial


# Parts of an autoencoder

# Questions
What is the difference between stochastic vs not stochastic latent ODEs?
what is the difference between MLP and GRU as the encoder?

# Latent ODEs for Irregularly Sampled Time Series
## RNNs
We first need to understand Recurrent Neural Networks (RNNs). They are a class of neural netwroks designed to work with sequential or time-ordered data by maintaining a memory of previous inputs. 
Unlike a standard feed-forward neural network, which processes each input independently, an RNN:
	•	Takes an input at time step t
	•	Combines it with an internal hidden state (information from previous steps)
	•	Produces a new hidden state (and possibly an output)
This lets the network model temporal dependencies such as:
	•	Words earlier in a sentence
	•	Previous sensor readings in a time series
	•	Earlier states of a dynamical system
They don't learn long-range dependencies well, but variants like LSTMs and GRUs addressed this. 

---

**Mathematical Form**
A simple (vanilla) RNN can be written as:
$$
\begin{gathered}
h_t = \tanh(W_x x_t + W_h h_{t-1} + b) \\
y_t = W_y h_t
\end{gathered}
$$
Where:
- $x_t$ = input at time t
- $h_t$ = hidden state (the “memory”)
- $y_t$ = output
- $W_x, W_h, W_y$ = learned weight matrices
The **same weights** are reused at every time step.

Example - predicting vibration faults in a rotating machine. You record vibration signals over time and want to detect or predict a fault. Each data sample is a time series, not a single vector. At each timestep $t$ you observe acceleration, maybe temp, RPM, etc. But the system's health status depends on patterns over many previous time steps, not just the instantaneous value. An RNN processes the signal one timestep at a time, carries a hidden state that summarizes the recent history, and learns temporal signatures like growing harmonics or periodic instability.

Variants to address vanishing and exploding gradient issues:
- LSTM (Long Short-Term Memory)
	- Uses gates (input, forget, output)
	- Can retain information over long time spans
- GRU (Gated Recurrent Unit)
	- Simpler than LSTM
	- Fewer parameters, often similar performance

![[Pasted image 20260108230822.png|500]]

**What this paper introduces**
RNNs are not suitable for irregularly-sampled time series data. Better approach is to construct a continuous-time model with a latent state defined at all times. 
![[Pasted image 20260109003931.png]]

Introduces two different ways to use the ODE-RNN:
1. Standalone autoregressive model. An AR model is one where future values are predicted using past values of the same variable. This is the generative model portion of the paper. 
	In a standalone ODE-RNN, the hidden state h(t):
		1.	Evolves continuously according to an ODE when nothing is observed
		2.	Jumps discretely via an RNN cell when a new observation arrives
		There is no latent variable z_0, no variational posterior, no encoder/decoder split. The model directly generates the next observation.
2. ODE-RNN as a recognition network (latent ODE)
Here the structure is:

Generative model (Latent ODE)

z_0 \sim p(z_0)
\dot{z}(t) = f_\theta(z(t))
x_{t_i} \sim p(x \mid z(t_i))

Recognition model

q(z_0 \mid \{x_{t_i}, t_i\})

And this is where the ODE-RNN comes in:
	•	The ODE-RNN processes observations
	•	Outputs (\mu_{z_0}, \sigma_{z_0})
	•	Defines the approximate posterior

In this case, the ODE-RNN is not the generative model — it is the encoder. The Neural ODE + decoder is the generative model.

2BP analogy (to lock it in)

Think of:
	•	ODE evolution = propagating an orbit
	•	RNN update = incorporating a tracking measurement

That’s exactly what’s happening.

If you were learning just gravity, you wouldn’t want the jumps.

"We generalize state transitions in RNNs to continuous-time dynamics specified by a neural network, as in NeuralODEs"

### 3.1 - Construction an ODE-RNN Hybrid
"We define the state between observations to be the solution to an ODE: h′i = ODESolve(fθ, hi−1, (ti−1, ti)) and then at each observation, update the hidden state using a  standard RNN update hi = RNNCell(h′i, xi)" meaning:
	A standard RNN assumes data arrives at regular, discrete time steps and just does ht=tanh⁡(Whht−1+Wxxt+b)h_t = \tanh(W_h h_{t-1} + W_x x_t + b) ht​=tanh(Wh​ht−1​+Wx​xt​+b). But what if your observations are irregularly spaced in time? A standard RNN has no notion of how much time passed between steps — it treats every step the same regardless of whether the gap was 1 second or 1 hour. This is the problem the paper solves.

The update happens in two steps. First, between observations, the hidden state is evolved continuously through time via $h_{i}^{`}=\text{ODESolve}(f_\theta,h_{i−1},(t_{i−1},t_i))$. Instead of doing nothing between observations, the hidden state is evolved forward using a Neural ODE. $f_\theta$​ is a learned vector field, and the ODE solver integrates it from ti−1t_{i-1} ti−1​ to tit_i ti​. This naturally respects the time gap — if the gap is large, the state evolves more. The hidden state $h'_i$​ represents your best guess of the system's state just before the i-th observation arrives.

Second, once a new observation arrives, you assimilate it via hi=RNNCell(hi′,xi)h_i = \text{RNNCell}(h'_i, x_i) hi​=RNNCell(hi′​,xi​). This is a standard RNN update that corrects or updates the hidden state based on what you actually observed.

The intuition is a two-phase cycle repeating at each observation: first predict using the ODE to propagate the hidden state forward through time, then update using the RNN cell to assimilate the new data point. This should feel familiar if you know Kalman filtering — it's essentially a predict/correct loop, but fully learned.

In your 2BP/CR3BP setting, the ODE step could encode meaningful dynamics rather than just being a black box — the hidden state between observations evolves according to learned orbital mechanics, and the RNN step corrects it when new measurements come in. This is very natural for astrodynamics where you might have sparse or irregular tracking data.

"Autoregressive models make a one-step-ahead prediction conditioned on the history of observations, i.e. they factor the joint density p(x) = ∏  i pθ(xi|xi−1, . . . , x0). As in standard RNNs, we can use  an ODE-RNN to specify the conditional distributions pθ(xi|xi−1...x0) (Algorithm 1)." meaning:

Say you have a sequence of observations x_0, x_1, x_2, ..., x_n — for example, a time series of satellite positions. The joint density p(x) is simply the probability of observing that entire sequence together. It answers the question: "how likely is this whole trajectory?"

A conditional distribution p(x_i | x_{i-1}, ..., x_0) is the probability of the next observation x_i given everything you've seen so far. It answers: "given the history of the trajectory up to now, how likely is the next observation to be x_i?"

The key identity is just the chain rule of probability. You can always decompose a joint probability over a sequence as p(x) = p(x_0) * p(x_1 | x_0) * p(x_2 | x_1, x_0) * ..., which is written compactly as p(x) = prod_i p_theta(x_i | x_{i-1}, ..., x_0). This is not an assumption — it's always exactly true by the rules of probability. The product just means you're multiplying together the probability of each observation given all previous ones.

An autoregressive model says: I want to learn each of those conditional distributions p_theta(x_i | x_{i-1}, ..., x_0). But storing the full history at every step is expensive, so instead you compress the history into a hidden state h_{i-1} and approximate p_theta(x_i | x_{i-1}, ..., x_0) ≈ p_theta(x_i | h_{i-1}). This is exactly what an RNN does — the hidden state is a lossy summary of the history, and you use it to predict the next observation. The ODE-RNN just replaces the discrete hidden state transition with a continuous ODE between observations, which is especially useful when the time gaps are irregular.

So the big picture is: the joint probability of a whole sequence can be broken into a product of one-step-ahead predictions, and a recurrent model (RNN or ODE-RNN) is a natural way to parameterize those predictions because it maintains a running summary of history.

### 3.2 Latent ODEs: a Latent-variable Construction
Intro information:

We've been discussing the autoregressive approach. In that framing, the ODE-RNN directly models the conditional distributions p(x_i | x_{i-1}, ..., x_0) by maintaining a hidden state that summarizes history and predicting the next observation from it. There's no latent variable — the hidden state is directly tied to the observations.

The latent variable construction is a different and more powerful approach. Instead of directly modeling the observations autoregressively, you postulate that there exists some underlying latent state z that evolves continuously according to an ODE, and the observations x_i are noisy projections of that latent state. This is the "Latent ODE" model of the paper's title, and it's more in the spirit of a variational autoencoder — you have an encoder that infers the initial latent state z_0 from the data, an ODE that evolves z_0 forward in time, and a decoder that maps the latent state to observations.

The key conceptual difference is that in the autoregressive case the model is generative in a sequential, causal way — it predicts one step ahead at a time. In the latent variable case, the model posits a clean underlying dynamical system that generates all the observations, which is a much more natural fit for physical systems like orbital mechanics where you believe there really is a true underlying continuous trajectory being noisily observed.




In a latent variable model, you have observed data x and a latent variable z that you believe generated x. What you want is the posterior p(z | x) — that is, "given the data I observed, what is the distribution over latent states that could have produced it?" This tells you what the underlying state probably was.

The problem is that computing p(z | x) exactly requires evaluating p(z | x) = p(x | z) * p(z) / p(x), and the denominator p(x) requires integrating over all possible values of z, which is generally intractable for complex models.

So instead you introduce an approximate posterior q_phi(z | x), which is a simpler, tractable distribution (usually Gaussian) that you train to be as close as possible to the true posterior p(z | x). This is the "approximation" — you're not computing the true posterior, you're finding the best approximation to it within some family of distributions. The parameters phi are learned, typically via an encoder network that takes x as input and outputs the parameters of the Gaussian (a mean and variance).

In the context of the Latent ODE paper, the approximate posterior is over the initial latent state z_0. The encoder (which is actually an ODE-RNN running backwards over the observations) looks at all the data and produces a distribution q_phi(z_0 | x_0, ..., x_n). You then sample z_0 from that distribution, run it forward through the ODE, and decode. The training objective pushes q_phi to be close to the true posterior while also making the reconstructions accurate — this is the standard variational autoencoder (VAE) framework.




**Why don't we use the autoregressive method? Could we still propagate dynamics forward in time?**
Good question. In the autoregressive case you can propagate forward by feeding predictions back in as inputs — you predict x_i, treat it as an observation, use it to update the hidden state, then predict x_{i+1}, and so on. This is called autoregressive rollout.

The problem is that errors accumulate. Each prediction is slightly wrong, and when you feed that wrong prediction back in as if it were a real observation, the next prediction is conditioned on corrupted input. Over many steps the errors compound and the trajectory drifts. This is sometimes called exposure bias — during training the model always sees real observations, but at inference time it sees its own (imperfect) predictions.

This is actually one of the key motivations for the latent variable construction in the paper. In the Latent ODE, you infer z_0 from the data once, then let the ODE propagate it forward continuously. There's no feedback of noisy predictions — the ODE just integrates forward cleanly, and the decoder reads off observations wherever you want them. This makes long-horizon propagation much more stable, which for your astrodynamics application is probably very important since you'd want to propagate orbits over many revolutions.


The authors use an ODE-RNN in their latent ODE model to find an approximate posterior from which to sample the initial condition, yes?
Yes, exactly. The ODE-RNN is used as the encoder — it runs backwards over the observations and produces the parameters of the approximate posterior q_phi(z_0 | x_0, ..., x_n), which is typically a Gaussian with some mean and variance. You then sample an initial latent state z_0 from that distribution, run it forward through the Latent ODE, and decode the latent states into observations.

So the ODE-RNN in this context isn't doing the dynamics propagation — it's doing inference. It's looking at all the irregularly-sampled observations and summarizing them into a distribution over plausible initial conditions. The actual forward propagation is handled by a separate Neural ODE operating in latent space.

![[Pasted image 20260224230028.png]]
On the left (the encoder), the wavy lines between the GRU updates represent the ODE being integrated between observations. The waves look different from each other, but that's not because different ODEs are being used — it's because the hidden state is at a different point in state space each time, so the same vector field f_theta produces different trajectories depending on where you are. The ODE is the same function, it's just being evaluated along different parts of the state space with different time gaps.

Think of it like a river current — the same underlying flow field governs everything, but a leaf dropped at different points will trace out a different path because it starts in a different location.

The GRU boxes are then the observation update steps, where each incoming x_i corrects the hidden state. So the sequence runs right to left (backwards in time), alternating between ODE propagation and GRU updates, until you've processed all observations and output the mean mu and variance sigma of q(z_0 | x_0, ..., x_N).

On the right (the decoder), you sample z_0, then run a single forward ODE solve from t_0 to t_N using the same learned dynamics f, reading off latent states z_1, z_i, z_N at the observation times and decoding them to reconstructions x_hat.

So they're both ODEs with the same structural form, but they're learning different things. The encoder ODE is learning something about how to propagate an inference state backwards through irregular time gaps. The decoder ODE is learning the actual generative dynamics of the system. In your astrodynamics case, the decoder ODE is the one you'd care about — that's the one learning something like orbital mechanics.


Re: my confusion about the jumps and updates in the ODE-RNN:
Remember that in the encoder, the hidden state h is not the physical state of the system — it's an abstract summary of the observations seen so far (running backwards). So when a GRU update causes a big jump in h, that's not saying the physical system jumped — it's saying "I just received a new observation and I'm updating my belief about what z_0 probably was." The hidden state is an inference object, not a physical one.

So the sequence is: the ODE smoothly propagates h between observation times, then the GRU does a potentially large correction when a new observation arrives. The ODE is saying "given my current summary of the data, let me evolve it forward to the next time point." The GRU is saying "now that I've seen x_i, let me update my summary to account for this new information."

The "jumps" are fine because h doesn't need to be smooth or physically meaningful — it just needs to, by the end of the backwards pass, encode enough information for the final output to produce a good mu and sigma for q(z_0 | x_0, ..., x_N).

This is again very analogous to a Kalman filter. In Kalman filtering you also have a predict step (smooth ODE-like propagation) and an update step (potentially large correction when a measurement arrives). The measurement update can cause a big jump in your state estimate, and that's not a problem — it just means the new observation was informative.


Re: my confusion about NN *g* in the paper:
After the ODE-RNN has run backwards through all the observations, you're left with a single hidden state vector h — a fixed-size summary of all the data. But what you actually want is the parameters of a Gaussian distribution over z_0, namely a mean mu and a variance sigma. These live in a different space than h and have different requirements (sigma has to be positive, for instance).

So g is just a small neural network that takes that final hidden state h and maps it to mu and sigma. It's essentially a learned projection from "summary of all observations" to "parameters of the distribution over initial conditions." In practice it's often just a linear layer or a shallow MLP.

So the full pipeline is: the ODE-RNN processes all observations and produces h, then g reads h and outputs mu and sigma, then you sample z_0 from the resulting Gaussian N(mu, sigma), and then the decoder ODE propagates z_0 forward to generate the trajectory. g is the bridge between the inference network and the generative model.


**"We jointly train the encoder and decoder by maximizing the evidence lower bound (ELBO):**
$ELBO(theta, phi) = E_{z0 ~ q_phi(z0 | {xi, ti})} [log p_theta(x0, ..., xN)] - KL[q_phi(z0 | {xi, ti}) || p(z0)]$
The first term, E[log p_theta(x_0, ..., x_N)], is the reconstruction term. You sample a z_0 from your approximate posterior q_phi, run it forward through the decoder ODE, decode the latent states into reconstructed observations x_hat, and then measure how well those reconstructions match the actual observations. The expectation means you do this averaging over samples of z_0. This term is pushing the model to actually explain the data well.

The second term, KL[q_phi(z_0 | {x_i, t_i}) || p(z_0)], is the regularization term. KL divergence is a measure of how different two distributions are — here it's measuring how far your approximate posterior q_phi(z_0 | data) is from the prior p(z_0), which is typically just a standard Gaussian N(0, I). This term penalizes the encoder for producing a posterior that strays too far from the prior. It stops the model from just memorizing the data by encoding everything into z_0.

The two terms are in tension. The reconstruction term wants the encoder to produce a very specific z_0 that explains the data perfectly. The KL term wants the posterior to stay close to the prior and not be too confident or specific. The ELBO balances these two pressures, and maximizing it trains all the parameters — the encoder phi (the ODE-RNN and g) and the decoder theta (the decoder ODE and the observation model) — jointly end to end.

The decoder ODE is the Neural ODE that learns the dynamics in latent space. It takes z_0 as an initial condition and integrates forward using a learned vector field f_theta to produce z_1, z_i, ..., z_N at whatever time points you want. This is the heart of the model — it's the part that's learning a continuous, smooth dynamical system in latent space that generates the data.

In your astrodynamics context, this is the part that would be learning something analogous to orbital mechanics, just operating in a learned latent space rather than directly in Cartesian coordinates.