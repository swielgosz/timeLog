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

"Autoregressive models make a one-step-ahead prediction conditioned on the history of observations, i.e. they factor the joint density p(x) = ∏  i pθ(xi|xi−1, . . . , x0). As in standard RNNs, we can use  an ODE-RNN to specify the conditional distributions pθ(xi|xi−1...x0) (Algorithm 1)."