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