# Timeline
Hopfield-> Jordean/Elman->RNN -> LSTM -> GRU -> Transformers

# Motivation
## Seabstian Raschka Intro to Deep Learning:
https://www.youtube.com/watch?v=q5YxK17tRm0&list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51&index=126
![[L15_intro-rnn__slides.pdf]]

### Sequence Modeling with RNNs
How do we know if our model uses sequence information? Logistic regression, MLPs do not. There are two types of sequential data that can exist in training data: 1. across the training example axis, 2. across the feature axis. 

Iris example: sepal length, sepal width, petal length, petal width.

|     | SL  | SW  | PL  | PW  |
| --- | --- | --- | --- | --- |
| 1   |     |     |     |     |
| 2   |     |     |     |     |
| ..  |     |     |     |     |
| 150 |     |     |     |     |
We split dataset into training and test set. We shuffle the dataset initially. If we evaluate the model on the test set and we shuffle the test set, performance should not be affected. This is an indication that there is no sequence information.
The data is i.i.d. - independent and identically distributed. Each training sample is independent of each other and from the same distribution. So, we can do a thought experiment - if we shuffle the test set, will it affect our performance? Another sequence we have is in the order of our columns. If we swap the columns, will model performance be affected? If no, then we do not have sequential data. In general, MLPs and logistic regression don't use sequence info across features. 

In sequence data order matters:
- The movie my friend has not seen is good
- The movie my friend has seen is not good
Bag of words model does not work in this case.
![[Pasted image 20260421102455.png]]

Applications:
- Text classification: we have time dimension over the words, and each document containing a sequence of words is a training example
- Speech recognition (acoustic modeling) is a sequence of sounds
- Language translation translates from one sequence to another
- Stock market prediction - each stock has multiple associated sequenced feature vectors - price, news, etc. This could be associated with a time dimension. Each stock is a different sample
- DNA sequence modeling - sequenced by character

Overview:
![[Pasted image 20260421103002.png]]
We give a feature at time step t to the hidden state, which results in an output at timestep t. Additionally, the hidden layer(s) are receiving information from the previous time steps

Here is the unfolded single layer RNN:
![[Pasted image 20260421103304.png]]
For example, look at time step t. It receives the feature vector, indicated by $x^{<t>}$ . It also receives input from the previous time step. So, we receive info from the feature vector at time t, as well as the hidden state of the previous time step t-1. So, the network is aware of the sequence. Left and right are equivalent representations 

The same concept applies to multilayer RNNs:
![[Pasted image 20260421103645.png]]

## Different Types of Sequence Modeling Tasks
![[Pasted image 20260421105137.png]]
Many to one - written review, outcome is whether it is positive or negative
One to many (use together with CNN) - put in an image (someone playing tennis), output is a sequence of text describing the image
Many-to-many - direct: video captioning to describe a sequence of images; delayed - translating between languages (you can't just translate word by word, you need the context of the entire sentence)

## Backpropagation Through Time
Weight matrices in single hidden layer RNN:
![[Pasted image 20260421105925.png]]
You would find $W_{hx}$ and $W_{yh}$ in a regular MLP. 
In the RNN, we also have $W_{hh}$. So, for an RNN we have two matrices for the hidden layer ($W_h$). We have $W_{hx}$ connecting the input to the hidden layer, and $W_{hh}$ connecting the previous hidden layer to the current hidden layer. We use the same matrices at each time step. So compared to the MLP, $W_{hh}$ is what's new.

Computing the loss:
![[Pasted image 20260421111727.png]]
Loss depends on the task performed by the model. For example, if we are just predicting one label for a given text then we technically wouldn't need $L^{<t-1>}$ or $L^{<t>}$. Some argue that you should still keep them and it helps with training the earlier layers. So some keep, some don't in many to one. In many to many, you keep multiple losses and just sum them up as shown. 

Issues with RNN:
![[Pasted image 20260421110035.png]]
The first two terms on the RHS are what we would have for a normal MLP, the sum term is for the other terms. 
This is problematic:
![[Pasted image 20260421110208.png]]

---

## Andrew Ng:
https://www.youtube.com/watch?v=S7oA5C43Rbc

### Why Sequence Models
Examples of supervised problems with input x and output y. Some have x and y as sequences, some only x, some only y. :
- Speech recognition: audio input x (sequence) -> text output y (sequenced)
- Music generation: integer (not sequenced, maybe corresponds to genre) -> music note output sequence
- Sentiment classification: e.g. "there is nothing to like in this movie" -> one syar review
- DNA sequence analysis: input sequence -> output portion corresponding to a protein
- Machine translation: french -> english
- video activity recognition: given sequence of frames -> output activity (e.g. running)
- Name entity recognition: given sentence -> identify people
- ---

# Relevance to RL

# Relevance to filtering

# Temporal Convolutional Networks
https://www.youtube.com/watch?v=rT77lBfAZm4
What is a convolution? How is it different than regression?
CNN: https://www.youtube.com/watch?v=QzY57FaENXg
Feedforward vs CNN vs RNN: https://www.youtube.com/watch?v=u7obuspdQu4
# RNN
https://www.youtube.com/watch?v=AsNTP8Kwu80&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=15
https://www.youtube.com/watch?v=LHXXI4-IEns

Foundational paper: "Finding Structure in Time" - Elman, 1990.
# LSTM
https://www.youtube.com/watch?v=YCzL96nL7j0&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=16
https://www.youtube.com/watch?v=8HyCNIVRbSU
Foundational paper: “Long Short-Term Memory” - Hochreiter & Schmidhuber, 1997.
# GRU
“Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation” — Cho et al., 2014.
https://www.youtube.com/watch?v=IBs8D8PWMc8
https://www.youtube.com/watch?v=8HyCNIVRbSU
# Seq2Seq - Don't focus on this
https://www.youtube.com/watch?v=L8HKweZIOmg&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=18
“Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation” — Cho et al., 2014
“Sequence to Sequence Learning with Neural Networks” — Sutskever, Vinyals & Le, 2014.

# ODE-RNN (Latent ODE)
“Latent ODEs for Irregularly-Sampled Time Series”
Bonus: 
- [Framing RNN as a kernel method: A neural ODE approach](https://www.youtube.com/watch?v=2_MF2LX9Q5E)
- [ MIA Special Seminar: David Duvenaud, It's time to talk about irregularly-sampled time series](https://www.youtube.com/watch?v=iB2d99K_vk8)
- [Latent Stochastic Differential Equations for Irregularly-Sampled Time Series - David Duvenaud](https://www.youtube.com/watch?v=93wr5vhI0F4)
# Transformers
https://www.youtube.com/watch?v=PSs6nxngL6k&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=19
https://www.youtube.com/watch?v=zxQyTK8quyY&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=20
[Harvard NLP: Annotated "Attention Is All You Need](https://nlp.seas.harvard.edu/annotated-transformer/)
[The Illustrated Transformer (more simplified than above)](https://jalammar.github.io/illustrated-transformer/)
attention = the core computational tool
transformer = the machine built around that tool
LLM = a very large, trained instance of such a machine for language

https://deeprevision.github.io/posts/001-transformer/
![[Pasted image 20260421194059.png]]
# Other
[Autoencoder vs VAE](https://towardsdatascience.com/difference-between-autoencoder-ae-and-variational-autoencoder-vae-ed7be1c038f2/)

# Visualizations
## RNN
![[Pasted image 20260421185526.png]]
Types of tasks:
![[Pasted image 20260421185655.png]]
Single layer RNN:
![[Pasted image 20260421185600.png]]
Multilayer RNN:
![[Pasted image 20260421185612.png]]
![[Pasted image 20260421192324.png]]
## LSTM
![[Pasted image 20260421185956.png]]
## GRU
![[Pasted image 20260421185810.png]]

LSTM v GRU: ![[Pasted image 20260421191316.png]]

## TCN
![[Pasted image 20260421192432.png]]
![[Pasted image 20260421192521.png]]
![[Pasted image 20260421192608.png]]
![[Pasted image 20260421192634.png]]

## Transformer
![[Pasted image 20260421194118.png]]
![[Pasted image 20260421194701.png]]

Suggestions from John:
be able to speak to transformers - don't need to process sequentially
do presentation at high level first. don't need full derivations
good for datasets with uncertainty - used for RL to construct datasets with some belief. connect to estimation and kalman filtering

# Notes
Idea behind LSTM and GRU is that a vanilla RNN repeatedly applied a nonlinear transformation. When you back propagate through many time steps, gradients will shrink or blow up. LSTMs, and subsequently GRUs, were introduced to create a path where information (and gradients) can flow more directly, with gates that learn when to keep, update, or forget information. 

LSTM:
LSTM has a separate memory cell (the cell state) that evolves mostly linearly, which allows gradients to flow backward through time wiithout repeatedly passing through squashing nonlinearities.

Gates are learned switches. We have forget gate, input gate, output gate. Sigmoid outputs between 0 and 1 - this is like an on/off switch for memory. Since the network learns to preserve or erase information, the memory is adaptive instead of fixed.

Updates are additive instead of purely multiplicative (for cell updates specifically). Additive structure avoids repeated multiplication and reduces gradient decay.

GRU:
No separate cell state - we combine hidden state and memory into one. This is a simpler architecture, fewer parameters, trains faster. Update gate is like forget and input combined. It decides whether to keep the old state or replace with new information. LSTM has two gates doing related jobs, while GRU merges them

Reset gate allows 


- Backpropagation through time
- mechanics of the training of RNN as a baseline
- LSTM 
- Methods sections - explanations that are commonly in ML papers
- ask claude for latex equations for pedagogical equations for the topics of interest
- use text prediction and trajectory for motivating examples
- type up the outline in bullets and send to John
- connect to neuralODEs - these are generalizations of RNNs(?)
- we can use these for hidden states -> use these to encode a latent state. VAE is a MLP historically, but now we use time sequence
- how do we represent latent state - is output of a gRU always a latent disrtibtuion?

# Outline
**Motivation**
- What is sequential data? Give examples:
	- Text (sentence is a sequence of words, word is a sequence of letters)
	- Audio
	- Spacecraft in orbit - trajectory, sensor measurements 
- General types of problems we can solve with sequential NNs
	- Many-to-one - sentiment classification; "this is my favorite movie of all time" --> 5 stars
	- One-to-many - description of an image
	- Many-to-many -  machine health monitoring; output signals --> health status
	- Many-to-many (delayed) - translating languages (you the whole sentence before translating, you can't just translate word-by-word)
- Timeline of sequential NNs and following models
	- RNN (1990) --> LSTM (1997) --> GRU (2014); Seq2Seq (2014); Attention (2014). --> Transformer (2017)
		- RNNs introduce sequential data processing, but have vanishing gradients
		- LSTMs mitigate the vanishing gradient problem in RNNs by using input, output, and forget gates to regulate the flow of information and gradients through the network over time
		- GRUs simplifies this by using fewer gates than LSTM, making it more compact but still helping preserve information over long sequences
		- Seq2seq models show that a deep LSTM encoder-decoder can map variable-length input sequences to variable-length output sequences in an end-to-end framework
		- Attention introduces a mechanism that lets the model align each output step with the most relevant parts of the input sequence
		- Transformers build on attention by using self-attention to model relationships across an entire sequence without recurrence, enabling more effective long-range modeling and parallel training
RNNs
- How MLP and RNNs differ - MLPs process inputs independently, while RNNs are designed for sequential data by updating the hidden states at each sequence