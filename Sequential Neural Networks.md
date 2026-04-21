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
Loss depends on the task performed by the model. For example, if we are just predicting one label for a given text then we wouldn't need $L^{<t-1>}$ or $L^{
![[Pasted image 20260421110035.png]]
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
# Seq2Seq
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

# Other
[Autoencoder vs VAE](https://towardsdatascience.com/difference-between-autoencoder-ae-and-variational-autoencoder-vae-ed7be1c038f2/)

Suggestions from John:
be able to speak to transformers - don't need to process sequentially
do presentation at high level first. don't need full derivations
good for datasets with uncertainty - used for RL to construct datasets with some belief. connect to estimation and kalman filtering