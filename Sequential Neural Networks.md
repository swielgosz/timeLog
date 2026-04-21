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
- 
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

Suggestions from John:
be able to speak to transformers - don't need to process sequentially
do presentation at high level first. don't need full derivations
good for datasets with uncertainty - used for RL to construct datasets with some belief. connect to estimation and kalman filtering