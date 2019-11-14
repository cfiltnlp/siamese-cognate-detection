# Siamese-Feed-Forword-Network-for-Cognate-Detection

The repository contains the Siamese FeedFoward implementation for the NLP application of True Cogante Detection. It can be extended for various similarity measure tasks for textual input.The code is implemented using PyTorch framework and uses the FastText Word Embeddings (P. Bojanowski,et al. Enriching Word Vectors with Subword Information) for Word Vector Representation.

## Siamese FeedForward Architecture
![alt text](https://github.com/SravanSai10/Siamese-Feed-Forword-Network-for-Cognate-Detection/blob/master/Siamese.png)

Siamese Neural Networks tries to learn a common subspace, by tying the trainable parameters of the Network.It uses the same weights while working in tandem on two different input vectors to compute comparable output vectors. The output vectors are compared using similarity scores like cosine similarity, Manhattan or Euclidean distances. 
The above architecture is designed for the True Cognat



