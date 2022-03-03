# Annotated bibliography 

## Training ensembles for improved performance/Ensemble and single model hybrids 

- Lee et al. 2016; "Stochastic Multiple Choice Learning for Training Diverse Deep Ensembles", arxiv 2016. If you have an oracle that can correct multiple competing hypotheses downstream, it can be a good idea to learn multiple likely outcomes instead of a single one. They introduce a loss, stochastic Multiple Choice Learning (sMCL) in which one considers an ensemble of models, and trains them together, but only propagates the error to the model that currently has the lowest loss on any given example. Does better than classical ensembles with oracle evaluation. 
- Wen et al. 2020; "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning", arxiv 2020. Ensembles are expensive, and it's hard to decouple their effectiveness (probably diversity) from their effectiveness. This paper suggests an approach in which one constructs an ensemble by hadamard-multiplying a base set of weights with a set of N different rank 1 matrices, and training the result with gradient descent. This is a different combination of a base network with a structured perturbation to what we propose.
- Warde-Farley et al. 2014: "An empirical analysis of dropout in piecewise linear networks". arxiv 2014. [link](https://arxiv.org/pdf/1312.6197.pdf). This paper analyzes the interpretation of dropout in ReLU networks as creating an exponentially large ensemble of networks that share parameters. The relevant component of their work for this section is an alternaltive loss that they introduce- "dropout boosting", wherein they update parameters only for the subnetwork that is active at the moment, but evaluate the entire ensemble loss instead of the subnet loss. This is "boosting" in the sense that we are forming "a direct assault on the representational problem" and asking the network to fit the ensemble cost. Definitely an interesting citation and one for us to consider in analyzing the interpolating ensemble/mean field ensemble. Note however, that in this context the authors saw that this cost did worse than dropout, and only as well as the standard network trained with SGD. We see that our models do no differently than ensembling.  
- TOREAD Hinton 2002; "Training Products of Experts by Minimizing Contrastive Divergence", Neural Computation. [link](https://www.cs.toronto.edu/~hinton/absps/nccd.pdf)
- TOREAD Havasi et al. 2021; "Training independent subnetworks for robust prediction", ICLR 2021.
- TOREAD Huang et al. 2016; "Deep Networks with Stochastic Depth", ECCV 2016. An alternative to wide networks is to use blockwise dropout- this is like implicitly training an ennsemble of networks too. 

## Attention 

- TOREAD Vaswani et al. 2017; "Attention is All You Need", arxiv, [link](https://arxiv.org/abs/1706.03762), redefining attention very broadly
- TOREAD Dosovitskiy et al. 2020; "An Image is Worth 16x16 Words", arxiv, [link](https://arxiv.org/abs/2010.11929), vision transformers
- TOREAD Gregor et al. 2015: "DRAW: A Recurrent Neural Network for Image Generation", arxiv, [link](https://arxiv.org/pdf/1502.04623.pdf), attention used in generative modeling
- TOREAD Cheng et al. 2016: "Long Short-Term Memory Networks for Machine Reading", arxiv, [link](https://arxiv.org/abs/1601.06733), self attention
- TOREAD Xu et al. 2015: "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention", arxiv [link](https://arxiv.org/abs/1502.03044) soft vs. hard attention in image models. 
