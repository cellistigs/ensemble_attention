# Annotated bibliography 

## Previous work relating ensembles and attention
- Kim et al. 2018l "Attention-based ensemble for deep metric learning", arxiv 2018. [link](https://arxiv.org/abs/1804.00382)

## Training ensembles for improved performance/Ensemble and single model hybrids 

- Lee et al. 2016; "Stochastic Multiple Choice Learning for Training Diverse Deep Ensembles", arxiv 2016. If you have an oracle that can correct multiple competing hypotheses downstream, it can be a good idea to learn multiple likely outcomes instead of a single one. They introduce a loss, stochastic Multiple Choice Learning (sMCL) in which one considers an ensemble of models, and trains them together, but only propagates the error to the model that currently has the lowest loss on any given example. Does better than classical ensembles with oracle evaluation. 
- Wen et al. 2020; "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning", arxiv 2020. Ensembles are expensive, and it's hard to decouple their effectiveness (probably diversity) from their effectiveness. This paper suggests an approach in which one constructs an ensemble by hadamard-multiplying a base set of weights with a set of N different rank 1 matrices, and training the result with gradient descent. This is a different combination of a base network with a structured perturbation to what we propose.
- Warde-Farley et al. 2014: "An empirical analysis of dropout in piecewise linear networks". arxiv 2014. [link](https://arxiv.org/pdf/1312.6197.pdf). This paper analyzes the interpretation of dropout in ReLU networks as creating an exponentially large ensemble of networks that share parameters. The relevant component of their work for this section is an alternaltive loss that they introduce- "dropout boosting", wherein they update parameters only for the subnetwork that is active at the moment, but evaluate the entire ensemble loss instead of the subnet loss. This is "boosting" in the sense that we are forming "a direct assault on the representational problem" and asking the network to fit the ensemble cost. Definitely an interesting citation and one for us to consider in analyzing the interpolating ensemble/mean field ensemble. Note however, that in this context the authors saw that this cost did worse than dropout, and only as well as the standard network trained with SGD. We see that our models do no differently than ensembling.  
- TOREAD Zhang et al. 2018; "Deep Mutual Learning", CVPR 2018. [link](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Deep_Mutual_Learning_CVPR_2018_paper.pdf). Regularizes each ensemble member output more towards the others. 
- TOREAD Hinton 2002; "Training Products of Experts by Minimizing Contrastive Divergence", Neural Computation. [link](https://www.cs.toronto.edu/~hinton/absps/nccd.pdf)
- TOREAD Havasi et al. 2021; "Training independent subnetworks for robust prediction", ICLR 2021.
- TOREAD Huang et al. 2016; "Deep Networks with Stochastic Depth", ECCV 2016. An alternative to wide networks is to use blockwise dropout- this is like implicitly training an ennsemble of networks too. 

## Attention

- Finn 2021; "Self-Attentive Ensemble Transformer: Representing Ensemble Interactions in Neural Networks for Earth System Models; arxiv, [link]("https://arxiv.org/pdf/2106.13924.pdf). Draws connections between self attention ensembles and particle sampling for modeling dynamical earth system models. 
- Vaswani et al. 2017; "Attention is All You Need", arxiv, [link](https://arxiv.org/abs/1706.03762)
  - What are the potential sources of "diversity" within a transformer model?  
    - One source of diversity comes from the attention mechanism itself. Any time that we are working with one part of the input at a time, but we expect that we can do better conditioned on the whole thing, we can consider an attention mechanism. Considering the important aspect of the input as the query, and all others as the keys and values. This system will give you the most useful average value for each key. 
      - Consider the following way to implement scaled dot-product attention as an aggregation mechanism in a deep ensemble: 
        1. initialize four independent classification nns. 
        2. extract the feature representations that they generate in their second to last layer, and hit them with 3 trainable projections to turn them into queries, keys and values. 
        3. treating each network output as a query, calculate the compatibility function with each of the other ensemble member outputs.
        4. Use compatibility functions to get the weighted outputs of each network.  
        5. Pass through remaining computation in each network.
        6. Generate ensemble output through a) averaging, b) voting, c) most confident, d) training another readout layer off concatenated outputs, e) arbitrarily choosing one.  
        Remark 1. Note that this method has nearly the same number of parameters as standard ensembling, and self-attention merely provides a coupling mechanism between different parts of the same network. Standard ensembling corresponds to the case where the query projection is equal to the key projection.
        Remark 2. It seems like you could be losing some diversity by averaging the ensemble member outputs- the other aggregation methods may be better. 
        Remark 3. This process can be iterated, and might actually be more useful to have at lower layers, given insights from treenet (Lee et al. 2015) and others. 
        Remark 4. This formulation treats the different feature detectors learned by each network as independent "positions", which should contextualize each other. It may be worthwhile to explicitly encourage more diversity, i.e. by taking input augmentations, or something like that. 
    - Another source of diversity comes from the fact that we have "multi-head" attention. Each key, value, and query are not ingested raw by the attention mechanism, but are randomly projected with different trainable projections. This process not only distinguishes the same token in key, query, and value roles, but also can be repeated to generate multiple separate attention heads, each of which generates different values for a given query. These individual head outputs can be combined together later on with an arbitrary learned matrix. 
      - We can extend our idea above to this context by also considering N different sets of query, key and value projections, and sharing them all between networks. Having multiple heads might also form another incentive to diversify outputs.  
    - We can further increase diversity by asking each ensemble member to predict on a different output space. We can imagine different things here, like randomly combining input classes, adding per-instance label noise, or other fun interventions. 
  - Remark 1. This is an interesting and potentially interpretable model, that can be compared to deep ensembles pretty directly. 
  - Remark 2. This method looks quite similar to mixture of experts as well: having a network that selects from input classifiers is the fundamental idea of mixture of experts.  
  - Remark 3. If we perform this self attention at each layer, we introduce an interesting computation into a big network model: we are now asking how the activations in different channels can inform each other. This might be useful for a variety of reasons, like getting conv information from across the model integrated faster.  

- TOREAD Bakkali et al. 2021; "EAML: ensemble self attention based mutual learning network for document image classification" [link](https://link.springer.com/article/10.1007/s10032-021-00378-0), cross-modality ensembles. 
- TOREAD Dosovitskiy et al. 2020; "An Image is Worth 16x16 Words", arxiv, [link](https://arxiv.org/abs/2010.11929), vision transformers
- TOREAD Gregor et al. 2015: "DRAW: A Recurrent Neural Network for Image Generation", arxiv, [link](https://arxiv.org/pdf/1502.04623.pdf), attention used in generative modeling
- TOREAD Cheng et al. 2016: "Long Short-Term Memory Networks for Machine Reading", arxiv, [link](https://arxiv.org/abs/1601.06733), self attention
- TOREAD Xu et al. 2015: "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention", arxiv [link](https://arxiv.org/abs/1502.03044) soft vs. hard attention in image models. 
