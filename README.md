# Repo for project with Kelly, Geoff, Rich Zemel and JPC studying a connection between ensemble models and attention. 

## Short summary of ideas (fromm google docs notes): 
Ensembles are somehow more parameter efficient than a standard single model when it comes to performance. Are there models that could be even better than ensembles? If we think about weaknesses of ensembles, we are training for the average case performance, in a highly redundant way. Are there ways in which we can take this "parameter efficient" aspect of ensembling and further improve upon it, or access it in a single network? 

- As an improvement over ensembles via averaging, consider something like mixture of experts. Would we expect this to do better? We are no longer training/for redundancy, but explicitly requiring decorrelation. 
- As a potentially related example back in the realm of single models, consider models with attention mechanisms. 
Is there a relationship between these two classes of models? Something about specialization, and having diversity within the model as a function of the data seems similar between these.  

Further remarks:
- Mixture of experts are coarse, because we average at the last layer, but attention is not. Can we think about attention as a dynamic ensemble? If there is such a framing here, that would be a cool perspective. 
- Consider three classes of model; 
  - Attention networks
  - "Static" ensembles, as we've studied so far
  - “Dynamic” ensembles, where our aggregation strategy is based on the data 
  - How do these compare?
- Another way to interpret mixture of expert like characteristics would be through hyperensembles: https://arxiv.org/abs/1609.09106

# Repo organization: 
- src: source code. 
- scripts: scripts that run against source code. 
- configs: configuration files that determine parameters used to run scripts (hydra format yaml). 
- refs: literature and references. 

# Requirements: 
You will have to install [this package](https://github.com/cellistigs/cifar10_ood) from source to get this to work. 
