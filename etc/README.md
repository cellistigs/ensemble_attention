# Train gamma ensemble

- Train lambda ensemble model and store logits and trained models.
```
# code to use without wandb
train_gamma_ensembles/train_gamma_ensemble_cifar10_resnet18.sh
# code to use with wandb
cd train_gamma_ensembles
wandb sweep train_gamma_ensemble_cifar10_resnet18.yaml
```
- Visualize multiple lambda ensemble models.
```
python scripts/vis_scripts/kl_weight_resnet18_cifar.py
```
- Grab lambda models, and check whether they are over confident or not by comparing ECE
an overconfident model should have low ece a and under confident model should have high ECE.
- ECE w/o for different gamma 

Experiments: <br>
[x] train an ensemble_jgap resnet 18 with different levels of label smoothing and gamma for different seeds.
axon/compare_ls_gamma_sub.sh
