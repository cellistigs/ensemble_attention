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

 
