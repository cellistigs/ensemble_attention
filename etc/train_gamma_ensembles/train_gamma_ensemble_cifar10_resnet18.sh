#!/bin/bash

# Train gamma ensemble models on cifar 10 for multiple values of gamma.

dataset_dir="/home/ekellbuch/pytorch_datasets/cifar10_ood/data"
max_epochs=100
logger="wandb"


#   --deterministic=${deterministic} \
python /data/Projects/linear_ensembles/ensemble_attention/scripts/run.py \
  --config-name="run_default_gpu" \
  --multirun gamma=0.1,1.0,1.5 \
  data_dir=${dataset_dir}  \
  max_epochs=${max_epochs} \
  logger=${logger} \
  num_workers=1 \
