#!/bin/bash

# Train gamma ensemble models on cifar 10 for multiple values of gamma.

max_epochs=1
logger="wandb"
module="ensemble_jgap"
seed=0

dataset_dir="${HOME}/pytorch_datasets/cifar10_ood/data"

if [ "$(uname)" = "Darwin" ] ; then
config_name="run_default_cpu"
num_workers=4

elif [ "$(uname)" = "Linux" ]; then
config_name="run_default_gpu"
num_workers=16
fi

pushd ../../

python scripts/run.py \
  --config-name="${config_name}" \
  data_dir=${dataset_dir}  \
  max_epochs=${max_epochs} \
  logger=${logger} \
  num_workers=${num_workers} \
  module=${module} \
  seed=${seed} \

popd