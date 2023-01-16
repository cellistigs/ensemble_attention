#!/bin/bash

# Train gamma ensemble models on cifar 10 for multiple values of gamma.

#logger="wandb"
logger="tensorboard"
#module="casregress_onehot"
module="casregress_ensemble_onehot"
seed=0
classifier="wideresnet28_10"
learning_rate=1e-1 #1e-2
weight_decay=1e-2 #1e-2
batch_size=64 #256
max_epochs=100

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
  classifier=${classifier} \
  learning_rate=${learning_rate} \
  weight_decay=${weight_decay} \
  batch_size=${batch_size} \

popd