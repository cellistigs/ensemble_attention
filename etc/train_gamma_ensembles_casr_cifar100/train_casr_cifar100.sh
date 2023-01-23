#!/bin/bash

# Train resnet18 on tiny imagenet model

#logger="wandb"
logger="wandb"
seed=0

dataset_dir="${HOME}/pytorch_datasets/cifar-100-python"
#dataset_dir="${HOME}/pytorch_datasets/cifar10_ood/data"
#dataset_dir="/data/Projects/linear_ensembles/cifar10_ood/data"

if [ "$(uname)" = "Darwin" ] ; then
exit
num_workers=4

elif [ "$(uname)" = "Linux" ]; then
config_name="run_gpu_cifar100"
#config_name="run_default_gpu"

num_workers=16

fi

pretrained=0
module="casregress_onehot"
max_epochs=2
classifier="wideresnet28_10"
resnet_stride=2
learning_rate=1e-2
weight_decay=1e-2
pushd ../../

python scripts/run.py \
  --config-name="${config_name}" \
  data_dir=${dataset_dir}  \
  logger=${logger} \
  num_workers=${num_workers} \
  classifier=${classifier} \
  module=${module} \
  max_epochs=${max_epochs} \
  # seed=${seed} \
  # pretrained=${pretrained} \
  # resnet_stride=${resnet_stride} \
  # learning_rate=%{learning_rate} \

popd