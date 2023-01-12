#!/bin/bash

# Train resnet18 on tiny imagenet model

logger="wandb"
seed=0

dataset_dir="${HOME}/pytorch_datasets/tiny-imagenet-200"
#dataset_dir="${HOME}/pytorch_datasets/cifar10_ood/data"

if [ "$(uname)" = "Darwin" ] ; then
config_name="run_default_cpu_tinyimagenet"
num_workers=4

elif [ "$(uname)" = "Linux" ]; then
config_name="run_default_gpu_tinyimagenet"
#config_name="run_default_gpu"

num_workers=16

fi

pretrained=0
module="base"
max_epochs=100

pushd ../../

python scripts/run_tinyimagenet.py \
  --config-name="${config_name}" \
  data_dir=${dataset_dir}  \
  max_epochs=${max_epochs} \
  logger=${logger} \
  num_workers=${num_workers} \
  module=${module} \
  seed=${seed} \
  pretrained=${pretrained} \

popd