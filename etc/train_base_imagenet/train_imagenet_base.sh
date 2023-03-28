#!/bin/bash

# Train resnet18 on imagenet

logger="wandb"
seed=0

dataset_dir="${HOME}/pytorch_datasets/imagenet"

if [ "$(uname)" = "Darwin" ] ; then
config_name="run_default_cpu_imagenet"
num_workers=4
accelerator=0
gpus=-1

elif [ "$(uname)" = "Linux" ]; then
config_name="run_default_gpu_imagenet"
#config_name="run_default_gpu"
num_workers=16

fi

# in lion overwrite the default config
if [ "$(hostname)" = "ekb-lp" ]; then
  echo "In Lion"
  gpus=1
  accelerator="dp"

fi


pretrained=0
module="base_imagenet"
max_epochs=2 # 100
classifier="resnet18"

pushd ../../

python scripts/run_imagenet.py \
  --config-name="${config_name}" \
  data_dir=${dataset_dir}  \
  max_epochs=${max_epochs} \
  logger=${logger} \
  num_workers=${num_workers} \
  module=${module} \
  seed=${seed} \
  pretrained=${pretrained} \
  classifier=${classifier} \
  gpus=${gpus} \
  accelerator=${accelerator} \

popd