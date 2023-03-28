#!/usr/bin/env bash

# call code to train tiny_imagenet
call_train() {
    echo "$1"
    sbatch -p ctn --qos=high-priority ./train_imagenet.sh "$1"
}

# ------ set outputdirectory ------
OUTPUTDIR="/share/ctn/users/ekb2154/data/libraries/interp_ensembles/results/pl_imagenet"
mkdir -p $OUTPUTDIR
now=$(date +"%m_%d_%Y/%H_%M_%S")
echo "$now"
# ---------------------------------

dataset_dir="${HOME}/pytorch_datasets/imagenet"

if [ "$(uname)" = "Darwin" ] ; then
config_name="run_default_cpu_imagenet"
num_workers=4

elif [ "$(uname)" = "Linux" ]; then
config_name="run_default_gpu_imagenet"
num_workers=56
fi

logger="wandb"

module="ensemble_dkl_imagenet"
classifier="resnet18"
label_smoothing=0
gamma=1.0
seed=0
batch_size=256

for seed in 0;
do
call_train "--config-name="${config_name}" \
  data_dir=${dataset_dir}  \
  logger=${logger} \
  num_workers=${num_workers} \
  module=${module} \
  label_smoothing=${label_smoothing} \
  gamma=${gamma} \
  seed=${seed} \
  classifier=${classifier} \
  batch_size=${batch_size} \

  "


done

