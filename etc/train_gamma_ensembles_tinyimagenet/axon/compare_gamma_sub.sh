#!/usr/bin/env bash

# call code to train tiny_imagenet
call_train() {
    echo "$1"
    sbatch -p ctn --qos=high-priority ./train_tinyimagenet.sh "$1"
}

# ------ set outputdirectory ------
OUTPUTDIR="/share/ctn/users/ekb2154/data/libraries/interp_ensembles/results/pl_tinyimagenet"
mkdir -p $OUTPUTDIR
now=$(date +"%m_%d_%Y/%H_%M_%S")
echo "$now"
# ---------------------------------

dataset_dir="${HOME}/pytorch_datasets/tiny-imagenet-200"

if [ "$(uname)" = "Darwin" ] ; then
config_name="run_default_cpu_tinyimagenet"
num_workers=4

elif [ "$(uname)" = "Linux" ]; then
config_name="run_default_gpu_tinyimagenet"
num_workers=4
fi

max_epochs=100
logger="wandb"

module="ensemble_jgap"
classifier="resnet18"
label_smoothing=0


for classifier in "resnet18"
do
for seed in 0
do
for gamma in {0.0,0.2,0.5,1.0,1.2,1.5,2.0};
do
call_train "--config-name="${config_name}" \
  data_dir=${dataset_dir}  \
  max_epochs=${max_epochs} \
  logger=${logger} \
  num_workers=${num_workers} \
  module=${module} \
  label_smoothing=${label_smoothing} \
  gamma=${gamma} \
  seed=${seed} \
  classifier=${classifier} \

  "
  sleep 1m # sleep for 1 minute
done
done
done
