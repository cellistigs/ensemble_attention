#!/usr/bin/env bash

# call code to train imagenet
call_train() {
    echo "$1"
    sbatch -p ctn --qos=high-priority ./train_cifar10.sh "$1"
}

# ------ set outputdirectory ------
OUTPUTDIR="/share/ctn/users/ekb2154/data/libraries/interp_ensembles/results/pl_cifar10"
mkdir -p $OUTPUTDIR
now=$(date +"%m_%d_%Y/%H_%M_%S")
echo "$now"
# ---------------------------------

dataset_dir="${HOME}/pytorch_datasets/cifar10_ood/data"

if [ "$(uname)" = "Darwin" ] ; then
config_name="run_default_cpu"
num_workers=4

elif [ "$(uname)" = "Linux" ]; then
config_name="run_default_gpu"
num_workers=4
fi

max_epochs=100
logger="wandb"
label_smoothing=0.1

module="ensemble_jgap"

for seed in 0 1 2 3 4 
do
for label_smoothing in {0.0,0.1};
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

  "
done
done
done
