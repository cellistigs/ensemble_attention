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

logger="wandb"
max_epochs=100
module="casregress_ensemble_jgap_onehot"
classifier="wideresnet28_10"
learning_rate=1e-1 #1e-2
weight_decay=1e-2 #1e-2
batch_size=64 #256

for classifier in "wideresnet28_10"
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
  gamma=${gamma} \
  seed=${seed} \
  classifier=${classifier} \
  learning_rate=${learning_rate} \
  weight_decay=${weight_decay} \
  batch_size=${batch_size} \

  "
done
done
done
