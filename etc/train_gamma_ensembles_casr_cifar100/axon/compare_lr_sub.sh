#!/usr/bin/env bash

# call code to train imagenet
call_train() {
    echo "$1"
    sbatch -p ctn --qos=high-priority ./train_cifar100.sh "$1"
}

# ------ set outputdirectory ------
OUTPUTDIR="/share/ctn/users/ekb2154/data/libraries/interp_ensembles/results/pl_cifar10"
mkdir -p $OUTPUTDIR
now=$(date +"%m_%d_%Y/%H_%M_%S")
echo "$now"
# ---------------------------------

dataset_dir="${HOME}/pytorch_datasets/cifar-100-python"

if [ "$(uname)" = "Darwin" ] ; then
exit
num_workers=4

elif [ "$(uname)" = "Linux" ]; then
config_name="run_gpu_cifar100"
num_workers=4
fi

logger="wandb"
max_epochs=100
module="casregress_onehot"
classifier="wideresnet28_10"
learning_rate=1e-1 #1e-2
weight_decay=5e-4 #1e-2
batch_size=164 #256
gamma=1
seed=0
scheduler="lambdalr"
test_set="CIFAR100Coarse"
num_classes=20

for classifier in "resnet18"
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
  scheduler=${scheduler} \
  test_set=${test_set} \
  ood_dataset=${test_set} \
  num_classes=${num_classes} \

  "
done
