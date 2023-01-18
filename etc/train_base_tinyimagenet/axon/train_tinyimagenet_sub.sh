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

max_epochs=200
logger="wandb"

module="base"
classifier="resnet101"
label_smoothing=0
gamma=1.0
seed=0
batch_size=128

for weight_decay in {0.0005,0.001};
do
for learning_rate in {0.001,0.01};
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
  learning_rate=${learning_rate} \
  weight_decay=${weight_decay} \
  batch_size=${batch_size} \

  "

done
done

