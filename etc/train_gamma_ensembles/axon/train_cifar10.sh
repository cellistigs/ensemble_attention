#!/usr/bin/env bash

#name the job pybench33 and place it's output in a file named slurm-<jobid>.out
# allow 40 minutes to run (it should not take 40 minutes however)
# set partition to 'all' so it runs on any available node on the cluster

#SBATCH -J 'slurm_ekb'
#SBATCH -o slurm_ekb-%j.out
#SBATCH -t 01:00:00
#SBATCH --mem 8gb
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mail-type=ALL            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ekb2154@columbia.edu         # Where to send mail (e.g. uni123@columbia.edu)
. activate interp


echo "Begin call: scrips/run.py $1"
pushd ../../../
python  scrips/run.py $1
echo "Ran scrips/run.py $1"

popd