#!/bin/bash

#$ -l rt_G.large=1
#$ -l h_rt=12:00:00
#$ -j y
#$ -cwd

# Init env(must)
source /etc/profile.d/modules.sh

# Make env
module load python/3.6/3.6.5
module load cuda/10.1/10.1.243
source ~/lib/pyenv/kaggle_siim/bin/activate

# Source directory(ABCI)
cd ~/kaggle/cloud-organization-2019/src

model=model003
conf_path=./configs/${model}.yaml
kfold=1

python train.py --kfold ${kfold} --config ${conf_path}
