#!/bin/bash

#$ -l rt_G.small=1
#$ -l h_rt=24:00:00
#$ -j y
#$ -cwd

# Init env(must)
source /etc/profile.d/modules.sh

# Make env
module load python/3.6/3.6.5
module load cuda/10.1/10.1.243
source ~/lib/pyenv/kaggle_cloud/bin/activate

# Source directory(ABCI)
cd ~/kaggle/cloud-organization-2019/src

model=model013
conf_path=./configs/${model}.yaml
kfold=1
mode=bestloss

python eval.py \
    --kfold ${kfold} \
    --config ${conf_path} \
    --model-path ../output/model/${model}/kfold_${kfold}_${mode}.ckpt

mode=latest
python eval.py \
    --kfold ${kfold} \
    --config ${conf_path} \
    --model-path ../output/model/${model}/kfold_${kfold}_${mode}.ckpt
