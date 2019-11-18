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

model=model021
conf_path=./configs/${model}.yaml
#mode=latest
mode=bestloss

kfold=1
python eval.py \
    --kfold ${kfold} \
    --config ${conf_path} \
    --model-path ../output/model/${model}/kfold_${kfold}_${mode}.pt

#kfold=2
#python eval.py \
#    --kfold ${kfold} \
#    --config ${conf_path} \
#    --model-path ../output/model/${model}/kfold_${kfold}_${mode}.pt
#
#kfold=3
#python eval.py \
#    --kfold ${kfold} \
#    --config ${conf_path} \
#    --model-path ../output/model/${model}/kfold_${kfold}_${mode}.pt
#
#kfold=4
#python eval.py \
#    --kfold ${kfold} \
#    --config ${conf_path} \
#    --model-path ../output/model/${model}/kfold_${kfold}_${mode}.pt
#
#kfold=5
#python eval.py \
#    --kfold ${kfold} \
#    --config ${conf_path} \
#    --model-path ../output/model/${model}/kfold_${kfold}_${mode}.pt
