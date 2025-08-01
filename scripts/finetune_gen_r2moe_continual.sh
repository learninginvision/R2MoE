#!/usr/bin/env bash
#### command to run with retrieved images as regularization
# 1st arg: target caption
# 2nd arg: path to target images
# 3rd arg: path where generated images are saved
# 4rth arg: name of the experiment
# 5th arg: config name
# 6th arg: pretrained model path

ARRAY=()

for i in "$@"
do 
    echo $i
    ARRAY+=("${i}")
done

# Replace "/your/path/to/dataset/" with the actual path to your dataset
# Replace "/your/path/to/reg_dataset/" with the actual path to your regularization dataset
python -u  train_lora_continual.py \
        --base configs/R2MoE/finetune_addtoken_c_lora.yaml  \
        -t --gpus 0, \
        --resume-from-checkpoint-custom sd-v1-5-emaonly.ckpt \
        --datapath /your/path/to/dataset/ \
        --reg_datapath /your/path/to/reg_dataset/ \
        --modifier_token "<new1>+<new2>+<new3>+<new4>+<new5>" \
        --name "ckpt-name"\
        --batch_size 2\
        --num_tasks 5\
        --concepts "dog,duck_toy,cat,backpack,teddybear"\
        --device "cuda:0"