#!/bin/bash
# Usage: ./eval_tvl_llama.sh $GPU_NUMBER /path/to/your/model /path/to/your/encoder /path/to/your/data 

gpu_num=$1
checkpoint_path=$2

if [ $# -eq 1 ]; then
    python main.py \
    --y_col type_label \
    --update \
    --edge_weight \
    --graphuse \
    --train_gender both \
    --num_token 150 \
    --phase test \
    --checkpoint_path $checkpoint_path

elif [ $# -eq 2 ]; then
    CUDA_VISIBLE_DEVICES=$gpu_num python main.py \
    --y_col type_label \
    --update \
    --edge_weight \
    --graphuse \
    --train_gender both \
    --num_token 150 \
    --phase test \
    --checkpoint_path $checkpoint_path \
    --gpu 0 \
    --use_gpu
else
  echo "Invalid GPU number"
  exit
fi

