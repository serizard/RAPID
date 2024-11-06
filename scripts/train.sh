#!/bin/bash
# Usage: ./eval_tvl_llama.sh $GPU_NUMBER /path/to/your/model /path/to/your/encoder /path/to/your/data 

gpu_num=$1

if [ $# -eq 0 ]; then
    python main.py \
    --y_col type \
    --update \
    --edge_weight \
    --graphuse \
    --train_gender both \
    --num_token 150
elif [ $# -eq 1 ]; then
    CUDA_VISIBLE_DEVICES=$gpu_num python main.py \
    --y_col type \
    --update \
    --edge_weight \
    --graphuse \
    --train_gender both \
    --num_token 150 \
    --gpu 0 \
    --use_gpu
else
  echo "Invalid GPU number"
  exit
fi

