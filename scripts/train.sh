#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python main.py \
  --y_col type_label \
  --update \
  --edge_weight \
  --graphuse \
  --train_gender both \
  --chunk_size 50 \
  --num_token 150 \
  --phase train_test \
  --gpu 0 \
  --use_gpu \
  --save chk50_nt150

