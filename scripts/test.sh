#!/bin/bash
# Usage: ./eval_tvl_llama.sh $GPU_NUMBER /path/to/your/model /path/to/your/encoder /path/to/your/data 

CUDA_VISIBLE_DEVICES=0 python main.py \
  --y_col type_label \
  --update \
  --graphuse \
  --edge_weight \
  --train_gender both \
  --chunk_size 50 \
  --num_token 150 \
  --phase test \
  --gpu 0 \
  --use_gpu \
  --checkpoint_path /workspace/ckpts/chk50_nt150_graph_edge-epoch=10-train_loss=0.58.ckpt
