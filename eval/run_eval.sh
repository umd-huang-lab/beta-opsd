#!/bin/bash

BASE_MODEL="/data0/shared/Qwen3-1.7B"

# evaluate base model performance
# Defaults match evaluate_math.py: temp=1.0, thinking on, max_new_tokens=38912,
# top_p=1.0 (no nucleus), top_k=-1, min_p=0, presence_penalty=0, val_n=12
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --temperature 1.0 \
    --max_new_tokens 38912 \
    --top_p 1.0 \
    --top_k -1 \
    --min_p 0.0 \
    --presence_penalty 0.0 \
    --val_n 1 \
    --tensor_parallel_size 4 
wait

# after trained, evaluate the performance of the trained model
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --temperature 1.0 \
    --max_new_tokens 38912 \
    --top_p 1.0 \
    --top_k -1 \
    --min_p 0.0 \
    --presence_penalty 0.0 \
    --val_n 1 \
    --tensor_parallel_size 4 \
    --checkpoint_dir /data1/siyanz/opsd/qwen31b_gen2048_fixteacher_temp11_lr2e4/checkpoint-200
wait
    
