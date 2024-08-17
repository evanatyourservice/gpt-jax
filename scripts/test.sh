#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export TF_CPP_MIN_LOG_LEVEL=3
export GPT_CONFIG=config/gpt2.yaml  # base config

python3 train.py \
    --out_dir=/Users/evanwalters/gpt_testing/$EXPERIMENT \
    --train_pattern=/Users/evanwalters/owt_10k_data/train_??.tfrecord \
    --val_pattern=/Users/evanwalters/owt_10k_data/val_??.tfrecord \
    --train_steps=100 \
    --eval_interval=10 \
    --eval_steps=5 \
    --hs_eval_steps=5 \
    --batch_size=2 \
    --optimizer.learning_rate=0.00001 \
    --optimizer.warmup_steps=20 \
    --model.n_embd=8 \
    --model.n_head=2 \
    --model.n_layer=1 \
    --model.n_inner=8 \
    --wandb.mode=disabled