#!/bin/bash

EXPERIMENT=run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export TF_CPP_MIN_LOG_LEVEL=3
export GPT_CONFIG=config/gpt2.yaml  # base config

python3 train.py \
    --out_dir=gs://uscentral1stuff/gpt_models/gpt_small/$EXPERIMENT \
    --train_pattern=gs://uscentral1stuff/openwebtext/train_??.tfrecord \
    --val_pattern=gs://uscentral1stuff/openwebtext/val_??.tfrecord \
    --bfloat16_compute \
    --model.n_embd=512 \
    --model.n_head=8 \
    --model.n_layer=8 \
    --model.n_inner=2048 \
    --optimizer.type=affine \
    --optimizer.weight_decay=0.01 \
    --optimizer.grad_clip=10.0 \
    --optimizer.preconditioner_update_probability=1.0 \
    --optimizer.update_global_norm_clip=5000.0 \
    --optimizer.update_elementwise_clip \
    --optimizer.max_size_triangular=0 \
    --optimizer.max_skew_triangular=0 \
    --optimizer.precond_lr=1.0 \
    --optimizer.precond_init_scale=1.0 \
    --optimizer.adaptive