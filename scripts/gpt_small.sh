#!/bin/bash

umask 000


export TF_CPP_MIN_LOG_LEVEL=3       # silence annoying TF warnings
export GPT_CONFIG=config/gpt2_small.yaml

python3 train.py