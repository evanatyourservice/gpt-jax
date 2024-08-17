#!/bin/bash
# usage: ./scripts/test.sh <wandb_api_key>

# wandb_api_key argument
if [ -z "$1" ]; then
    echo "Usage: $0 <wandb_api_key>"
    exit 1
fi

# Set the WANDB_API_KEY environment variable
export WANDB_API_KEY=$1

EXPERIMENT=gpt2-test/run_$(date +%Y-%m-%d_%H-%M-%S)
echo $EXPERIMENT

export TF_CPP_MIN_LOG_LEVEL=3       # silence annoying TF warnings
export GPT_CONFIG=config/test.yaml  # this is the default GPT config for this run

python3 train.py  --out_dir=/Users/evanwalters/gpt_testing/$EXPERIMENT