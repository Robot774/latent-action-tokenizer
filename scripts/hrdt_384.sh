#!/bin/bash

# Train Latent Motion Tokenizer on HRDT Mixed Dataset (384x384)
# This script trains the tokenizer on EgoDx + RobotWin datasets with higher resolution

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd ${PROJECT_ROOT}/latent_motion_tokenizer/train
accelerate launch --main_process_port 29501 train_latent_motion_tokenizer.py \
  --config_path "${PROJECT_ROOT}/latent_motion_tokenizer/configs/train/hrdt_mix_train_384.yaml"

<<COMMENT
# Usage:
conda activate moto
export PROJECT_ROOT="/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy"

# Kill existing processes if needed
# ps aux | grep "hrdt_mix_train_384" | awk '{print $2}' | xargs kill -9

# Run training
cd ${PROJECT_ROOT}/scripts/
nohup bash train_latent_motion_tokenizer_on_hrdt_mix_384.sh > train_hrdt_mix_384.log 2>&1 &
tail -f train_hrdt_mix_384.log
COMMENT
