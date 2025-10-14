#!/bin/bash

# Train Latent Motion Tokenizer on HRDT Mixed Dataset (224x224)
# This script trains the tokenizer on EgoDex + RobotWin datasets

export CUDA_VISIBLE_DEVICES=0
export PROJECT_ROOT="/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy"
cd ${PROJECT_ROOT}/latent_motion_tokenizer/train
accelerate launch --main_process_port 29501 train_latent_motion_tokenizer.py \
  --config_path "${PROJECT_ROOT}/latent_motion_tokenizer/configs/train/hrdt_mix_train.yaml"

<<COMMENT
# Usage:
conda activate moto
export PROJECT_ROOT="/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy"

# Kill existing processes if needed
# ps aux | grep "hrdt_mix_train" | awk '{print $2}' | xargs kill -9

# Run training
cd ${PROJECT_ROOT}/scripts/
nohup bash hrdt.sh > train_hrdt_mix_224.log 2>&1 &
tail -f train_hrdt_mix_224.log
COMMENT
