#!/bin/bash

# RobotWin + EgoDex Embodiment-aware Action-only Training
set -e

export CUDA_VISIBLE_DEVICES=0,1
export PROJECT_ROOT="/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy"

cd ${PROJECT_ROOT}/latent_motion_tokenizer/train

accelerate launch --main_process_port 29502 train_latent_motion_tokenizer.py \
  --config_path "${PROJECT_ROOT}/latent_motion_tokenizer/configs/train/action_only_robotwin_egodex.yaml"

echo ""
echo "✅ RobotWin+EgoDex action-only co-training completed!"


