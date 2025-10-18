#!/bin/bash

# HRDT RobotWin Training Script (224px)
# Single dataset training on RobotWin only

set -e

export CUDA_VISIBLE_DEVICES=2,3

# Set PROJECT_ROOT to the absolute path of your Moto_copy project
export PROJECT_ROOT="/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy"

cd ${PROJECT_ROOT}/latent_motion_tokenizer/train

accelerate launch --main_process_port 29501 train_latent_motion_tokenizer.py \
  --config_path "${PROJECT_ROOT}/latent_motion_tokenizer/configs/train/action_only_dct.yaml"

echo ""
echo "✅ HRDT RobotWin training completed!"

<<COMMENT
Usage Instructions:
1. Make sure you are in the correct conda environment
2. Ensure RobotWin dataset is properly extracted and accessible
3. Run: bash scripts/train_hrdt_robotwin_224.sh
4. For background execution: nohup bash scripts/train_hrdt_robotwin_224.sh > robotwin_224_train.log 2>&1 &
5. Monitor progress: tail -f robotwin_224_train.log

Configuration Details:
- Dataset: RobotWin only (no EgoDx mixing)
- Resolution: 224x224
- Task: open_laptop (configurable)
- Max Episodes: Unlimited (full dataset)
- Batch Size: 8
- Mixed Precision: fp16
COMMENT
