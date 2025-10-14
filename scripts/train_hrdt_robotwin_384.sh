#!/bin/bash

# HRDT RobotWin Training Script (384px)
# Single dataset training on RobotWin only with higher resolution

set -e

export CUDA_VISIBLE_DEVICES=0 # Adjust as needed

# Set PROJECT_ROOT to the absolute path of your Moto_copy project
export PROJECT_ROOT="/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy"

cd ${PROJECT_ROOT}/latent_motion_tokenizer/train

echo "🚀 Starting HRDT RobotWin training (384px)..."
echo "Project Root: $PROJECT_ROOT"
echo "Config: hrdt_robotwin_train_384.yaml"
echo ""

accelerate launch --main_process_port 29502 train_latent_motion_tokenizer.py \
  --config_path "${PROJECT_ROOT}/latent_motion_tokenizer/configs/train/hrdt_robotwin_train_384.yaml"

echo ""
echo "✅ HRDT RobotWin training completed!"

<<COMMENT
Usage Instructions:
1. Make sure you are in the correct conda environment
2. Ensure RobotWin dataset is properly extracted and accessible
3. Run: bash scripts/train_hrdt_robotwin_384.sh
4. For background execution: nohup bash scripts/train_hrdt_robotwin_384.sh > robotwin_384_train.log 2>&1 &
5. Monitor progress: tail -f robotwin_384_train.log

Configuration Details:
- Dataset: RobotWin only (no EgoDx mixing)
- Resolution: 384x384 (higher quality)
- Task: open_laptop (configurable)
- Max Episodes: 100 (limited for faster training)
- Batch Size: 4 (reduced for memory)
- Mixed Precision: fp16
- Gradient Accumulation: 2 steps
COMMENT
