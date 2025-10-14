export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd ${PROJECT_ROOT}/latent_motion_tokenizer/train
accelerate launch --main_process_port 29501 train_latent_motion_tokenizer.py --config_path "${PROJECT_ROOT}/latent_motion_tokenizer/configs/train/${CONFIG_NAME}.yaml"


<<COMMENT
conda activate moto
export PROJECT_ROOT=[your path to Moto project]
export CONFIG_NAME="data_rtx-vq_size128_dim32_num8_legacyTrue-vision_MaeLarge-decoder_queryFusionModeAdd_Patch196_useMaskFalse-mformer_legacyTrue-train_lr0.0001_bs256-aug_shiftTrue_resizedCropFalse"
# ps aux | grep ${CONFIG_NAME} | awk '{print $2}' | xargs kill -9
cd ${PROJECT_ROOT}/scripts/
nohup bash train_latent_motion_tokenizer_on_oxe.sh > train_latent_motion_tokenizer_on_oxe.log 2>&1 &
tail -f train_latent_motion_tokenizer_on_oxe.log
COMMENT