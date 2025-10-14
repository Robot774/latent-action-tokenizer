#!/usr/bin/env python3
"""
测试 DinoSigLip 集成到 Latent Motion Tokenizer
"""
import sys
import os
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy')

import torch
from common.processors.preprocessor_utils import get_model_vision_basic_config, get_rgb_preprocessor

def test_rgb_preprocessor():
    """测试 RGB 预处理器的双路径功能"""
    print("=== 测试 RGB Preprocessor ===")
    
    # 测试 MAE 配置
    print("1. 测试 MAE 配置")
    mae_config = get_model_vision_basic_config("mae")
    print(f"MAE config: {mae_config}")
    
    mae_preprocessor = get_rgb_preprocessor("mae")
    test_input = torch.randn(2, 2, 3, 224, 224)  # [B, T, C, H, W]
    mae_output = mae_preprocessor(test_input, train=False)
    print(f"MAE output shape: {mae_output.shape}")
    print(f"MAE output type: {type(mae_output)}")
    
    # 测试 DinoSigLip 配置
    print("\n2. 测试 DinoSigLip 配置")
    try:
        dinosiglip_config = get_model_vision_basic_config("dinosiglip")
        print(f"DinoSigLip config: {dinosiglip_config}")
        
        dinosiglip_preprocessor = get_rgb_preprocessor("dinosiglip")
        test_input_384 = torch.randn(2, 2, 3, 384, 384)  # [B, T, C, H, W]
        dinosiglip_output = dinosiglip_preprocessor(test_input_384, train=False)
        print(f"DinoSigLip output type: {type(dinosiglip_output)}")
        if isinstance(dinosiglip_output, dict):
            for k, v in dinosiglip_output.items():
                print(f"  {k}: {v.shape}")
        else:
            print(f"DinoSigLip output shape: {dinosiglip_output.shape}")
    except Exception as e:
        print(f"DinoSigLip 配置测试失败: {e}")
    
    print("✅ RGB Preprocessor 测试完成")

def test_encoder_compatibility():
    """测试编码器类型检测"""
    print("\n=== 测试编码器兼容性 ===")
    
    try:
        from transformers import ViTMAEModel
        from latent_motion_tokenizer.src.models.latent_motion_tokenizer import LatentMotionTokenizer
        
        # 创建一个简单的 MAE 模型测试
        print("1. 测试 MAE 编码器检测")
        mae_model = ViTMAEModel.from_pretrained("facebook/vit-mae-base", local_files_only=False)
        
        # 这里我们只测试编码器类型检测，不创建完整的 tokenizer
        print("✅ MAE 编码器加载成功")
        
    except Exception as e:
        print(f"编码器兼容性测试失败: {e}")
        print("这可能是因为缺少预训练模型或网络问题")

def test_training_pipeline_compatibility():
    """测试训练管道兼容性"""
    print("\n=== 测试训练管道兼容性 ===")
    
    try:
        # 模拟训练器的 calculate_loss 逻辑
        mae_preprocessor = get_rgb_preprocessor("mae")
        dinosiglip_preprocessor = get_rgb_preprocessor("dinosiglip")
        
        # 创建模拟批次数据
        batch = {
            'rgb_initial': torch.randn(2, 1, 3, 224, 224),
            'rgb_future': torch.randn(2, 1, 3, 224, 224)
        }
        
        # 测试 MAE 路径
        print("1. 测试 MAE 训练路径")
        rgb_seq = torch.cat([batch['rgb_initial'], batch['rgb_future']], dim=1)
        mae_processed = mae_preprocessor(rgb_seq, train=False)
        print(f"MAE 处理后形状: {mae_processed.shape}")
        
        # 测试分离时间维度
        cond_pixel_values = mae_processed[:, 0]
        target_pixel_values = mae_processed[:, 1]
        print(f"MAE cond/target 形状: {cond_pixel_values.shape}, {target_pixel_values.shape}")
        
        # 测试 DinoSigLip 路径
        print("\n2. 测试 DinoSigLip 训练路径")
        batch_384 = {
            'rgb_initial': torch.randn(2, 1, 3, 384, 384),
            'rgb_future': torch.randn(2, 1, 3, 384, 384)
        }
        rgb_seq_384 = torch.cat([batch_384['rgb_initial'], batch_384['rgb_future']], dim=1)
        dinosiglip_processed = dinosiglip_preprocessor(rgb_seq_384, train=False)
        
        if isinstance(dinosiglip_processed, dict):
            print(f"DinoSigLip 处理后类型: dict")
            cond_dict = {k: v[:, 0] for k, v in dinosiglip_processed.items()}
            target_dict = {k: v[:, 1] for k, v in dinosiglip_processed.items()}
            print(f"DinoSigLip cond 键: {list(cond_dict.keys())}")
            for k, v in cond_dict.items():
                print(f"  {k}: {v.shape}")
        
        print("✅ 训练管道兼容性测试完成")
        
    except Exception as e:
        print(f"训练管道测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 开始 DinoSigLip 集成测试\n")
    
    test_rgb_preprocessor()
    test_encoder_compatibility() 
    test_training_pipeline_compatibility()
    
    print("\n🎉 所有测试完成！")



