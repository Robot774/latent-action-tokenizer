#!/usr/bin/env python3
"""
详细调试数据读取流程，追踪每个步骤的数据变化
"""

import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

import torch
import numpy as np
from common.data.data_utils import load_dataset
from hrdt.datasets.dataset import MultiDataCollatorForVLAConsumerDataset
from torch.utils.data import DataLoader
import omegaconf
from PIL import Image
import torchvision.transforms as T

def print_detailed_stats(tensor, name, step):
    """打印详细的tensor统计信息"""
    if isinstance(tensor, dict):
        print(f"\n🔍 [{step}] {name} (dict):")
        for key, val in tensor.items():
            if isinstance(val, torch.Tensor):
                # 转换为float进行统计计算
                val_float = val.float() if val.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64] else val
                print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                print(f"       min={val_float.min():.6f}, max={val_float.max():.6f}, mean={val_float.mean():.6f}, std={val_float.std():.6f}")
    elif isinstance(tensor, torch.Tensor):
        # 转换为float进行统计计算
        tensor_float = tensor.float() if tensor.dtype in [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64] else tensor
        print(f"\n🔍 [{step}] {name}:")
        print(f"  shape={tensor.shape}, dtype={tensor.dtype}")
        print(f"  min={tensor_float.min():.6f}, max={tensor_float.max():.6f}, mean={tensor_float.mean():.6f}, std={tensor_float.std():.6f}")
        
        # 按通道分析
        if tensor.dim() >= 3 and tensor.shape[-1] == 3:  # 假设最后一维是RGB
            for c in range(3):
                channel_name = ['R', 'G', 'B'][c]
                channel_data = tensor_float[..., c]
                print(f"  {channel_name}: min={channel_data.min():.6f}, max={channel_data.max():.6f}, mean={channel_data.mean():.6f}")
    elif isinstance(tensor, np.ndarray):
        print(f"\n🔍 [{step}] {name} (numpy):")
        print(f"  shape={tensor.shape}, dtype={tensor.dtype}")
        print(f"  min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.mean():.6f}, std={tensor.std():.6f}")

def debug_single_sample():
    """调试单个样本的完整流程"""
    
    print("🚀 开始详细调试数据读取流程...")
    
    # 1. 直接从robotwin数据集获取原始数据
    print("\n" + "="*60)
    print("步骤1: 直接从robotwin数据集获取原始数据")
    
    from hrdt.datasets.robotwin2.robotwin_agilex_dataset import RobotwinAgilexDataset
    
    # 创建robotwin数据集实例
    config = {
        'common': {
            'img_history_size': 1,
            'action_chunk_size': 16,
            'chunk_size': 16,
            'num_cameras': 3,
            'state_dim': 48,
            'action_dim': 48
        }
    }
    
    robotwin_dataset = RobotwinAgilexDataset(
        mode="multi_task",
        multi_task_root_dir="/dataset_rc_mm/share/datasets/huggingface.co/TianxingChen/RoboTwin2.0/dataset",
        config=config,
        val=False
    )
    
    # 获取一个原始样本
    raw_sample = robotwin_dataset.get_item()
    if raw_sample is None:
        print("❌ 无法获取原始样本")
        return
    
    print("✅ 成功获取原始样本")
    
    # 检查原始图像数据
    if 'current_images' in raw_sample:
        current_imgs = raw_sample['current_images']  # numpy array
        print_detailed_stats(torch.from_numpy(current_imgs), "原始current_images", "1a")
        
        # 保存原始图像样本
        if current_imgs.shape[0] > 0:  # 有相机数据
            sample_img = current_imgs[0, 0]  # 第一个相机，第一帧 (H, W, 3)
            if sample_img.max() > 1:  # 如果是0-255范围
                sample_img_pil = Image.fromarray(sample_img.astype(np.uint8))
            else:  # 如果是0-1范围
                sample_img_pil = Image.fromarray((sample_img * 255).astype(np.uint8))
            sample_img_pil.save("/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/debug_output/raw_original.png")
            print(f"  💾 保存原始图像到: debug_output/raw_original.png")
    
    # 2. 通过VLAConsumerDataset处理
    print("\n" + "="*60)
    print("步骤2: 通过VLAConsumerDataset处理")
    
    dataset_config_path = "/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer/configs/data/hrdt_robotwin.yaml"
    extra_data_config = {
        'sequence_length': 1,
        'do_extract_future_frames': True,
        'do_extract_action': False
    }
    
    train_dataset, eval_dataset = load_dataset(dataset_config_path, extra_data_config)
    
    # 获取一个处理后的样本
    processed_sample = eval_dataset[0]
    
    if 'images' in processed_sample:
        images = processed_sample['images']
        print_detailed_stats(images, "VLAConsumerDataset处理后的images", "2a")
        
        # 如果是dict格式，检查pixel_values
        if isinstance(images, dict) and 'pixel_values' in images:
            pixel_values = images['pixel_values']
            print_detailed_stats(pixel_values, "pixel_values", "2b")
            
            # 保存处理后的图像
            if pixel_values.dim() >= 3:
                # 取第一个样本
                sample_tensor = pixel_values[0] if pixel_values.dim() == 4 else pixel_values
                
                # 尝试反normalization看看
                imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                
                # 反normalization
                denorm_tensor = sample_tensor * imagenet_std + imagenet_mean
                denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
                
                print_detailed_stats(denorm_tensor, "反normalization后", "2c")
                
                # 保存图像
                denorm_pil = T.ToPILImage()(denorm_tensor)
                denorm_pil.save("/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/debug_output/processed_denorm.png")
                print(f"  💾 保存反normalization图像到: debug_output/processed_denorm.png")
    
    # 3. 通过DataLoader处理
    print("\n" + "="*60)
    print("步骤3: 通过DataLoader处理")
    
    collator = MultiDataCollatorForVLAConsumerDataset(
        unified_action_dim=48, 
        use_precomp_lang_embed=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False
    )
    
    batch = next(iter(eval_dataloader))
    
    if 'rgb_initial' in batch:
        print_detailed_stats(batch['rgb_initial'], "DataLoader输出的rgb_initial", "3a")
    if 'rgb_future' in batch:
        print_detailed_stats(batch['rgb_future'], "DataLoader输出的rgb_future", "3b")
    
    # 4. 检查image_transform函数
    print("\n" + "="*60)
    print("步骤4: 检查image_transform函数")
    
    from common.data.data_utils import create_hrdt_image_transform
    
    # 创建一个测试图像
    test_img = Image.new('RGB', (224, 224), (128, 128, 128))  # 中等灰度
    
    transform_fn = create_hrdt_image_transform((224, 224))
    transformed = transform_fn(test_img)
    
    print_detailed_stats(transformed, "transform函数输出", "4a")
    
    # 5. 手动测试normalization
    print("\n" + "="*60)
    print("步骤5: 手动测试normalization")
    
    # 创建一个已知的测试tensor
    test_tensor = torch.ones(3, 224, 224) * 0.5  # 中等亮度
    print_detailed_stats(test_tensor, "测试tensor (0.5亮度)", "5a")
    
    # 应用ImageNet normalization
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    normalized = (test_tensor - imagenet_mean) / imagenet_std
    print_detailed_stats(normalized, "手动normalization后", "5b")
    
    print("\n" + "="*60)
    print("🎯 调试完成！请检查上述输出找出问题所在。")

if __name__ == "__main__":
    debug_single_sample()
