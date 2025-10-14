#!/usr/bin/env python3
"""
简化的手部关键点可视化测试 - 移除错误捕获便于调试
"""

import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

import os
import sys
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# 添加可视化模块导入
sys.path.append('Moto_copy/hrdt')
from visualization.hand_keypoint_visualizer import HandKeypointVisualizer, visualize_egodx_sample
from hrdt.datasets.dataset import VLAConsumerDataset

def create_simple_image_transform():
    """Create simple image transform for testing"""
    def transform_fn(image):
        if image is None:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        pixel_values = transform(image)
        return {"pixel_values": pixel_values}
    
    return transform_fn

def test_hand_visualization():
    """简化的手部关键点可视化测试 - 无错误捕获"""
    print("🎬 Testing Hand Keypoint Visualization (Direct Debug Mode)...")
    
    # 配置
    base_config = {
        "common": {
            "img_history_size": 1,
            "action_chunk_size": 16,
            "chunk_size": 16,
            "num_cameras": 1,
            "state_dim": 48,
            "action_dim": 48
        }
    }
    
    image_transform = create_simple_image_transform()
    
    # 创建数据集
    print("🔧 Creating EgoDx dataset...")
    egodx_dataset = VLAConsumerDataset(
        config=base_config,
        image_transform=image_transform,
        num_cameras=1,
        dataset_name="egodex",
        dataset_type="pretrain",
        image_aug=False,
        upsample_rate=3,
        val=True,
        use_precomp_lang_embed=True,
    )
    
    print(f"✅ Dataset created (size: {len(egodx_dataset)})")
    
    # 获取样本
    print("📦 Getting sample...")
    sample = egodx_dataset.get_item(0)
    
    print("✅ Sample obtained")
    print("📊 Sample structure:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} ({value.dtype})")
        elif isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape} ({value.dtype})")
        else:
            print(f"  {key}: {type(value)}")
    
    # 检查必要字段
    required_fields = ['actions', 'camera_intrinsics']
    camera_ext_fields = ['action_camera_extrinsics', 'current_camera_extrinsics']
    
    has_required = all(field in sample for field in required_fields)
    has_camera_ext = any(field in sample for field in camera_ext_fields)
    
    if not has_required or not has_camera_ext:
        print("❌ Missing required fields")
        return
    
    print("✅ All required fields present")
    
    # 基础可视化器测试
    print("🎨 Testing basic visualizer...")
    visualizer = HandKeypointVisualizer(fps=10)
    
    # 测试数据准备
    if isinstance(sample['actions'], torch.Tensor):
        test_action = sample['actions'][0].cpu().numpy()
    else:
        test_action = sample['actions'][0]
    
    # 测试48D解析
    print("🔄 Testing 48D parsing...")
    parsed_transforms = visualizer.parse_48d_to_transforms(test_action)
    print(f"✅ Parsed {len(parsed_transforms)} transforms")
    
    # 准备相机参数
    if 'action_camera_extrinsics' in sample:
        camera_extrinsics = sample['action_camera_extrinsics']
    else:
        current_ext = sample['current_camera_extrinsics']
        action_length = sample['actions'].shape[0]
        if isinstance(current_ext, torch.Tensor):
            camera_extrinsics = current_ext.repeat(action_length, 1, 1)
        else:
            camera_extrinsics = np.repeat(current_ext, action_length, axis=0)
    
    camera_intrinsics = sample['camera_intrinsics']
    
    # 转换为numpy
    if torch.is_tensor(camera_extrinsics):
        camera_extrinsics = camera_extrinsics.cpu().numpy()
    if torch.is_tensor(camera_intrinsics):
        camera_intrinsics = camera_intrinsics.cpu().numpy()
    
    # 测试相机变换
    print("🔄 Testing camera transform...")
    transforms_in_cam = visualizer.convert_to_camera_frame(parsed_transforms, camera_extrinsics[0])
    print("✅ Camera transform successful")
    
    # 测试单帧生成
    print("🖼️ Testing frame generation...")
    frame = visualizer.create_frame_with_keypoints(
        test_action, 
        camera_extrinsics[0], 
        camera_intrinsics,
        image_height=1080,
        image_width=1920
    )
    print(f"✅ Frame generated: {frame.shape}")
    
    # 测试视频生成
    print("🎬 Testing video generation...")
    output_path = "/workspace/chenby10@xiaopeng.com/test_hand_keypoints_debug.mp4"
    stat_path = "/workspace/chenby10@xiaopeng.com/Moto_copy/hrdt/datasets/pretrain/egodex_stat.json"
    
    # 只用前3帧测试
    short_actions = sample['actions'][:15]
    short_camera_ext = camera_extrinsics[:15]
    
    short_sample = {
        'actions': short_actions,
        'action_camera_extrinsics': short_camera_ext,
        'camera_intrinsics': camera_intrinsics
    }
    
    # 直接调用，无错误捕获
    visualize_egodx_sample(
        sample_data=short_sample,
        output_path=output_path,
        stat_path=stat_path,
        image_size=(1920, 1080),  # 使用原始图像尺寸
        fps=5
    )
    
    # 检查结果
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ Video created: {output_path}")
        print(f"   File size: {file_size:.2f} MB")
        print(f"   Frames: {len(short_actions)}")
    else:
        print("❌ Video not created")
    
    print("🎉 Test completed!")

if __name__ == "__main__":
    test_hand_visualization()
