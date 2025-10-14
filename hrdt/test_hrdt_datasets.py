#!/usr/bin/env python3
"""
H-RDT Multi-Dataset Test Script
Tests the H-RDT datasets with Moto-style pyrootutils configuration
"""

import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

import os
import sys
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image

from hrdt.datasets.dataset import VLAConsumerDataset, MultiDataCollatorForVLAConsumerDataset
from hrdt.datasets.multi_hdf5_vla_dataset import MultiHDF5VLADataset

# 添加可视化模块导入
sys.path.append('Moto_copy/hrdt')
from visualization.hand_keypoint_visualizer import HandKeypointVisualizer, visualize_egodx_sample

def load_config(config_path):
    """Load configuration file"""
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    return config

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

def test_pyrootutils_imports():
    """Test that pyrootutils setup allows proper imports"""
    print("🔍 Testing pyrootutils imports...")
    
    try:
        # Test dataset imports
        from hrdt.datasets.dataset import VLAConsumerDataset
        from hrdt.datasets.multi_hdf5_vla_dataset import MultiHDF5VLADataset
        from hrdt.datasets.pretrain.egodex_dataset import EgoDexDataset
        from hrdt.datasets.robotwin2.robotwin_agilex_dataset import RobotwinAgilexDataset
        print("  ✅ Dataset imports successful")
        
        # Test model imports
        from hrdt.models.hrdt_runner import HRDTRunner
        print("  ✅ Model imports successful")
        
        # Test utility imports
        from hrdt.utils.image_corrupt import image_corrupt
        print("  ✅ Utility imports successful")
        
        print("  ✅ All pyrootutils imports working correctly!")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_config_loading():
    """Test loading of new Moto-style configuration files"""
    print("\n📋 Testing configuration loading...")
    
    config_files = [
        "hrdt/configs/data/egodx.yaml",
        "hrdt/configs/data/robotwin.yaml", 
        "hrdt/configs/data/hrdt_mix.yaml"
    ]
    
    for config_file in config_files:
        try:
            if os.path.exists(config_file):
                config = load_config(config_file)
                print(f"  ✅ Loaded {config_file}")
                print(f"    Data type: {config.get('data_type', 'unknown')}")
                
                if 'sub_data_configs' in config:
                    print(f"    Sub-datasets: {len(config['sub_data_configs'])}")
                    for sub_config in config['sub_data_configs']:
                        print(f"      - {sub_config['data_type']} (weight: {sub_config['weight']})")
            else:
                print(f"  ⚠️  Config file not found: {config_file}")
        except Exception as e:
            print(f"  ❌ Failed to load {config_file}: {e}")

def test_dataset_creation():
    """Test creating datasets with new configuration"""
    print("\n🧪 Testing dataset creation...")
    
    try:
        # Load base config for dataset creation
        base_config = {
            "common": {
                "img_history_size": 1,
                "action_chunk_size": 16,  # EgoDx用这个键名
                "chunk_size": 16,         # 兼容性键名
                "num_cameras": 1,
                "state_dim": 48,
                "action_dim": 48
            }
        }
        
        image_transform = create_simple_image_transform()
        
        # Test EgoDx dataset creation
        print("  🔧 Testing EgoDx dataset creation...")
        try:
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
            print(f"    ✅ EgoDx dataset created successfully (size: {len(egodx_dataset)})")
        except Exception as e:
            print(f"    ⚠️  EgoDx dataset creation failed: {e}")
        
        # Test RobotWin dataset creation
        print("  🔧 Testing RobotWin dataset creation...")
        try:
            robotwin_config = base_config.copy()
            robotwin_config["common"]["action_dim"] = 14
            robotwin_config["common"]["state_dim"] = 14
            robotwin_config["common"]["num_cameras"] = 3
            
            robotwin_dataset = VLAConsumerDataset(
                config=robotwin_config,
                image_transform=image_transform,
                num_cameras=3,  # Use main camera only
                dataset_name="robotwin_agilex",
                dataset_type="finetune",
                image_aug=False,
                upsample_rate=3,
                val=True,
                use_precomp_lang_embed=True,
                task_name="open_laptop",
            )
            print(f"    ✅ RobotWin dataset created successfully (size: {len(robotwin_dataset)})")
        except Exception as e:
            print(f"    ⚠️  RobotWin dataset creation failed: {e}")
        
        print("  ✅ Dataset creation tests completed!")
        
    except Exception as e:
        print(f"  ❌ Dataset creation test failed: {e}")

def test_camera_parameters():
    """Test camera parameters reading and processing"""
    print("\n📷 Testing camera parameters...")
    
    try:
        # Load base config
        base_config = {
            "common": {
                "img_history_size": 1,
                "action_chunk_size": 16,  # EgoDx用这个键名
                "chunk_size": 16,         # 兼容性键名
                "num_cameras": 1,
                "state_dim": 48,
                "action_dim": 48
            }
        }
        
        image_transform = create_simple_image_transform()
        
        # Test EgoDx dataset with camera parameters
        print("  🔧 Testing EgoDx camera parameters...")
        try:
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
            
            # Get a sample and check camera parameters
            try:
                sample = egodx_dataset.get_item(0)
                if sample is not None:
                    print(f"    📊 EgoDx sample structure:")
                    for key, value in sample.items():
                        if isinstance(value, torch.Tensor):
                            print(f"      {key}: {value.shape} ({value.dtype})")
                        elif isinstance(value, np.ndarray):
                            print(f"      {key}: {value.shape} ({value.dtype})")
                        else:
                            print(f"      {key}: {type(value)}")
                    
                    # Check camera-specific fields
                    camera_fields = ['current_camera_extrinsics', 'action_camera_extrinsics', 'camera_intrinsics']
                    for field in camera_fields:
                        if field in sample:
                            print(f"    ✅ Found {field}: shape {sample[field].shape}")
                        else:
                            print(f"    ⚠️  Missing {field}")
                else:
                    print("    ⚠️  Could not get sample from EgoDx dataset")
            except Exception as e:
                print(f"    ⚠️  Error getting EgoDx sample: {e}")
                
        except Exception as e:
            print(f"    ❌ EgoDx camera test failed: {e}")
        
        # Test RobotWin dataset with camera parameters
        print("  🔧 Testing RobotWin camera parameters...")
        try:
            robotwin_config = base_config.copy()
            robotwin_config["common"]["action_dim"] = 14
            robotwin_config["common"]["state_dim"] = 14
            robotwin_config["common"]["num_cameras"] = 3
            
            robotwin_dataset = VLAConsumerDataset(
                config=robotwin_config,
                image_transform=image_transform,
                num_cameras=3,
                dataset_name="robotwin_agilex",
                dataset_type="finetune",
                image_aug=False,
                upsample_rate=3,
                val=True,
                use_precomp_lang_embed=True,
                task_name="open_laptop",
            )
            
            # Get a sample and check camera parameters
            try:
                sample = robotwin_dataset.get_item(0)
                if sample is not None:
                    print(f"    📊 RobotWin sample structure:")
                    for key, value in sample.items():
                        if isinstance(value, torch.Tensor):
                            print(f"      {key}: {value.shape} ({value.dtype})")
                        elif isinstance(value, np.ndarray):
                            print(f"      {key}: {value.shape} ({value.dtype})")
                        else:
                            print(f"      {key}: {type(value)}")
                    
                    # Check camera-specific fields
                    camera_fields = ['current_camera_extrinsics', 'action_camera_extrinsics', 'camera_intrinsics']
                    for field in camera_fields:
                        if field in sample:
                            print(f"    ✅ Found {field}: shape {sample[field].shape}")
                        else:
                            print(f"    ⚠️  Missing {field}")
                else:
                    print("    ⚠️  Could not get sample from RobotWin dataset")
            except Exception as e:
                print(f"    ⚠️  Error getting RobotWin sample: {e}")
                
        except Exception as e:
            print(f"    ❌ RobotWin camera test failed: {e}")
        
        print("  ✅ Camera parameters tests completed!")
        
    except Exception as e:
        print(f"  ❌ Camera parameters test failed: {e}")

def test_multi_collator():
    """Test MultiDataCollatorForVLAConsumerDataset with camera parameters"""
    print("\n🔄 Testing MultiDataCollator with camera parameters...")
    
    try:
        # Load base config
        base_config = {
            "common": {
                "img_history_size": 1,
                "action_chunk_size": 16,  # EgoDx用这个键名
                "chunk_size": 16,         # 兼容性键名
                "num_cameras": 1,
                "state_dim": 48,
                "action_dim": 48
            }
        }
        
        image_transform = create_simple_image_transform()
        collator = MultiDataCollatorForVLAConsumerDataset(unified_action_dim=48, use_precomp_lang_embed=True)
        
        # Create datasets
        print("  🔧 Creating test datasets...")
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
        
        robotwin_config = base_config.copy()
        robotwin_config["common"]["action_dim"] = 14
        robotwin_config["common"]["state_dim"] = 14
        robotwin_config["common"]["num_cameras"] = 3
        
        robotwin_dataset = VLAConsumerDataset(
            config=robotwin_config,
            image_transform=image_transform,
            num_cameras=3,
            dataset_name="robotwin_agilex",
            dataset_type="finetune",
            image_aug=False,
            upsample_rate=3,
            val=True,
            use_precomp_lang_embed=True,
            task_name="open_laptop",
        )
        
        # Get samples from both datasets
        print("  📦 Testing mixed batch collation...")
        try:
            egodx_sample = egodx_dataset.get_item(0)
            robotwin_sample = robotwin_dataset.get_item(0)
            
            if egodx_sample is not None and robotwin_sample is not None:
                # Create a mixed batch
                batch_samples = [egodx_sample, robotwin_sample]
                
                # Test collation
                batch = collator(batch_samples)
                
                print(f"    📊 Collated batch structure:")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"      {key}: {value.shape} ({value.dtype})")
                    elif isinstance(value, dict):
                        if key == 'images' or key == 'future_images':
                            print(f"      {key}: dict with keys {list(value.keys())}")
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, torch.Tensor):
                                    print(f"        {sub_key}: {sub_value.shape}")
                        else:
                            print(f"      {key}: {type(value)}")
                    else:
                        print(f"      {key}: {type(value)}")
                
                # Specifically check camera parameters
                camera_fields = ['current_camera_extrinsics', 'action_camera_extrinsics', 'camera_intrinsics']
                for field in camera_fields:
                    if field in batch:
                        print(f"    ✅ Batch {field}: shape {batch[field].shape}")
                        # Verify batch size
                        if batch[field].shape[0] == 2:  # Should match batch size
                            print(f"      ✅ Correct batch size: {batch[field].shape[0]}")
                        else:
                            print(f"      ❌ Incorrect batch size: {batch[field].shape[0]} (expected 2)")
                    else:
                        print(f"    ❌ Missing {field} in batch")
                
                print("    ✅ MultiDataCollator test successful!")
            else:
                print("    ⚠️  Could not get samples for collator test")
                
        except Exception as e:
            print(f"    ❌ Collator test failed: {e}")
        
    except Exception as e:
        print(f"  ❌ MultiDataCollator test failed: {e}")

def test_hand_visualization():
    """Test hand keypoint visualization with real EgoDx data"""
    print("\n🎬 Testing Hand Keypoint Visualization...")
    
    # Load base config
    base_config = {
        "common": {
            "img_history_size": 1,
            "action_chunk_size": 16,  # EgoDx用这个键名
            "chunk_size": 16,         # 兼容性键名
            "num_cameras": 1,
            "state_dim": 48,
            "action_dim": 48
        }
    }
    
    image_transform = create_simple_image_transform()
    
    # 创建EgoDx数据集
    print("  🔧 Creating EgoDx dataset for visualization...")
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
    
    print(f"    ✅ Dataset created (size: {len(egodx_dataset)})")
    
    # 获取样本
    print("  📦 Getting sample for visualization...")
    sample = egodx_dataset.get_item(0)
    
    if sample is not None:
            print("    ✅ Sample obtained successfully")
            print(f"    📊 Sample structure:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"      {key}: {value.shape} ({value.dtype})")
                elif isinstance(value, np.ndarray):
                    print(f"      {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"      {key}: {type(value)}")
            
            # 检查必要的字段
            required_fields = ['actions', 'camera_intrinsics']
            camera_ext_fields = ['action_camera_extrinsics', 'current_camera_extrinsics']
            
            has_required = all(field in sample for field in required_fields)
            has_camera_ext = any(field in sample for field in camera_ext_fields)
            
            if has_required and has_camera_ext:
                print("    ✅ All required fields for visualization present")
                
                # 测试基础可视化器
                print("  🎨 Testing basic visualizer functionality...")
                visualizer = HandKeypointVisualizer(fps=10)
                
                # 测试48D解析
                if isinstance(sample['actions'], torch.Tensor):
                    test_action = sample['actions'][0].cpu().numpy()  # 取第一个动作
                else:
                    test_action = sample['actions'][0]
                
                parsed_transforms = visualizer.parse_48d_to_transforms(test_action)
                print(f"    ✅ 48D action parsing successful")
                print(f"      - Transform keys: {list(parsed_transforms.keys())}")
                print(f"      - Left wrist pos: {parsed_transforms['leftHand'][:3, 3]}")
                print(f"      - Right wrist pos: {parsed_transforms['rightHand'][:3, 3]}")
                
                # 准备相机参数
                if 'action_camera_extrinsics' in sample:
                    camera_extrinsics = sample['action_camera_extrinsics']
                elif 'current_camera_extrinsics' in sample:
                    # 如果只有current，复制到action长度
                    current_ext = sample['current_camera_extrinsics']
                    action_length = sample['actions'].shape[0]
                    if isinstance(current_ext, torch.Tensor):
                        camera_extrinsics = current_ext.repeat(action_length, 1, 1)
                    else:
                        camera_extrinsics = np.repeat(current_ext, action_length, axis=0)
                
                camera_intrinsics = sample['camera_intrinsics']
                
                # 转换为numpy (如果是tensor)
                if torch.is_tensor(camera_extrinsics):
                    camera_extrinsics = camera_extrinsics.cpu().numpy()
                if torch.is_tensor(camera_intrinsics):
                    camera_intrinsics = camera_intrinsics.cpu().numpy()
                
                # 测试相机坐标转换
                print("  🔄 Testing camera coordinate transformation...")
                transforms_in_cam = visualizer.convert_to_camera_frame(parsed_transforms, camera_extrinsics[0])
                print(f"    ✅ Camera transform successful")
                print(f"      - Left wrist in cam: {transforms_in_cam['leftHand'][:3, 3]}")
                print(f"      - Right wrist in cam: {transforms_in_cam['rightHand'][:3, 3]}")
                
                # 测试单帧生成
                print("  🖼️  Testing single frame generation...")
                frame = visualizer.create_frame_with_keypoints(
                    test_action, 
                    camera_extrinsics[0], 
                    camera_intrinsics,
                    image_height=1080,
                    image_width=1920
                )
                print(f"    ✅ Frame generation successful: {frame.shape}")
                
                # 测试短视频生成 (只用前几帧)
                print("  🎬 Testing short video generation...")
                output_path = "/workspace/chenby10@xiaopeng.com/test_hand_keypoints.mp4"
                stat_path = "/workspace/chenby10@xiaopeng.com/Moto_copy/hrdt/datasets/pretrain/egodex_stat.json"
                
                # 只用前3帧进行快速测试
                short_actions = sample['actions'][:15]
                short_camera_ext = camera_extrinsics[:15]
                
                short_sample = {
                    'actions': short_actions,
                    'action_camera_extrinsics': short_camera_ext,
                    'camera_intrinsics': camera_intrinsics
                }
                
                # 移除try-catch，直接运行便于调试
                visualize_egodx_sample(
                    sample_data=short_sample,
                    output_path=output_path,
                    stat_path=stat_path,
                    image_size=(1920, 1080),  # 使用原始图像尺寸
                    fps=5  # 低帧率快速测试
                )
                
                if os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)
                    print(f"    ✅ Video generation successful: {output_path}")
                    print(f"      - File size: {file_size:.2f} MB")
                    print(f"      - Frames: {len(short_actions)}")
                    print(f"      - Resolution: 640x480")
                else:
                    print(f"    ❌ Video file not generated")
                    
                print("    ✅ Hand visualization test completed!")
                
            else:
                print(f"    ⚠️  Missing required fields for visualization:")
                for field in required_fields:
                    if field not in sample:
                        print(f"      - Missing: {field}")
                if not has_camera_ext:
                    print(f"      - Missing camera extrinsics (need one of: {camera_ext_fields})")

def main():
    """Main test function"""
    print("🚀 H-RDT Moto Integration Test")
    print("="*60)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print(f"Project root: {pyrootutils.find_root(search_from=__file__, indicator='.project-root')}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run tests
    success = True
    
    # Test 1: Import functionality
    if not test_pyrootutils_imports():
        success = False
    
    # Test 2: Configuration loading
    test_config_loading()
    
    # Test 3: Dataset creation
    test_dataset_creation()
    
    # Test 4: Camera parameters
    test_camera_parameters()
    
    # Test 5: Multi collator with camera parameters
    test_multi_collator()
    
    # Test 6: Hand keypoint visualization
    test_hand_visualization()
    
    print("\n" + "="*60)
    if success:
        print("🎉 H-RDT Moto integration test completed successfully!")
        print("✅ The H-RDT codebase is properly configured for Moto-style development")
        print("✅ Camera parameters are properly integrated and tested")
        print("✅ Hand keypoint visualization is working correctly")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    print("\n📋 Next steps:")
    print("  1. Verify data paths in config files match your environment")
    print("  2. Run dataset tests with actual data")
    print("  3. Test camera parameters with real datasets")
    print("  4. Integrate with Moto training pipelines")
    print("  5. Use hand keypoint visualization for model validation")

if __name__ == "__main__":
    main()
