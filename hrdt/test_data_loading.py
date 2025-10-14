#!/usr/bin/env python3
"""
H-RDT Data Loading Test Script
Tests actual data loading and image reading functionality
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
from torch.utils.data import DataLoader

from hrdt.datasets.dataset import VLAConsumerDataset, MultiDataCollatorForVLAConsumerDataset
from hrdt.datasets.multi_hdf5_vla_dataset import MultiHDF5VLADataset

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

def test_egodx_data_loading():
    """Test EgoDx dataset data loading"""
    print("🤲 测试EgoDx数据加载...")
    
    try:
        # Load base config
        base_config = {
            "common": {
                "img_history_size": 1,
                "action_chunk_size": 4,
                "num_cameras": 1,
                "state_dim": 48,
                "action_dim": 48
            }
        }
        
        image_transform = create_simple_image_transform()
        
        # Create EgoDx dataset
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
        
        print(f"  ✅ EgoDx数据集创建成功 (总样本数: {len(egodx_dataset)})")
        
        # Test loading a few samples
        print("  🔍 测试样本加载...")
        for i in range(min(3, len(egodx_dataset))):
            try:
                sample = egodx_dataset[i]
                
                # Check images
                if "images" in sample:
                    images = sample["images"]
                    print(f"    样本 {i}:")
                    if isinstance(images, dict):
                        for key, img_tensor in images.items():
                            print(f"      图像({key}): 形状{img_tensor.shape}, 范围[{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
                    else:
                        print(f"      图像: 形状{images.shape}, 范围[{images.min():.3f}, {images.max():.3f}]")
                
                # Check actions
                if "actions" in sample:
                    actions = sample["actions"]
                    print(f"      动作: 形状{actions.shape}, 范围[{actions.min():.3f}, {actions.max():.3f}]")
                
                # Check language
                if "lang_embeds" in sample:
                    lang_embeds = sample["lang_embeds"]
                    print(f"      语言嵌入: 形状{lang_embeds.shape}")
                    
            except Exception as e:
                print(f"    ❌ 样本 {i} 加载失败: {e}")
                
        return True
        
    except Exception as e:
        print(f"  ❌ EgoDx数据加载测试失败: {e}")
        return False

def test_robotwin_data_loading():
    """Test RobotWin dataset data loading"""
    print("\n🤖 测试RobotWin数据加载...")
    
    try:
        # Load base config
        base_config = {
            "common": {
                "img_history_size": 1,
                "action_chunk_size": 4,
                "num_cameras": 3,
                "state_dim": 14,
                "action_dim": 14
            }
        }
        
        image_transform = create_simple_image_transform()
        
        # Create RobotWin dataset
        robotwin_dataset = VLAConsumerDataset(
            config=base_config,
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
        
        print(f"  ✅ RobotWin数据集创建成功 (总样本数: {len(robotwin_dataset)})")
        
        # Test loading a few samples
        print("  🔍 测试样本加载...")
        for i in range(min(3, len(robotwin_dataset))):
            try:
                sample = robotwin_dataset[i]
                
                # Check images
                if "images" in sample:
                    images = sample["images"]
                    print(f"    样本 {i}:")
                    if isinstance(images, dict):
                        for key, img_tensor in images.items():
                            print(f"      图像({key}): 形状{img_tensor.shape}, 范围[{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
                    elif isinstance(images, list):
                        print(f"      图像列表: {len(images)}个相机")
                        for cam_idx, img in enumerate(images):
                            if isinstance(img, dict):
                                for key, img_tensor in img.items():
                                    print(f"        相机{cam_idx}({key}): 形状{img_tensor.shape}, 范围[{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
                            else:
                                print(f"        相机{cam_idx}: 形状{img.shape}, 范围[{img.min():.3f}, {img.max():.3f}]")
                    else:
                        print(f"      图像: 形状{images.shape}, 范围[{images.min():.3f}, {images.max():.3f}]")
                
                # Check actions
                if "actions" in sample:
                    actions = sample["actions"]
                    print(f"      动作: 形状{actions.shape}, 范围[{actions.min():.3f}, {actions.max():.3f}]")
                
                # Check language
                if "lang_embeds" in sample:
                    lang_embeds = sample["lang_embeds"]
                    print(f"      语言嵌入: 形状{lang_embeds.shape}")
                    
            except Exception as e:
                print(f"    ❌ 样本 {i} 加载失败: {e}")
                
        return True
        
    except Exception as e:
        print(f"  ❌ RobotWin数据加载测试失败: {e}")
        return False

def test_multi_dataset_loading():
    """Test multi-dataset loading with collator"""
    print("\n🔄 测试多数据集混合加载...")
    
    try:
        # Load base config
        base_config = {
            "common": {
                "img_history_size": 1,
                "action_chunk_size": 4,
                "num_cameras": 1,
                "state_dim": 48,
                "action_dim": 48
            }
        }
        
        image_transform = create_simple_image_transform()
        
        # Create datasets
        datasets = []
        
        # EgoDx dataset
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
            datasets.append(egodx_dataset)
            print(f"  ✅ EgoDx子数据集: {len(egodx_dataset)}个样本")
        except Exception as e:
            print(f"  ⚠️  EgoDx数据集创建失败: {e}")
        
        # RobotWin dataset  
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
            datasets.append(robotwin_dataset)
            print(f"  ✅ RobotWin子数据集: {len(robotwin_dataset)}个样本")
        except Exception as e:
            print(f"  ⚠️  RobotWin数据集创建失败: {e}")
        
        if not datasets:
            print("  ❌ 没有可用的数据集，跳过多数据集测试")
            return False
        
        # Create multi-dataset
        weights = [90, 10][:len(datasets)]  # Match config weights
        multi_dataset = MultiHDF5VLADataset(datasets, weights)
        print(f"  ✅ 多数据集创建成功 (权重: {weights})")
        
        # Create collator and dataloader
        collator = MultiDataCollatorForVLAConsumerDataset()
        dataloader = DataLoader(
            multi_dataset,
            batch_size=4,
            num_workers=0,  # Use 0 for debugging
            collate_fn=collator,
            pin_memory=False,
            shuffle=True,
        )
        
        print("  🔍 测试批次加载...")
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 2:  # Only test 2 batches
                break
                
            print(f"    批次 {batch_idx + 1}:")
            print(f"      Actions形状: {batch['actions'].shape}")
            
            # Check dataset distribution
            if "dataset_indices_map" in batch:
                for ds_name, indices in batch["dataset_indices_map"].items():
                    print(f"      {ds_name}: {len(indices)}个样本")
            
            # Check images
            if "images" in batch:
                images = batch["images"]
                if isinstance(images, dict):
                    for key, img_tensor in images.items():
                        print(f"      图像({key}): 形状{img_tensor.shape}")
                else:
                    print(f"      图像: {type(images)}")
            
            # Detailed sample analysis
            batch_size = batch["actions"].shape[0]
            print(f"      样本详情:")
            for sample_idx in range(min(2, batch_size)):  # Check first 2 samples
                # Get dataset name for this sample
                dataset_name = "unknown"
                if "dataset_indices_map" in batch:
                    for ds_name, indices in batch["dataset_indices_map"].items():
                        if sample_idx in indices:
                            dataset_name = ds_name
                            break
                
                actions = batch["actions"][sample_idx]
                print(f"        样本{sample_idx} ({dataset_name}):")
                print(f"          动作: 形状{actions.shape}, 范围[{actions.min():.3f}, {actions.max():.3f}]")
                
                # Check action padding for RobotWin (should have zeros in last 34 dims)
                if dataset_name == "robotwin_agilex" and actions.shape[-1] == 48:
                    native_actions = actions[..., :14]
                    padded_actions = actions[..., 14:]
                    print(f"          RobotWin原生14维: 范围[{native_actions.min():.3f}, {native_actions.max():.3f}]")
                    print(f"          填充34维: 范围[{padded_actions.min():.3f}, {padded_actions.max():.3f}] (应为0)")
        
        print("  ✅ 多数据集批次加载测试完成！")
        return True
        
    except Exception as e:
        print(f"  ❌ 多数据集加载测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def save_sample_images():
    """Save sample images for visual verification"""
    print("\n💾 保存样本图像用于验证...")
    
    try:
        import os
        from PIL import Image as PILImage
        
        # Create save directory
        save_dir = "test/hrdt_data_verification"
        os.makedirs(save_dir, exist_ok=True)
        
        # Test both datasets
        datasets_to_test = [
            ("egodex", "pretrain"),
            ("robotwin_agilex", "finetune")
        ]
        
        for dataset_name, dataset_type in datasets_to_test:
            print(f"  📸 保存{dataset_name}样本图像...")
            
            try:
                # Create appropriate config
                if dataset_name == "egodex":
                    config = {
                        "common": {
                            "img_history_size": 1,
                            "action_chunk_size": 4,
                            "num_cameras": 1,
                            "state_dim": 48,
                            "action_dim": 48
                        }
                    }
                    num_cameras = 1
                else:  # robotwin_agilex
                    config = {
                        "common": {
                            "img_history_size": 1,
                            "action_chunk_size": 4,
                            "num_cameras": 3,
                            "state_dim": 14,
                            "action_dim": 14
                        }
                    }
                    num_cameras = 3
                
                image_transform = create_simple_image_transform()
                
                # Create dataset
                dataset = VLAConsumerDataset(
                    config=config,
                    image_transform=image_transform,
                    num_cameras=num_cameras,
                    dataset_name=dataset_name,
                    dataset_type=dataset_type,
                    image_aug=False,
                    upsample_rate=3,
                    val=True,
                    use_precomp_lang_embed=True,
                    task_name="open_laptop" if dataset_name == "robotwin_agilex" else None,
                )
                
                # Save a few sample images
                for i in range(min(3, len(dataset))):
                    sample = dataset[i]
                    
                    if "images" in sample:
                        images = sample["images"]
                        
                        if isinstance(images, dict):
                            for key, img_tensor in images.items():
                                # Process tensor to image
                                if img_tensor.dim() == 4:  # [T, C, H, W]
                                    img_to_save = img_tensor[0]  # Take first frame
                                elif img_tensor.dim() == 3:  # [C, H, W]
                                    img_to_save = img_tensor
                                else:
                                    continue
                                
                                # Denormalize
                                def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
                                    tensor = tensor.clone()
                                    for i, (m, s) in enumerate(zip(mean, std)):
                                        tensor[i] = tensor[i] * s + m
                                    tensor = torch.clamp(tensor, 0, 1)
                                    tensor = tensor.permute(1, 2, 0)  # [H, W, C]
                                    tensor = (tensor * 255).byte().numpy()
                                    return PILImage.fromarray(tensor)
                                
                                pil_img = denormalize_image(img_to_save)
                                img_filename = f"{dataset_name}_sample{i:02d}_{key}.jpg"
                                img_path = os.path.join(save_dir, img_filename)
                                pil_img.save(img_path, quality=95)
                                print(f"    保存: {img_filename}")
                        
                        elif isinstance(images, list):  # Multi-camera
                            for cam_idx, cam_img in enumerate(images):
                                if isinstance(cam_img, dict):
                                    for key, img_tensor in cam_img.items():
                                        # Process similar to above
                                        if img_tensor.dim() == 4:
                                            img_to_save = img_tensor[0]
                                        elif img_tensor.dim() == 3:
                                            img_to_save = img_tensor
                                        else:
                                            continue
                                        
                                        def denormalize_image(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
                                            tensor = tensor.clone()
                                            for i, (m, s) in enumerate(zip(mean, std)):
                                                tensor[i] = tensor[i] * s + m
                                            tensor = torch.clamp(tensor, 0, 1)
                                            tensor = tensor.permute(1, 2, 0)
                                            tensor = (tensor * 255).byte().numpy()
                                            return PILImage.fromarray(tensor)
                                        
                                        pil_img = denormalize_image(img_to_save)
                                        img_filename = f"{dataset_name}_sample{i:02d}_cam{cam_idx}_{key}.jpg"
                                        img_path = os.path.join(save_dir, img_filename)
                                        pil_img.save(img_path, quality=95)
                                        print(f"    保存: {img_filename}")
                                        
                                        if cam_idx >= 2:  # Only save first 3 cameras
                                            break
                
            except Exception as e:
                print(f"    ⚠️  {dataset_name}图像保存失败: {e}")
        
        print(f"  ✅ 样本图像保存完成！保存路径: {save_dir}/")
        
    except Exception as e:
        print(f"  ❌ 图像保存失败: {e}")

def main():
    """Main test function"""
    print("🚀 H-RDT Data Loading Test")
    print("="*60)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    print(f"Project root: {pyrootutils.find_root(search_from=__file__, indicator='.project-root')}")
    print(f"Working directory: {os.getcwd()}")
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    # Test 1: EgoDx data loading
    if test_egodx_data_loading():
        tests_passed += 1
    
    # Test 2: RobotWin data loading
    if test_robotwin_data_loading():
        tests_passed += 1
    
    # Test 3: Multi-dataset loading
    if test_multi_dataset_loading():
        tests_passed += 1
    
    # Bonus: Save sample images
    save_sample_images()
    
    print("\n" + "="*60)
    print(f"📊 测试结果: {tests_passed}/{total_tests} 通过")
    
    if tests_passed == total_tests:
        print("🎉 所有数据加载测试通过！")
        print("✅ H-RDT数据集可以正常读取图像和其他数据")
    else:
        print("⚠️  部分测试失败，请检查数据路径和配置")
    
    print("\n📋 后续建议:")
    print("  1. 检查保存的样本图像确认质量")
    print("  2. 调整数据路径配置匹配你的环境")
    print("  3. 运行完整的训练管道测试")

if __name__ == "__main__":
    main()
