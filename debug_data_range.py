#!/usr/bin/env python3
"""
Debug script to monitor data ranges throughout the visualization pipeline
"""

import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

import torch
import numpy as np
from common.data.data_utils import load_dataset
from common.processors.preprocessor_utils import get_rgb_preprocessor
from hrdt.datasets.dataset import MultiDataCollatorForVLAConsumerDataset
from torch.utils.data import DataLoader
import omegaconf

def print_tensor_stats(tensor, name):
    """Print detailed statistics of a tensor"""
    if isinstance(tensor, dict):
        print(f"\n📊 {name} (dict):")
        for key, val in tensor.items():
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}, min={val.min():.4f}, max={val.max():.4f}, mean={val.mean():.4f}")
    else:
        print(f"\n📊 {name}: shape={tensor.shape}, dtype={tensor.dtype}, min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")

def debug_data_pipeline():
    """Debug the complete data pipeline"""
    
    print("🔍 Starting data range debugging...")
    
    # 1. Load dataset
    print("\n" + "="*50)
    print("1️⃣ Loading dataset...")
    
    dataset_config_path = "/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer/configs/data/hrdt_robotwin.yaml"
    extra_data_config = {
        'sequence_length': 1,
        'do_extract_future_frames': True,
        'do_extract_action': False
    }
    
    train_dataset, eval_dataset = load_dataset(dataset_config_path, extra_data_config)
    print(f"✅ Dataset loaded: train={len(train_dataset)}, eval={len(eval_dataset)}")
    
    # 2. Create dataloader
    collator = MultiDataCollatorForVLAConsumerDataset(
        unified_action_dim=48, 
        use_precomp_lang_embed=True
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False
    )
    
    # 3. Get one batch
    print("\n" + "="*50)
    print("2️⃣ Getting batch from dataloader...")
    
    batch = next(iter(eval_dataloader))
    
    # Check raw batch data
    if 'rgb_initial' in batch:
        print_tensor_stats(batch['rgb_initial'], "Raw rgb_initial")
    if 'rgb_future' in batch:
        print_tensor_stats(batch['rgb_future'], "Raw rgb_future")
    
    # 4. Create RGB preprocessor
    print("\n" + "="*50)
    print("3️⃣ Creating RGB preprocessor...")
    
    rgb_preprocessor_config = {
        'model_vision_type': 'mae',
        'vision_aug_config': {
            'do_random_resized_crop': False,
            'do_random_shift': True
        }
    }
    
    rgb_preprocessor = get_rgb_preprocessor(**rgb_preprocessor_config)
    
    # Print preprocessor parameters
    print(f"RGB mean: {rgb_preprocessor.rgb_mean.flatten().tolist()}")
    print(f"RGB std: {rgb_preprocessor.rgb_std.flatten().tolist()}")
    
    # 5. Prepare data like in trainer
    print("\n" + "="*50)
    print("4️⃣ Processing data like in trainer...")
    
    # Concatenate initial and future frames
    orig_rgb_seq = torch.cat([batch['rgb_initial'], batch['rgb_future']], dim=1)  # (b, 2, c, h, w)
    print_tensor_stats(orig_rgb_seq, "Concatenated orig_rgb_seq")
    
    # Apply preprocessing
    rgb_seq = rgb_preprocessor(orig_rgb_seq, train=True)
    print_tensor_stats(rgb_seq, "After preprocessing (rgb_seq)")
    
    # 6. Simulate post-processing
    print("\n" + "="*50)
    print("5️⃣ Simulating post-processing...")
    
    # Post-process back (like in visualization)
    post_processed = rgb_preprocessor.post_process(rgb_seq)
    print_tensor_stats(post_processed, "After post_process")
    
    # 7. Check individual frames for visualization
    print("\n" + "="*50)
    print("6️⃣ Checking individual frames for visualization...")
    
    for i in range(min(2, post_processed.shape[0])):  # Check first 2 samples
        initial_frame = post_processed[i, 0]  # (c, h, w)
        next_frame = post_processed[i, 1]     # (c, h, w)
        
        print_tensor_stats(initial_frame, f"Sample {i} - initial_frame")
        print_tensor_stats(next_frame, f"Sample {i} - next_frame")
        
        # Check if values are in valid range for ToPILImage
        if initial_frame.min() < 0 or initial_frame.max() > 1:
            print(f"⚠️  Sample {i} initial_frame values outside [0,1] range!")
        if next_frame.min() < 0 or next_frame.max() > 1:
            print(f"⚠️  Sample {i} next_frame values outside [0,1] range!")
    
    # 8. Test different normalization parameters
    print("\n" + "="*50)
    print("7️⃣ Testing alternative normalization...")
    
    # Test with [0.5, 0.5, 0.5] normalization
    alt_rgb_preprocessor_config = {
        'model_vision_type': 'theia',  # Uses [0.5, 0.5, 0.5] normalization
        'vision_aug_config': {
            'do_random_resized_crop': False,
            'do_random_shift': True
        }
    }
    
    alt_rgb_preprocessor = get_rgb_preprocessor(**alt_rgb_preprocessor_config)
    print(f"Alternative RGB mean: {alt_rgb_preprocessor.rgb_mean.flatten().tolist()}")
    print(f"Alternative RGB std: {alt_rgb_preprocessor.rgb_std.flatten().tolist()}")
    
    # Process with alternative normalization
    alt_rgb_seq = alt_rgb_preprocessor(orig_rgb_seq, train=True)
    alt_post_processed = alt_rgb_preprocessor.post_process(alt_rgb_seq)
    
    print_tensor_stats(alt_post_processed, "Alternative normalization result")
    
    # 9. Save sample images for visual inspection
    print("\n" + "="*50)
    print("8️⃣ Saving sample images...")
    
    import torchvision.transforms as T
    from PIL import Image
    import os
    
    save_dir = "/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/debug_output"
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(2, post_processed.shape[0])):
        # Original normalization
        initial_img = T.ToPILImage()(post_processed[i, 0])
        next_img = T.ToPILImage()(post_processed[i, 1])
        
        initial_img.save(f"{save_dir}/sample_{i}_initial_imagenet_norm.png")
        next_img.save(f"{save_dir}/sample_{i}_next_imagenet_norm.png")
        
        # Alternative normalization
        alt_initial_img = T.ToPILImage()(alt_post_processed[i, 0])
        alt_next_img = T.ToPILImage()(alt_post_processed[i, 1])
        
        alt_initial_img.save(f"{save_dir}/sample_{i}_initial_simple_norm.png")
        alt_next_img.save(f"{save_dir}/sample_{i}_next_simple_norm.png")
    
    print(f"✅ Sample images saved to {save_dir}")
    
    # 10. Summary
    print("\n" + "="*50)
    print("📋 SUMMARY")
    print("="*50)
    
    print(f"Original data range: [{orig_rgb_seq.min():.4f}, {orig_rgb_seq.max():.4f}]")
    print(f"After ImageNet norm: [{post_processed.min():.4f}, {post_processed.max():.4f}]")
    print(f"After simple norm: [{alt_post_processed.min():.4f}, {alt_post_processed.max():.4f}]")
    
    if post_processed.min() < 0 or post_processed.max() > 1:
        print("🚨 ImageNet normalization produces values outside [0,1] - this will cause color issues!")
    else:
        print("✅ ImageNet normalization produces valid [0,1] range")
        
    if alt_post_processed.min() < 0 or alt_post_processed.max() > 1:
        print("🚨 Simple normalization produces values outside [0,1]")
    else:
        print("✅ Simple normalization produces valid [0,1] range")

if __name__ == "__main__":
    debug_data_pipeline()
