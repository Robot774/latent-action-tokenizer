#!/usr/bin/env python3
"""
测试手部关键点可视化功能
Test Hand Keypoint Visualization
"""

import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

import os
import sys
import numpy as np
import torch
from pathlib import Path

# 添加可视化模块路径
sys.path.append('Moto_copy/hrdt')
from visualization.hand_keypoint_visualizer import HandKeypointVisualizer, visualize_egodx_sample
from datasets.pretrain.egodex_dataset import EgoDexDataset
from datasets.dataset import VLAConsumerDataset, MultiDataCollatorForVLAConsumerDataset


def test_basic_visualizer():
    """测试基础可视化器功能"""
    print("🧪 测试1: 基础可视化器功能")
    
    # 创建可视化器
    visualizer = HandKeypointVisualizer(image_width=640, image_height=480, fps=10)
    
    # 创建测试数据
    # 生成简单的48D动作数据 (未归一化)
    test_action = np.zeros(48)
    
    # 左手数据 (前24维)
    test_action[0:3] = [0.0, 0.0, 0.5]     # 左手腕位置
    test_action[3:9] = [1, 0, 0, 1, 0, 0]  # 左手腕旋转(6D)
    # 左手指尖位置 (相对分布)
    test_action[9:12] = [0.1, 0.0, 0.6]    # 拇指
    test_action[12:15] = [0.0, 0.1, 0.7]   # 食指
    test_action[15:18] = [0.0, 0.0, 0.8]   # 中指
    test_action[18:21] = [0.0, -0.1, 0.7]  # 无名指
    test_action[21:24] = [-0.1, -0.1, 0.6] # 小指
    
    # 右手数据 (后24维)
    test_action[24:27] = [0.0, 0.0, 0.5]   # 右手腕位置
    test_action[27:33] = [1, 0, 0, 1, 0, 0] # 右手腕旋转(6D)
    # 右手指尖位置
    test_action[33:36] = [-0.1, 0.0, 0.6]  # 拇指
    test_action[36:39] = [0.0, 0.1, 0.7]   # 食指
    test_action[39:42] = [0.0, 0.0, 0.8]   # 中指
    test_action[42:45] = [0.0, -0.1, 0.7]  # 无名指
    test_action[45:48] = [0.1, -0.1, 0.6]  # 小指
    
    # 创建测试相机参数
    camera_intrinsics = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0]
    ])
    
    camera_extrinsics = np.eye(4)  # 单位矩阵表示相机在世界坐标原点
    
    # 测试48D解析
    parsed = visualizer.parse_48d_action(test_action)
    print(f"   ✅ 48D解析成功:")
    print(f"      - 左手腕位置: {parsed['left_wrist_pos']}")
    print(f"      - 左手指尖数量: {parsed['left_fingertips'].shape}")
    print(f"      - 右手腕位置: {parsed['right_wrist_pos']}")
    print(f"      - 右手指尖数量: {parsed['right_fingertips'].shape}")
    
    # 测试3D到2D投影
    test_points_3d = np.array([[0, 0, 1], [0.1, 0, 1], [-0.1, 0, 1]])
    points_2d = visualizer.project_3d_to_2d(test_points_3d, camera_extrinsics, camera_intrinsics)
    print(f"   ✅ 3D到2D投影成功: {test_points_3d.shape} -> {points_2d.shape}")
    print(f"      - 投影结果: {points_2d}")
    
    # 测试单帧生成
    frame = visualizer.create_frame_with_keypoints(test_action, camera_extrinsics, camera_intrinsics)
    print(f"   ✅ 单帧生成成功: {frame.shape}")
    
    print("✅ 测试1完成\n")


def test_egodx_dataset_visualization():
    """测试使用EgoDx数据集进行可视化"""
    print("🧪 测试2: EgoDx数据集可视化")
    
    try:
        # 创建EgoDx数据集 (包含相机参数)
        config = {
            "common": {
                "img_history_size": 1,
                "chunk_size": 10
            }
        }
        
        dataset = EgoDexDataset(
            config=config,
            upsample_rate=5,
            val=False,
            use_precomp_lang_embed=True,
            include_camera_params=True,  # 启用相机参数
            data_root="/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex",
            stat_path="/root/workspace/chenby10@xiaopeng.com/Moto_copy/hrdt/datasets/pretrain/egodex_stat.json"
        )
        
        print(f"   ✅ 数据集创建成功, 样本数量: {len(dataset)}")
        
        # 获取一个样本
        sample = dataset.get_item(0)
        if sample is None:
            print("   ❌ 无法获取有效样本")
            return
            
        print(f"   ✅ 样本获取成功")
        print(f"      - 样本键: {list(sample.keys())}")
        print(f"      - 动作形状: {sample['actions'].shape}")
        
        if 'current_camera_extrinsics' in sample:
            print(f"      - 当前相机外参: {sample['current_camera_extrinsics'].shape}")
        if 'action_camera_extrinsics' in sample:
            print(f"      - 动作相机外参: {sample['action_camera_extrinsics'].shape}")
        if 'camera_intrinsics' in sample:
            print(f"      - 相机内参: {sample['camera_intrinsics'].shape}")
        
        # 测试可视化
        output_path = "/root/workspace/chenby10@xiaopeng.com/test_hand_visualization.mp4"
        stat_path = "/root/workspace/chenby10@xiaopeng.com/Moto_copy/hrdt/datasets/pretrain/egodex_stat.json"
        
        print(f"   🎬 开始生成可视化视频...")
        visualize_egodx_sample(
            sample_data=sample,
            output_path=output_path,
            stat_path=stat_path,
            image_size=(640, 480),
            fps=10
        )
        
        # 检查输出文件
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
            print(f"   ✅ 视频生成成功: {output_path}")
            print(f"      - 文件大小: {file_size:.2f} MB")
        else:
            print(f"   ❌ 视频文件未生成: {output_path}")
            
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ 测试2完成\n")


def test_multi_dataset_visualization():
    """测试多数据集的可视化"""
    print("🧪 测试3: 多数据集可视化")
    
    try:
        # 创建VLA数据集
        config = {
            "common": {
                "img_history_size": 1,
                "chunk_size": 10
            }
        }
        
        # 测试EgoDx数据集
        egodx_dataset = VLAConsumerDataset(
            config=config,
            image_transform=None,  # 暂时不处理图像
            num_cameras=1,
            dataset_name="egodex"
        )
        
        print(f"   ✅ EgoDx VLA数据集创建成功")
        
        # 创建collator
        collator = MultiDataCollatorForVLAConsumerDataset(
            unified_action_dim=48,
            use_precomp_lang_embed=True
        )
        
        # 获取批量样本
        samples = [egodx_dataset[0], egodx_dataset[1]]
        batch = collator(samples)
        
        print(f"   ✅ 批量数据处理成功")
        print(f"      - 批量大小: {batch['actions'].shape[0]}")
        print(f"      - 动作形状: {batch['actions'].shape}")
        
        if 'action_camera_extrinsics' in batch:
            print(f"      - 相机外参形状: {batch['action_camera_extrinsics'].shape}")
        if 'camera_intrinsics' in batch:
            print(f"      - 相机内参形状: {batch['camera_intrinsics'].shape}")
        
        # 可视化第一个样本
        first_sample = {
            'actions': batch['actions'][0],
            'action_camera_extrinsics': batch['action_camera_extrinsics'][0],
            'camera_intrinsics': batch['camera_intrinsics'][0]
        }
        
        output_path = "/root/workspace/chenby10@xiaopeng.com/test_multi_visualization.mp4"
        stat_path = "/root/workspace/chenby10@xiaopeng.com/Moto_copy/hrdt/datasets/pretrain/egodex_stat.json"
        
        print(f"   🎬 开始生成多数据集可视化...")
        visualize_egodx_sample(
            sample_data=first_sample,
            output_path=output_path,
            stat_path=stat_path,
            image_size=(640, 480),
            fps=15
        )
        
        if os.path.exists(output_path):
            print(f"   ✅ 多数据集可视化成功: {output_path}")
        else:
            print(f"   ❌ 多数据集可视化失败")
            
    except Exception as e:
        print(f"   ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("✅ 测试3完成\n")


def main():
    """主测试函数"""
    print("🚀 开始手部关键点可视化测试")
    print("=" * 60)
    
    # 基础功能测试
    test_basic_visualizer()
    
    # EgoDx数据集测试
    test_egodx_dataset_visualization()
    
    # 多数据集测试
    test_multi_dataset_visualization()
    
    print("=" * 60)
    print("🎉 所有测试完成!")
    print("\n📁 输出文件:")
    
    output_files = [
        "/root/workspace/chenby10@xiaopeng.com/test_hand_visualization.mp4",
        "/root/workspace/chenby10@xiaopeng.com/test_multi_visualization.mp4"
    ]
    
    for file_path in output_files:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   ✅ {file_path} ({file_size:.2f} MB)")
        else:
            print(f"   ❌ {file_path} (未生成)")


if __name__ == "__main__":
    main()
