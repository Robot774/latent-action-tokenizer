#!/usr/bin/env python3
"""
EgoDex dataset loader
Implements 48-dimensional hand action representation, single-view images and language embedding data loading
"""

import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

import h5py
import numpy as np
import torch
import os
import cv2
from pathlib import Path
import warnings
import random
import json
warnings.filterwarnings("ignore")


class EgoDexDataset:
    """EgoDex dataset loader"""
    
    def __init__(self, 
                 data_root=None, 
                 config=None,
                 upsample_rate=3,
                 val=False,
                 use_precomp_lang_embed=True,
                 stat_path=None,
                 include_camera_params=True):
        """
        Args:
            data_root: Data root directory (e.g., "/share/hongzhe/datasets/egodex")
            config: Configuration dictionary
            upsample_rate: Temporal data upsampling rate (frame sampling interval)
            val: Whether it's validation set (True for test, False for train)
            use_precomp_lang_embed: Whether to use precomputed language embeddings
            stat_path: Statistics file path (default: datasets/pretrain/egodex_stat.json)
            include_camera_params: Whether to include camera parameters paired with actions
        """
        self.DATASET_NAME = "egodex"
        self.data_root = Path(data_root)
        self.config = config
        self.upsample_rate = upsample_rate
        self.val = val
        self.use_precomp_lang_embed = use_precomp_lang_embed
        self.include_camera_params = include_camera_params
        
        if config:
            self.chunk_size = config['common']['action_chunk_size']
            self.state_dim = config['common']['action_dim']
            self.img_history_size = config['common']['img_history_size']
        else:
            self.chunk_size = 16
            self.state_dim = 48
            self.img_history_size = 1
        
        # Set default stat_path if not provided (relative to this file)
        if stat_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            stat_path = os.path.join(current_dir, 'egodex_stat.json')
        
        # Load data file list
        self.data_files = self._load_file_list()
        split_name = "test" if self.val else "train"
        print(f"Loaded {len(self.data_files)} {split_name} data files")
        
        # Load statistics for normalization
        self.action_min = None
        self.action_max = None
        if os.path.exists(stat_path):
            with open(stat_path, 'r') as f:
                stat = json.load(f)
            if 'egodex' in stat:
                self.action_min = np.array(stat['egodex']['min'])
                self.action_max = np.array(stat['egodex']['max'])
    
    def get_dataset_name(self):
        """Return dataset name"""
        return self.DATASET_NAME
    
    def _load_file_list(self):
        """Load data file list"""
        data_files = []
        
        if not self.val:
            # Training set: part1-part5 + extra
            for part in ['part1', 'part2', 'part3', 'part4', 'part5', 'extra']:
                part_dir = self.data_root / part
                if part_dir.exists():
                    data_files.extend(self._scan_directory(part_dir))
        else:
            # Test set: test
            test_dir = self.data_root / 'test'
            if test_dir.exists():
                data_files.extend(self._scan_directory(test_dir))
        
        return data_files
    
    def _scan_directory(self, directory):
        """Scan files in directory"""
        files = []
        for task_dir in directory.iterdir():
            if task_dir.is_dir():
                # Collect all triplets: hdf5, mp4, pt
                hdf5_files = list(task_dir.glob('*.hdf5'))
                for hdf5_file in hdf5_files:
                    file_index = hdf5_file.stem  # Get filename without extension
                    mp4_file = task_dir / f"{file_index}.mp4"
                    pt_file = task_dir / f"{file_index}.pt"
                    
                    # Ensure all required files exist
                    if (hdf5_file.exists() and mp4_file.exists() and 
                        pt_file.exists()):
                        files.append({
                            'hdf5': hdf5_file,
                            'mp4': mp4_file,
                            'pt': pt_file,

                            'task': task_dir.name,
                            'file_index': file_index
                        })
        return files
    
    def construct_48d_action(self, hdf5_file, frame_indices):
        """
        Directly extract precomputed 48-dimensional hand action representation
        
        Args:
            hdf5_file: HDF5 file object
            frame_indices: List of frame indices to extract
            
        Returns:
            actions: (T, 48) action sequence
        """
        if 'actions_48d' not in hdf5_file:
            raise ValueError("Missing precomputed actions_48d data in HDF5 file, please run precompute_48d_actions.py first")
        
        # Directly read precomputed 48-dimensional action data
        precomputed_actions = hdf5_file['actions_48d'][:]
        
        # Extract actions for specified frames
        selected_actions = precomputed_actions[frame_indices]
        
        return selected_actions.astype(np.float32)
    
    def construct_paired_action_camera(self, hdf5_file, frame_indices):
        """
        Extract paired 48D actions and camera parameters for the same frame indices
        
        Args:
            hdf5_file: HDF5 file object
            frame_indices: List of frame indices to extract
            
        Returns:
            actions: (T, 48) action sequence
            cam_extrinsics: (T, 4, 4) camera extrinsics for each action frame
            cam_intrinsics: (3, 3) camera intrinsics (constant)
        """
        # Extract actions
        actions = self.construct_48d_action(hdf5_file, frame_indices)
        
        # Extract corresponding camera parameters
        try:
            # Camera extrinsics - one for each action frame
            cam_extrinsics = []
            for frame_idx in frame_indices:
                cam_ext = hdf5_file['/transforms/camera'][frame_idx]  # (4, 4)
                cam_extrinsics.append(cam_ext)
            cam_extrinsics = np.array(cam_extrinsics).astype(np.float32)
            
            # Camera intrinsics - constant for all frames  
            cam_intrinsics = hdf5_file['/camera/intrinsic'][:].astype(np.float32)  # (3, 3)
            
            # Verify dimensions match
            assert actions.shape[0] == cam_extrinsics.shape[0], \
                f"Action frames ({actions.shape[0]}) and camera frames ({cam_extrinsics.shape[0]}) mismatch"
            
            return actions, cam_extrinsics, cam_intrinsics
            
        except KeyError as e:
            print(f"Warning: Camera parameter {e} not found in HDF5 file")
            # Return default camera parameters if not available
            num_frames = len(frame_indices)
            default_extrinsics = np.eye(4)[None].repeat(num_frames, axis=0).astype(np.float32)
            default_intrinsics = np.array([
                [736.6339, 0., 960.],
                [0., 736.6339, 540.], 
                [0., 0., 1.]
            ], dtype=np.float32)
            return actions, default_extrinsics, default_intrinsics
    
    def parse_img_data(self, mp4_path, idx):
        """
        Load image frames following cvpr_real_dataset.py sampling logic
        
        Args:
            mp4_path: MP4 file path
            idx: Current frame index
            
        Returns:
            frames: (img_history_size, H, W, 3) image frames
        """
        cap = cv2.VideoCapture(str(mp4_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate sampling range following cvpr_real_dataset.py logic
        start_i = max(idx - self.img_history_size * self.upsample_rate + 1, 0)
        num_frames = (idx - start_i) // self.upsample_rate + 1
        
        frames = []
        
        try:
            for i, frame_idx in enumerate(range(start_i, idx + 1, self.upsample_rate)):
                if frame_idx < total_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        # BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                    else:
                        print(f"Warning: Not enough frames in {mp4_path}")
                        break
                else:
                    # If frame index exceeds total frames, use last valid frame
                    print(f"Warning: Frame index exceeds total frames in {mp4_path}")
                    break
        except Exception as e:
            print(f"Error loading image frames: {e}")
        
        cap.release()
        
        # Convert to numpy array
        if frames:
            frames = np.array(frames)
        else:
            frames = np.zeros((1, 1080, 1920, 3), dtype=np.uint8)
        
        # Pad if necessary (following cvpr_real_dataset.py logic)
        if frames.shape[0] < self.img_history_size:
            pad_frames = np.repeat(frames[:1], self.img_history_size - frames.shape[0], axis=0)
            frames = np.concatenate([pad_frames, frames], axis=0)
        
        return frames
    
    def parse_img_data_decord(self, mp4_path, idx):
        """
        Load image frames using decord with single thread (alternative implementation)
        
        Args:
            mp4_path: MP4 file path
            idx: Current frame index
            
        Returns:
            frames: (img_history_size, H, W, 3) image frames
        """
        from decord import VideoReader, cpu
        
        # Calculate sampling range following cvpr_real_dataset.py logic
        start_i = max(idx - self.img_history_size * self.upsample_rate + 1, 0)
        
        frames = []
        actual_frame_indices = []  # 记录实际读取的帧索引
        
        try:
            # Initialize VideoReader with single thread
            vr = VideoReader(str(mp4_path), ctx=cpu(0), num_threads=1)
            total_frames = len(vr)
            
            # Sample frames according to the logic
            for i, frame_idx in enumerate(range(start_i, idx + 1, self.upsample_rate)):
                if frame_idx < total_frames:
                    # Read single frame using decord
                    frame = vr[frame_idx].asnumpy()  # Returns RGB format directly
                    frames.append(frame)
                    actual_frame_indices.append(frame_idx)
                else:
                    # If frame index exceeds total frames, use last valid frame
                    print(f"Warning: Frame index {frame_idx} exceeds total frames {total_frames} in {mp4_path}")
                    break
            
            # 调试信息：显示实际读取的帧索引（已注释）
            # print(f"parse_img_data_decord: target_idx={idx}, start_i={start_i}, actual_frames={actual_frame_indices}")
                    
        except Exception as e:
            print(f"Error loading image frames with decord: {e}")
            # Fallback to OpenCV implementation
            return self.parse_img_data(mp4_path, idx)
        
        # Convert to numpy array
        if frames:
            frames = np.array(frames)
        else:
            frames = np.zeros((1, 1080, 1920, 3), dtype=np.uint8)
        
        # Pad if necessary (following cvpr_real_dataset.py logic)
        if frames.shape[0] < self.img_history_size:
            pad_frames = np.repeat(frames[:1], self.img_history_size - frames.shape[0], axis=0)
            frames = np.concatenate([pad_frames, frames], axis=0)
        
        return frames
    
    def __len__(self):
        return len(self.data_files)
    
    def get_item(self, idx=None):
        """
        Get a data sample
        
        Returns:
            Data dictionary containing all required fields
        """
        if idx is None:
            idx = random.randint(0, len(self.data_files) - 1)
        
        file_info = self.data_files[idx % len(self.data_files)]
        
        try:
            # Load HDF5 data
            with h5py.File(file_info['hdf5'], 'r') as f:
                # Get total number of frames
                transforms_group = f['transforms']
                total_frames = list(transforms_group.values())[0].shape[0]
                
                # Calculate random index following cvpr_real_dataset.py logic
                max_index = total_frames - 2
                if max_index < 0:
                    print(f"Warning: Not enough frames in {file_info['hdf5']}")
                    return None
                
                # Random index for sampling
                index = random.randint(0, max_index)
                
                # Prepare frame indices for actions and camera parameters
                # Current frame
                current_frame_indices = [index]
                
                # Future action sequence frame indices
                action_end = min(index + self.chunk_size * self.upsample_rate, max_index + 1)
                action_indices = list(range(index + 1, action_end + 1, self.upsample_rate))
                
                # If not enough action frames, repeat the last one
                while len(action_indices) < self.chunk_size:
                    action_indices.append(action_indices[-1] if action_indices else index + 1)
                action_indices = action_indices[:self.chunk_size]
                
                # Extract paired actions and camera parameters
                if self.include_camera_params:
                    # Use paired extraction for current state
                    current_action, current_cam_ext, cam_int = self.construct_paired_action_camera(f, current_frame_indices)
                    # Use paired extraction for action sequence
                    actions, action_cam_exts, _ = self.construct_paired_action_camera(f, action_indices)
                else:
                    # Original logic without camera parameters
                    current_action = self.construct_48d_action(f, current_frame_indices)
                    actions = self.construct_48d_action(f, action_indices)
                    current_cam_ext = None
                    action_cam_exts = None
                    cam_int = None
                
                # Pad actions if necessary
                if actions.shape[0] < self.chunk_size:
                    last_action = actions[-1:] if len(actions) > 0 else current_action
                    padding = np.repeat(last_action, self.chunk_size - actions.shape[0], axis=0)
                    actions = np.concatenate([actions, padding], axis=0)
                    
                    # Pad camera extrinsics correspondingly if included
                    if self.include_camera_params and action_cam_exts is not None:
                        last_cam_ext = action_cam_exts[-1:] if len(action_cam_exts) > 0 else current_cam_ext
                        cam_padding = np.repeat(last_cam_ext, self.chunk_size - action_cam_exts.shape[0], axis=0)
                        action_cam_exts = np.concatenate([action_cam_exts, cam_padding], axis=0)
            
            # Normalize actions
            if self.action_min is not None and self.action_max is not None:
                current_action = (current_action - self.action_min) / (self.action_max - self.action_min) * 2 - 1
                current_action = np.clip(current_action, -1, 1)
                actions = (actions - self.action_min) / (self.action_max - self.action_min) * 2 - 1
                actions = np.clip(actions, -1, 1)
            
            # Load single-view image frames using batch reading for better performance
            # 首帧：历史序列（现有逻辑）
            frames_history = self.parse_img_data_decord(file_info['mp4'], index)
            frames_history = frames_history[-self.img_history_size:]
            
            # 尾帧：未来序列（新增）
            future_index = action_indices[-1]  # 最后一个动作对应的帧
            # print(f"Debug: future_index={future_index}, current_idx={index}, diff={future_index-index}")
            # print(f"Debug: action_indices={action_indices}")
            # print(f"Debug: img_history_size={self.img_history_size}, upsample_rate={self.upsample_rate}")
            
            frames_future = self.parse_img_data_decord(file_info['mp4'], future_index)
            frames_future = frames_future[-self.img_history_size:]
                        
            # Load language embedding
            lang_embed_path = file_info['pt']
            
            # Prepare return dictionary
            return_dict = {
                'states': current_action,  # (1, 48)
                'actions': actions,  # (chunk_size, 48)
                'action_norm': np.ones_like(actions),  # Action indicator
                'current_images': [frames_history],  # [(img_history_size, H, W, 3)] 首帧历史序列
                'future_images': [frames_future],    # [(img_history_size, H, W, 3)] 尾帧未来序列
                'current_images_mask': [np.ones(self.img_history_size, dtype=bool)],  # Image mask
                'future_images_mask': [np.ones(self.img_history_size, dtype=bool)],   # Future image mask
                'instruction': str(lang_embed_path),  # Language embedding file path
                'dataset_name': self.DATASET_NAME,
                'task': file_info['task'],
                'file_info': {
                    'hdf5_path': str(file_info['hdf5']),
                    'mp4_path': str(file_info['mp4']),
                    'pt_path': str(file_info['pt']),
                    'total_frames': total_frames,
                    'selected_index': index,
                    'action_indices': action_indices,
                    'future_index': future_index  # 新增：记录未来帧索引
                }
            }
            
            # Add camera parameters if included
            if self.include_camera_params:
                return_dict.update({
                    'current_camera_extrinsics': current_cam_ext,  # (1, 4, 4) 当前帧相机外参
                    'action_camera_extrinsics': action_cam_exts,   # (chunk_size, 4, 4) 动作序列对应的相机外参
                    'camera_intrinsics': cam_int,                  # (3, 3) 相机内参
                })
                
                # Add camera info to file_info for debugging
                return_dict['file_info']['include_camera_params'] = True
                return_dict['file_info']['camera_ext_shape'] = action_cam_exts.shape if action_cam_exts is not None else None
            
            return return_dict
            
        except Exception as e:
            print(f"Error loading data {file_info['hdf5']}: {e}")
            return None

    def __getitem__(self, idx):
        """PyTorch Dataset interface"""
        return self.get_item(idx)


# if __name__ == "__main__":
#     # Test dataset without camera parameters
#     print("=== Testing without camera parameters ===")
#     dataset = EgoDexDataset(
#         data_root="/share/hongzhe/datasets/egodex",
#         val=False,
#         upsample_rate=3,
#         include_camera_params=False
#     )
    
#     print(f"Dataset size: {len(dataset)}")
#     sample = dataset.get_item(0)
#     print("Sample data structure (without camera):")
#     for key, value in sample.items():
#         if isinstance(value, np.ndarray):
#             print(f"  {key}: {value.shape}")
#         else:
#             print(f"  {key}: {type(value)}")
    
#     print("\n=== Testing with camera parameters ===")
#     # Test dataset with camera parameters
#     dataset_with_camera = EgoDexDataset(
#         data_root="/share/hongzhe/datasets/egodex",
#         val=False,
#         upsample_rate=3,
#         include_camera_params=True
#     )
    
#     sample_with_camera = dataset_with_camera.get_item(0)
#     print("Sample data structure (with camera):")
#     for key, value in sample_with_camera.items():
#         if isinstance(value, np.ndarray):
#             print(f"  {key}: {value.shape}")
#         else:
#             print(f"  {key}: {type(value)}")
    
#     # Verify action-camera pairing
#     if 'actions' in sample_with_camera and 'action_camera_extrinsics' in sample_with_camera:
#         actions_shape = sample_with_camera['actions'].shape
#         camera_shape = sample_with_camera['action_camera_extrinsics'].shape
#         print(f"\nAction-Camera pairing verification:")
#         print(f"  Actions shape: {actions_shape}")
#         print(f"  Camera extrinsics shape: {camera_shape}")
#         print(f"  Paired correctly: {actions_shape[0] == camera_shape[0]}")
