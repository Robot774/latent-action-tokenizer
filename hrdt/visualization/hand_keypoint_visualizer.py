#!/usr/bin/env python3
"""
Hand Keypoint Visualizer for EgoDex Dataset
严格按照 visualize_2d.py 的实现方式，只适配不同的输入格式
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import torch


class HandKeypointVisualizer:
    """手部关键点可视化器 - 基于 visualize_2d.py"""
    
    def __init__(self, fps: int = 30):
        """
        初始化可视化器
        
        Args:
            fps: 视频帧率
        """
        self.fps = fps
        
        # 严格按照visualize_2d.py的颜色映射
        self.finger_colors = {
            'little': (0, 152, 191),    # light blue
            'ring': (173, 255, 47),     # green yellow  
            'middle': (230, 245, 250),  # pale torquoise
            'index': (255, 99, 71),     # tomato
            'thumb': (238, 130, 238)    # violet
        }
        
    def denormalize_actions(self, 
                          normalized_actions: np.ndarray, 
                          stat_path: str) -> np.ndarray:
        """反归一化动作数据"""
        with open(stat_path, 'r') as f:
            stats = json.load(f)
        
        action_min = np.array(stats['egodex']['min'])
        action_max = np.array(stats['egodex']['max'])
        
        # 反归一化: x_real = (x_norm + 1) / 2 * (max - min) + min
        denormalized = (normalized_actions + 1) / 2 * (action_max - action_min) + action_min
        
        return denormalized
        
    def parse_48d_to_transforms(self, action_48d: np.ndarray) -> Dict[str, np.ndarray]:
        """
        将48D动作数据转换为变换矩阵格式 (模拟visualize_2d.py的tfs格式)
        
        Args:
            action_48d: (48,) 48维动作向量
            
        Returns:
            transforms: 包含手部和指尖变换矩阵的字典
        """
        transforms = {}
        
        # 解析48D数据
        # 左手数据 (前24维)
        left_data = action_48d[:24]
        left_wrist_pos = left_data[:3]           # 位置 (3D)
        left_wrist_rot = left_data[3:9]          # 旋转 6D (暂不使用)
        left_fingertips = left_data[9:24].reshape(5, 3)  # 5个指尖 (5x3D)
        
        # 右手数据 (后24维)  
        right_data = action_48d[24:48]
        right_wrist_pos = right_data[:3]         # 位置 (3D)
        right_wrist_rot = right_data[3:9]        # 旋转 6D (暂不使用)
        right_fingertips = right_data[9:24].reshape(5, 3)  # 5个指尖 (5x3D)
        
        # 创建4x4变换矩阵 (按照visualize_2d.py的格式)
        def create_transform_matrix(position):
            """创建4x4变换矩阵，位置信息在最后一列"""
            tf = np.eye(4)
            tf[:3, 3] = position  # 位置存储在最后一列 (和visualize_2d.py一致)
            return tf
        
        # 左手变换矩阵
        transforms['leftHand'] = create_transform_matrix(left_wrist_pos)
        transforms['leftThumbTip'] = create_transform_matrix(left_fingertips[0])
        transforms['leftIndexFingerTip'] = create_transform_matrix(left_fingertips[1])
        transforms['leftMiddleFingerTip'] = create_transform_matrix(left_fingertips[2])
        transforms['leftRingFingerTip'] = create_transform_matrix(left_fingertips[3])
        transforms['leftLittleFingerTip'] = create_transform_matrix(left_fingertips[4])
        
        # 右手变换矩阵
        transforms['rightHand'] = create_transform_matrix(right_wrist_pos)
        transforms['rightThumbTip'] = create_transform_matrix(right_fingertips[0])
        transforms['rightIndexFingerTip'] = create_transform_matrix(right_fingertips[1])
        transforms['rightMiddleFingerTip'] = create_transform_matrix(right_fingertips[2])
        transforms['rightRingFingerTip'] = create_transform_matrix(right_fingertips[3])
        transforms['rightLittleFingerTip'] = create_transform_matrix(right_fingertips[4])
        
        return transforms
        
    def convert_to_camera_frame(self, transforms_dict: Dict[str, np.ndarray], 
                               cam_ext: np.ndarray) -> Dict[str, np.ndarray]:
        """
        严格按照visualize_2d.py的convert_to_camera_frame实现
        
        Args:
            transforms_dict: 变换矩阵字典
            cam_ext: 相机外参 (4, 4)
            
        Returns:
            transforms_in_cam: 相机坐标系下的变换矩阵字典
        """
        # 严格按照 utils/data_utils.py: return np.linalg.inv(cam_ext)[None] @ tfs
        world_to_cam = np.linalg.inv(cam_ext)
        
        transforms_in_cam = {}
        for name, tf in transforms_dict.items():
            # 应用相机变换
            transforms_in_cam[name] = world_to_cam @ tf
            
        return transforms_in_cam
        
    def draw_line(self, pointa: np.ndarray, pointb: np.ndarray, 
                  image: np.ndarray, intrinsic: np.ndarray, 
                  color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 5):
        """
        严格按照visualize_2d.py的draw_line实现
        """
        # 严格按照原版：project 3d points into 2d
        pointa2, _ = cv2.projectPoints(pointa, np.eye(3), np.zeros(3), intrinsic, distCoeffs=np.zeros(5))  
        pointb2, _ = cv2.projectPoints(pointb, np.eye(3), np.zeros(3), intrinsic, distCoeffs=np.zeros(5))  
        pointa2 = pointa2.squeeze()
        pointb2 = pointb2.squeeze()

        # 严格按照原版：don't draw if the line is out of bounds
        H, W, _ = image.shape
        if (pointb2[0] < 0 and pointa2[0] > W) or (pointb2[1] < 0 and pointa2[1] > H) or (pointa2[0] < 0 and pointb2[0] > W) or (pointa2[1] < 0 and pointb2[1] > H):
            return 

        # 严格按照原版：draws a line in-place
        cv2.line(image, pointa2.astype(int), pointb2.astype(int), color=color, thickness=thickness)
        cv2.circle(image, (int(pointa2[0]), int(pointa2[1])), 15, color, -1)
        cv2.circle(image, (int(pointb2[0]), int(pointb2[1])), 15, color, -1)

    def draw_line_sequence(self, points_list: List[np.ndarray], 
                          image: np.ndarray, intrinsic: np.ndarray, 
                          color: Tuple[int, int, int] = (0, 255, 0)):
        """
        严格按照visualize_2d.py的draw_line_sequence实现
        """
        # 严格按照原版：draw a sequence of lines in-place
        ptm = points_list[0]
        for pt in points_list[1:]:
            self.draw_line(ptm, pt, image, intrinsic, color)
            ptm = pt

    def get_finger_pts(self, finger_names: List[str], 
                      transforms_in_cam: Dict[str, np.ndarray], 
                      hand_name: str) -> List[np.ndarray]:
        """
        严格按照visualize_2d.py的get_finger_pts实现
        
        Args:
            finger_names: 指尖名称列表 (如 ['rightThumbTip'])
            transforms_in_cam: 相机坐标系下的变换矩阵
            hand_name: 手腕名称 (如 'rightHand')
        """
        # 严格按照原版：grab 3D position from SE(3) pose
        finger_points = [transforms_in_cam[hand_name][:3, 3]]  # 手腕位置
        for tfname in finger_names:
            finger_points.append(transforms_in_cam[tfname][:3, 3])  # 指尖位置

        return finger_points
        
    def draw_hand(self, hand_dict: Dict[str, List[str]], 
                  transforms_in_cam: Dict[str, np.ndarray], 
                  cam_img: np.ndarray, cam_int: np.ndarray, 
                  right: bool = True):
        """
        严格按照visualize_2d.py的draw_hand实现
        """
        hand_name = 'rightHand' if right else 'leftHand'
        
        # 严格按照原版：draw fingers
        for finger in ['little', 'ring', 'middle', 'index', 'thumb']:  # roughly stack lines so closer fingers are drawn on top
            if finger in hand_dict:
                points = self.get_finger_pts(hand_dict[finger], transforms_in_cam, hand_name)
                self.draw_line_sequence(points, cam_img, cam_int,
                                      color=self.finger_colors[finger])
        
    def create_frame_with_keypoints(self, 
                                  action_48d: np.ndarray,
                                  camera_extrinsics: np.ndarray,
                                  camera_intrinsics: np.ndarray,
                                  image_height: int = 1080,
                                  image_width: int = 1920,
                                  background_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        创建包含手部关键点的单帧图像 (按照visualize_2d.py的主流程)
        """
        # 创建背景图像
        cam_img = np.full((image_height, image_width, 3), background_color, dtype=np.uint8)
        
        # 第一步：将48D转换为变换矩阵格式
        transforms = self.parse_48d_to_transforms(action_48d)
        
        # 第二步：转换到相机坐标系 (严格按照visualize_2d.py)
        transforms_in_cam = self.convert_to_camera_frame(transforms, camera_extrinsics)
        
        # 第三步：定义手指映射 (严格按照visualize_2d.py)
        right_dict = {
            'index': ['rightIndexFingerTip'], 
            'thumb': ['rightThumbTip'], 
            'middle': ['rightMiddleFingerTip'], 
            'ring': ['rightRingFingerTip'], 
            'little': ['rightLittleFingerTip']
        }
        left_dict = {
            'index': ['leftIndexFingerTip'], 
            'thumb': ['leftThumbTip'], 
            'middle': ['leftMiddleFingerTip'], 
            'ring': ['leftRingFingerTip'], 
            'little': ['leftLittleFingerTip']
        }

        # 第四步：绘制手部 (严格按照visualize_2d.py)
        self.draw_hand(right_dict, transforms_in_cam, cam_img, camera_intrinsics, right=True)
        self.draw_hand(left_dict, transforms_in_cam, cam_img, camera_intrinsics, right=False)

        return cam_img

    def imgs_to_mp4(self, img_list: List[np.ndarray], mp4_path: str, 
                   fps: int = 30, fourcc=None):
        """
        使用 PyAV 来写入视频，避免 OpenCV VideoWriter 的编解码器问题
        """
        if not img_list:
            return
            
        H, W, _ = img_list[0].shape
        
        try:
            import av
            
            # 创建输出容器
            container = av.open(mp4_path, mode='w')
            
            # 添加视频流
            stream = container.add_stream('libx264', rate=fps)
            stream.width = W
            stream.height = H
            stream.pix_fmt = 'yuv420p'
            
            for i, img in enumerate(img_list):
                # 确保图像是 RGB 格式且为 uint8
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                
                # 创建 PyAV 帧
                frame = av.VideoFrame.from_ndarray(img, format='rgb24')
                frame.pts = i
                
                # 编码帧
                for packet in stream.encode(frame):
                    container.mux(packet)
            
            # 刷新编码器
            for packet in stream.encode():
                container.mux(packet)
                
            # 关闭容器
            container.close()
                
        except ImportError:
            # 回退到原来的 OpenCV 方法
            self._imgs_to_mp4_opencv(img_list, mp4_path, fps)
        except Exception as e:
            # 回退到 OpenCV
            self._imgs_to_mp4_opencv(img_list, mp4_path, fps)
    
    def _imgs_to_mp4_opencv(self, img_list: List[np.ndarray], mp4_path: str, fps: int = 30):
        """OpenCV 的后备方法，尝试多种编解码器"""
        H, W, _ = img_list[0].shape
        
        # 尝试不同的编解码器
        codecs = [
            cv2.VideoWriter_fourcc(*'mp4v'),
            cv2.VideoWriter_fourcc(*'XVID'),
            cv2.VideoWriter_fourcc(*'MJPG'),
            -1
        ]
        
        for fourcc in codecs:
            video_out = cv2.VideoWriter(mp4_path, fourcc, fps, (W, H))
            
            if video_out.isOpened():
                for img in img_list:
                    BGR_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    video_out.write(BGR_img)
                video_out.release()
                
                # 检查文件是否成功创建
                import os
                if os.path.exists(mp4_path) and os.path.getsize(mp4_path) > 0:
                    return

    def visualize_action_sequence(self, 
                                actions_48d: np.ndarray,
                                camera_extrinsics: np.ndarray,
                                camera_intrinsics: np.ndarray,
                                output_path: str,
                                image_height: int = 1080,
                                image_width: int = 1920,
                                is_normalized: bool = True,
                                stat_path: Optional[str] = None) -> None:
        """
        可视化48D动作序列并生成视频 (按照visualize_2d.py的主流程)
        """
        # 反归一化 (如果需要)
        if is_normalized:
            if stat_path is None:
                raise ValueError("Normalized data requires stat_path for denormalization")
            actions_48d = self.denormalize_actions(actions_48d, stat_path)
        
        print(f"🎬 开始生成视频: {output_path}")
        print(f"   - 帧数: {len(actions_48d)}")
        print(f"   - 分辨率: {image_width}x{image_height}")
        print(f"   - 帧率: {self.fps}fps")
        
        # 严格按照visualize_2d.py的主循环
        out_imgs = []
        for i, (action_48d, cam_ext) in enumerate(zip(actions_48d, camera_extrinsics)):
            # 创建包含关键点的帧
            cam_img = self.create_frame_with_keypoints(
                action_48d, cam_ext, camera_intrinsics,
                image_height=image_height, image_width=image_width
            )
            
            out_imgs.append(cam_img)
            
            if (i + 1) % 10 == 0:
                print(f"   - 已处理: {i+1}/{len(actions_48d)} 帧")

        # 严格按照visualize_2d.py的视频写入
        self.imgs_to_mp4(out_imgs, output_path, fps=self.fps)
        print(f"✅ 视频生成完成: {output_path}")


def visualize_egodx_sample(sample_data: Dict[str, Any], 
                          output_path: str,
                          stat_path: str,
                          image_size: Tuple[int, int] = (1920, 1080),
                          fps: int = 30) -> None:
    """
    可视化EgoDx数据集样本的接口函数
    """
    # 创建可视化器
    visualizer = HandKeypointVisualizer(fps=fps)
    
    # 提取数据
    actions = sample_data['actions']  # (chunk_size, 48)
    camera_extrinsics = sample_data['action_camera_extrinsics']  # (chunk_size, 4, 4)
    camera_intrinsics = sample_data['camera_intrinsics']  # (3, 3)
    
    # 转换为numpy (如果是tensor)
    if torch.is_tensor(actions):
        actions = actions.cpu().numpy()
    if torch.is_tensor(camera_extrinsics):
        camera_extrinsics = camera_extrinsics.cpu().numpy()
    if torch.is_tensor(camera_intrinsics):
        camera_intrinsics = camera_intrinsics.cpu().numpy()
    
    # 生成可视化视频
    visualizer.visualize_action_sequence(
        actions_48d=actions,
        camera_extrinsics=camera_extrinsics,
        camera_intrinsics=camera_intrinsics,
        output_path=output_path,
        image_height=image_size[1],
        image_width=image_size[0],
        is_normalized=True,  # EgoDx数据集使用归一化
        stat_path=stat_path
    )


if __name__ == "__main__":
    print("🧪 Hand Keypoint Visualizer - 基于 visualize_2d.py 实现")
