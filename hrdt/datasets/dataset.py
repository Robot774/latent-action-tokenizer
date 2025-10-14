import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

import random
from typing import Dict, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from hrdt.utils.image_corrupt import image_corrupt
from hrdt.datasets.robotwin2.robotwin_agilex_dataset import RobotwinAgilexDataset
from hrdt.datasets.pretrain.egodex_dataset import EgoDexDataset
from PIL import Image
from torchvision import transforms


class VLAConsumerDataset(Dataset):
    """A vision-language-action Dataset for supervised training.
    This dataset will load data from the buffer directory.
    """

    def __init__(
        self,
        config,
        image_transform,
        num_cameras,
        image_size=None,
        auto_adjust_image_brightness=False,
        image_aug=False,
        image_corrupt_severity=None,
        dataset_type=None,
        state_noise_snr=None,
        use_precomp_lang_embed=True,
        upsample_rate=None,
        val=False,
        task_name="open_laptop",
        dataset_name="test_robotwin",  # Add dataset_name parameter
    ):
        super(VLAConsumerDataset, self).__init__()
        self.dataset_name = dataset_name
        DATASET_NAMES = {self.dataset_name}
        
        # Create the mapping between dataset name and id
        self.dataset_name2id = {name: i for i, name in enumerate(DATASET_NAMES)}
        self.dataset_id2name = {i: name for i, name in enumerate(DATASET_NAMES)}

        self.state_noise_snr = state_noise_snr
        self.num_cameras = num_cameras
        self.img_history_size = config["common"]["img_history_size"]
        self.image_transform = image_transform   

        # Initialize dataset based on dataset_name
        if self.dataset_name == "egodex":
            self.hdf5_dataset = EgoDexDataset(
                config=config,
                upsample_rate=upsample_rate,
                val=val,
                use_precomp_lang_embed=use_precomp_lang_embed,
                # Note: override default paths if needed
                data_root="/dataset_rc_mm/share/datasets/ml-site.cdn-apple.com/egodex",
                stat_path="/workspace/chenby10@xiaopeng.com/H_RDT/datasets/pretrain/egodex_stat.json",
            )
        elif self.dataset_name == "robotwin_agilex":
            self.hdf5_dataset = RobotwinAgilexDataset(
                mode="multi_task",
                config=config,
                # Note: override default paths
                multi_task_root_dir="/dataset_rc_mm/share/datasets/huggingface.co/TianxingChen/RoboTwin2.0/dataset",
            )
            '''
            self.hdf5_dataset = RobotwinAgilexDataset(
                mode="single_task",
                task_name=task_name,
                hdf5_folder="Aloha-AgileX/data",
                max_episodes=50,
                config=config
                # Note: override default paths
                # single_task_root_dir="/path/to/your/robotwin2/single",
            )
            '''
        else:
            raise ValueError(f"Unknown dataset_name: {self.dataset_name}")
            
        print(f"Initialized dataset: {self.dataset_name}")

        self.use_precomp_lang_embed = use_precomp_lang_embed
        self.dataset_type = dataset_type

        self.image_size = image_size
        self.auto_adjust_image_brightness = auto_adjust_image_brightness
        # self.image_aug_transform = get_image_augmentation()
        self.image_aug = image_aug

    def get_dataset_name2id(self):
        return self.dataset_name2id

    def get_dataset_id2name(self):
        return self.dataset_id2name

    @staticmethod
    def pairwise(iterable):
        a = iter(iterable)
        return zip(a, a)

    def __len__(self) -> int:
        return len(self.hdf5_dataset)
    
    def get_item(self, index=None):
        """为MultiHDF5VLADataset提供兼容接口"""
        if index is None:
            index = random.randint(0, len(self) - 1)
        return self.__getitem__(index)

    def __getitem__(self, index):
        # Get data from backend dataset
        try:
            res = self.hdf5_dataset.get_item(index)
        except Exception as e:
            print(f"Error loading episode {index}: {e}")
            return None
            
        # Add check for res being None, retry a few times if it's None
        retry_count = 0
        max_retries = 5
        while res is None and retry_count < max_retries:
            retry_count += 1
            print(f"Got None data item, retrying {retry_count} time...")
            try:
                res = self.hdf5_dataset.get_item(index)
            except Exception as e:
                print(f"Error during retry data loading: {e}")
                
        # If still None after multiple retries, return a default value to prevent training interruption
        if res is None:
            print("Warning: Still unable to get valid data after multiple retries, returning default value")

        data_dict = {}
        data_dict['dataset_name'] = res['dataset_name']
        data_dict['data_idx'] = self.dataset_name2id[data_dict['dataset_name']]

        # Process state and action data
        data_dict["states"] = res['states']
        data_dict["actions"] = res['actions']
        data_dict["action_norm"] = res['action_norm']

        # Helper function to process images (current or future)
        def process_images(raw_images, mask_key, field_name):
            if self.dataset_name in ['egodx']:
                # Single camera / stitched image processing
                image_metas = []
                images = raw_images[0]
                valid_mask = res.get(mask_key, [np.ones(self.img_history_size, dtype=bool)])[0]
                image_metas.append((images, valid_mask))
                
                rearranged_images = []
                for hist_idx in range(self.img_history_size):
                    images, valid_mask = image_metas[0]
                    if valid_mask[hist_idx]:
                        rearranged_images.append((images[hist_idx], True))
                    else:
                        rearranged_images.append((None, False))
            else:
                # Multi-view processing (original logic)
                image_metas = []
                for cam_idx in range(self.num_cameras):
                    images = raw_images[cam_idx]
                    valid_mask = res.get(mask_key, np.ones((self.num_cameras, self.img_history_size), dtype=bool))[cam_idx]
                    image_metas.append((images, valid_mask))

                rearranged_images = []
                for hist_idx in range(self.img_history_size):
                    for cam_idx in range(self.num_cameras):
                        images, valid_mask = image_metas[cam_idx]
                        if valid_mask[hist_idx]:
                            rearranged_images.append((images[hist_idx], True))
                        else:
                            rearranged_images.append((None, False))

            all_pixel_values = []
            for image, valid in rearranged_images:
                image = Image.fromarray(image) if image is not None else None

                if valid and self.auto_adjust_image_brightness:
                    pixel_values = list(image.getdata())
                    average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                    if average_brightness <= 0.15:
                        image = transforms.ColorJitter(brightness=(1.75,1.75))(image)

                # Only apply image augmentation to 50% of the images
                if valid and self.image_aug and (random.random() > 0.5):
                    aug_type = random.choice([
                        "corrput_only", "color_only", "both"])
                    if aug_type != "corrput_only":
                        image = transforms.ColorJitter(
                            brightness=0.3, contrast=0.4, saturation=0.5, hue=0.03)(image)
                    if aug_type != "color_only":
                        image = image_corrupt(image)

                pixel_values = self.image_transform(image)
                all_pixel_values.append(pixel_values)

            # Process dino-siglip format images
            pv_example = all_pixel_values[0]
            merged_pixel_values = {
                k: torch.stack(
                    [pv[k] for pv in all_pixel_values]
                )
                for k in pv_example
            }
            data_dict[field_name] = merged_pixel_values

        # Process current images
        process_images(res['current_images'], 'current_images_mask', 'images')
        
        # Process future images if available
        if 'future_images' in res:
            process_images(res['future_images'], 'future_images_mask', 'future_images')

        if self.use_precomp_lang_embed:
            # All datasets should provide lang_embeds as tensor
            if "lang_embeds" in res:
                data_dict["lang_embeds"] = res["lang_embeds"]
            elif torch.is_tensor(res["instruction"]):
                data_dict["lang_embeds"] = res["instruction"]
            else:
                # Legacy: load from file path
                data_dict["lang_embeds"] = torch.load(res["instruction"])["embeddings"].squeeze(0)

        # Add camera parameters if available in res
        camera_fields = ['current_camera_extrinsics', 'action_camera_extrinsics', 'camera_intrinsics']
        for field in camera_fields:
            if field in res:
                data_dict[field] = res[field]

        # Convert all numpy arrays to torch tensors
        for k, v in data_dict.items():
            if isinstance(v, np.ndarray):
                data_dict[k] = torch.from_numpy(v)

        # Verify all data is tensors
        for k, v in data_dict.items():
            assert not isinstance(v, np.ndarray), f"key: {k}, value: {v}"

        return data_dict

class DataCollatorForVLAConsumerDataset(object):
    """Collate examples for supervised training."""

    def __init__(self, use_precomp_lang_embed=True) -> None:
        self.use_precomp_lang_embed = use_precomp_lang_embed
        
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Initialize batch with common fields
        batch = {
            "states": [],
            "actions": [],
            "action_norm": [],
            "images": [],
            "data_indices": [],
        }
        
        if self.use_precomp_lang_embed:
            lang_embeds = []
            lang_embed_lens = []

        # Process each instance in the batch
        for instance in instances:
            # Process numeric data
            keys_to_check = [
                'states', 'actions',
                'action_norm',
            ]
            for key in keys_to_check:
                if isinstance(instance[key], torch.Tensor):
                    item = instance[key]
                else:
                    item = torch.from_numpy(instance[key])
                batch[key].append(item)

            # Process images
            batch["images"].append(instance["images"])
            batch["data_indices"].append(instance["data_idx"])

            if self.use_precomp_lang_embed and "lang_embeds" in instance:
                lang_embeds.append(instance["lang_embeds"])
                lang_embed_lens.append(instance["lang_embeds"].shape[0])

        # Stack tensors for numeric data
        keys_to_stack = [
            'states', 'actions',
            'action_norm',
        ]
        for key in keys_to_stack:
            batch[key] = torch.stack(batch[key], dim=0)

        # Process dino-siglip format images
        pv_example = batch["images"][0]
        merged_pixel_values = {
            k: torch.stack(
                [pv[k] for pv in batch["images"]]
            )
            for k in pv_example
        }
        batch["images"] = merged_pixel_values

        if self.use_precomp_lang_embed:
            lang_embeds = torch.nn.utils.rnn.pad_sequence(
                lang_embeds,
                batch_first=True,
                padding_value=0)
            input_lang_attn_mask = torch.zeros(
                lang_embeds.shape[0], lang_embeds.shape[1], dtype=torch.bool)
            for i, length in enumerate(lang_embed_lens):
                input_lang_attn_mask[i, :length] = True
            batch["lang_embeds"] = lang_embeds
            batch["lang_attn_mask"] = input_lang_attn_mask

        return batch



class MultiDataCollatorForVLAConsumerDataset(object):
    """Collate examples for supervised training."""

    def __init__(self, unified_action_dim=48, use_precomp_lang_embed=True) -> None:
        self.unified_action_dim = unified_action_dim  # 🆕 3. 统一pad维度
        self.use_precomp_lang_embed = use_precomp_lang_embed
        
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # 🆕 4. 构建数据集索引映射
        dataset_indices_map = {}
        for i, instance in enumerate(instances):
            dataset_name = instance['dataset_name']
            if dataset_name not in dataset_indices_map:
                dataset_indices_map[dataset_name] = []
            dataset_indices_map[dataset_name].append(i)
        
        # Initialize batch with common fields
        batch = {
            "states": [],
            "actions": [],
            "action_norm": [],
            "images": [],
            "data_indices": [],
            "future_images": [],  # 🆕 1. 添加future_images
            "current_camera_extrinsics": [],  # 相机外参 - 当前帧
            "action_camera_extrinsics": [],   # 相机外参 - 动作序列
            "camera_intrinsics": [],          # 相机内参
        }
        
        if self.use_precomp_lang_embed:
            lang_embeds = []
            lang_embed_lens = []

        # Process each instance in the batch
        for instance in instances:
            # Process numeric data with padding
            keys_to_check = [
                'states', 'actions',
                'action_norm',
            ]
            for key in keys_to_check:
                if isinstance(instance[key], torch.Tensor):
                    item = instance[key]
                else:
                    item = torch.from_numpy(instance[key])
                
                # 🆕 3. 统一pad处理
                original_dim = item.shape[-1]
                if original_dim < self.unified_action_dim:
                    pad_size = self.unified_action_dim - original_dim
                    padded_item = torch.cat([item, torch.zeros(*item.shape[:-1], pad_size)], dim=-1)
                else:
                    padded_item = item
                
                batch[key].append(padded_item)
            

            # 🆕 2. 只使用主摄像头
            images = instance["images"]
            if isinstance(images, dict):
                # 已经处理过的图像格式 - 检查是否有多相机维度
                processed_images = {}
                for k, v in images.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 3:
                        # 如果第一个维度>1，说明是多相机，只取第一个
                        if v.shape[0] > 1:
                            processed_images[k] = v[0:1]  # 只取第一个相机
                        else:
                            processed_images[k] = v
                    else:
                        processed_images[k] = v
                batch["images"].append(processed_images)
            else:
                # 如果是多摄像头列表，取第一个
                main_camera_images = images[0] if isinstance(images, list) else images
                batch["images"].append(main_camera_images)
            
            # 🆕 1. future_images兼容
            if "future_images" in instance:
                future_images = instance["future_images"]
                if isinstance(future_images, dict):
                    # 已经处理过的图像格式 - 检查是否有多相机维度
                    processed_future_images = {}
                    for k, v in future_images.items():
                        if isinstance(v, torch.Tensor) and v.dim() >= 3:
                            # 如果第一个维度>1，说明是多相机，只取第一个
                            if v.shape[0] > 1:
                                processed_future_images[k] = v[0:1]  # 只取第一个相机
                            else:
                                processed_future_images[k] = v
                        else:
                            processed_future_images[k] = v
                    batch["future_images"].append(processed_future_images)
                else:
                    # 如果是多摄像头，取主摄像头
                    main_future = future_images[0] if isinstance(future_images, list) else future_images
                    batch["future_images"].append(main_future)
            else:
                # 如果没有future_images，复用current images
                batch["future_images"].append(batch["images"][-1])
            
            # 处理相机参数
            camera_keys = ['current_camera_extrinsics', 'action_camera_extrinsics', 'camera_intrinsics']
            for key in camera_keys:
                if key in instance:
                    if isinstance(instance[key], torch.Tensor):
                        item = instance[key]
                    else:
                        item = torch.from_numpy(instance[key])
                    batch[key].append(item)
            
            batch["data_indices"].append(instance["data_idx"])

            if self.use_precomp_lang_embed and "lang_embeds" in instance:
                lang_embeds.append(instance["lang_embeds"])
                lang_embed_lens.append(instance["lang_embeds"].shape[0])

        # Stack tensors for numeric data
        keys_to_stack = [
            'states', 'actions',
            'action_norm',
        ]
        for key in keys_to_stack:
            batch[key] = torch.stack(batch[key], dim=0)
        
        # Stack camera parameters
        camera_keys_to_stack = [
            'current_camera_extrinsics', 
            'action_camera_extrinsics', 
            'camera_intrinsics'
        ]
        for key in camera_keys_to_stack:
            if len(batch[key]) == 0:
                continue
            batch[key] = torch.stack(batch[key], dim=0)

        # Process dino-siglip format images - flatten structure for DataPrefetcher compatibility
        pv_example = batch["images"][0]
        merged_pixel_values = {
            k: torch.stack(
                [pv[k] for pv in batch["images"]]
            )
            for k in pv_example
        }
        # Extract pixel_values to top level for DataPrefetcher compatibility
        batch["rgb_initial"] = merged_pixel_values["pixel_values"]
        
        # 🆕 1. Process future_images - flatten structure for DataPrefetcher compatibility
        future_pv_example = batch["future_images"][0]
        merged_future_pixel_values = {
            k: torch.stack(
                [pv[k] for pv in batch["future_images"]]
            )
            for k in future_pv_example
        }
        # Extract pixel_values to top level for DataPrefetcher compatibility
        batch["rgb_future"] = merged_future_pixel_values["pixel_values"]

        # if self.use_precomp_lang_embed:
        #     lang_embeds = torch.nn.utils.rnn.pad_sequence(
        #         lang_embeds,
        #         batch_first=True,
        #         padding_value=0)
        #     input_lang_attn_mask = torch.zeros(
        #         lang_embeds.shape[0], lang_embeds.shape[1], dtype=torch.bool)
        #     for i, length in enumerate(lang_embed_lens):
        #         input_lang_attn_mask[i, :length] = True
        #     batch["lang_embeds"] = lang_embeds
        #     batch["lang_attn_mask"] = input_lang_attn_mask

        # 🆕 4. 添加数据集映射
        batch["dataset_indices_map"] = dataset_indices_map

        return batch



