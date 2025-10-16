import omegaconf
import hydra
import pyrootutils
import os
import sys
import torch
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
from transformers import AutoTokenizer
from transformers.utils import FEATURE_EXTRACTOR_NAME, get_file_from_repo
import json
from common.data.datasets import LMDBDataset_for_MotoGPT_RT1, LMDBDataset_for_MotoGPT_OXE, LMDBDataset_for_MotoGPT_Video, LMDBDataset_Mix, JsonDataset_for_MotoGPT_Video, NpzDataset_for_MotoGPT_Video, LMDBDataset_for_MotoGPT_CALVIN
from common.data.mix_utils import BASE_STEPSIZE, DISPLAY_KEY
from torchvision.transforms.v2 import Resize, InterpolationMode
from torch.utils.data import ConcatDataset, WeightedRandomSampler

# Import HRDT datasets
try:
    from hrdt.datasets.dataset import VLAConsumerDataset, MultiDataCollatorForVLAConsumerDataset
    HRDT_AVAILABLE = True
    print("✅ HRDT datasets imported successfully")
except ImportError as e:
    print(f"❌ Warning: HRDT datasets not available. Import error: {e}")
    print("Make sure hrdt module is in PYTHONPATH and all dependencies are installed.")
    import traceback
    print("Full traceback:")
    traceback.print_exc()
    HRDT_AVAILABLE = False

data_type2dataset_cls = {
    'rt1': LMDBDataset_for_MotoGPT_RT1,
    'video': LMDBDataset_for_MotoGPT_Video,
    'oxe': LMDBDataset_for_MotoGPT_OXE,
    'video_json': JsonDataset_for_MotoGPT_Video,
    'video_npz': NpzDataset_for_MotoGPT_Video,
    'calvin': LMDBDataset_for_MotoGPT_CALVIN,
}

# Add HRDT dataset types if available
if HRDT_AVAILABLE:
    data_type2dataset_cls.update({
        'hrdt_egodx': VLAConsumerDataset,
        'hrdt_robotwin': VLAConsumerDataset,
        'hrdt_mix': VLAConsumerDataset,
    })

def create_hrdt_image_transform(rgb_shape):
    """Create image transform for HRDT datasets compatible with Motion Tokenizer"""
    from torchvision import transforms
    from PIL import Image
    
    def transform_fn(image):
        if image is None:
            image = Image.new('RGB', tuple(rgb_shape), (0, 0, 0))
        
        transform = transforms.Compose([
            transforms.Resize(tuple(rgb_shape)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        pixel_values = transform(image)
        return {"pixel_values": pixel_values}
    
    return transform_fn

def load_dataset(data_config, extra_data_config):
    if type(data_config) is str:
        data_config = omegaconf.OmegaConf.load(data_config)
        data_config = dict(data_config)

    data_type = data_config.pop('data_type')

    key_map = {
        'latent_motion_pred': 'do_extract_future_frames',
        'act_pred': 'do_extract_action'
    }
    for k, v in extra_data_config.items():
        mapped_k = key_map.get(k, k)
        data_config[mapped_k] = v

    if data_type == 'mix':
        sub_data_configs = data_config.pop('sub_data_configs')
        rgb_preprocessor = Resize(data_config['rgb_shape'], interpolation=InterpolationMode.BICUBIC, antialias=True)
        train_datasets = []
        eval_datasets = []
        train_sample_weights = []
        eval_sample_weights = []

        for sub_data_config in sub_data_configs:
            sub_data_config = dict(sub_data_config)
            data_name = sub_data_config.pop('data_name')
            weight = sub_data_config.pop('weight')
            sub_data_type = sub_data_config.get('data_type', '')
            
            # Handle HRDT datasets differently
            if sub_data_type.startswith('hrdt_'):
                # For HRDT datasets, preserve all necessary parameters
                sub_data_config['rgb_shape'] = data_config['rgb_shape']
                # Don't need rgb_preprocessor for HRDT as it handles its own transforms
            else:
                # Standard dataset handling
                if ('lmdb_dir' not in sub_data_config) and ('lmdb_dir' in data_config):
                    sub_data_config['lmdb_dir'] = os.path.join(data_config['lmdb_dir'], data_name)
                if ('video_dir' not in sub_data_config) and ('video_dir' in data_config):
                    sub_data_config['video_dir'] = os.path.join(data_config['video_dir'], data_name, DISPLAY_KEY.get(data_name, 'image'))
                step_size = max(round(BASE_STEPSIZE.get(data_name, 1) / BASE_STEPSIZE['fractal20220817_data']), 1)
                sub_data_config['skip_frame'] = data_config['skip_frame'] * step_size
                
                if 'max_skip_frame' in data_config:
                    sub_data_config['max_skip_frame'] = data_config['max_skip_frame'] * step_size
                    
                sub_data_config['rgb_shape'] = data_config['rgb_shape']
                sub_data_config['rgb_preprocessor'] = rgb_preprocessor

            train_dataset, eval_dataset =  load_dataset(sub_data_config, extra_data_config)
            train_datasets.append(train_dataset)
            eval_datasets.append(eval_dataset)
            train_sample_weights.append(weight)
            eval_sample_weights.append(weight)

        
        if data_config['weighted']:
            train_dataset = LMDBDataset_Mix(datasets=train_datasets, sample_weights=train_sample_weights)
            eval_dataset = LMDBDataset_Mix(datasets=eval_datasets, sample_weights=eval_sample_weights)
        else:
            train_dataset = ConcatDataset(train_datasets)
            eval_dataset = ConcatDataset(eval_datasets)
            
    else:
        dataset_cls = data_type2dataset_cls[data_type]
        
        # Special handling for HRDT datasets
        if data_type.startswith('hrdt_') and HRDT_AVAILABLE:
            # Create image transform for HRDT
            image_transform = create_hrdt_image_transform(data_config['rgb_shape'])
            
            # Extract HRDT-specific configuration
            hrdt_config = {
                "common": {
                    "img_history_size": data_config.get('img_history_size', 1),
                    "action_chunk_size": data_config.get('action_chunk_size', 16),
                    "chunk_size": data_config.get('chunk_size', 16),
                    "num_cameras": data_config.get('num_cameras', 1),
                    "state_dim": data_config.get('state_dim', 48),
                    "action_dim": data_config.get('action_dim', 48)
                }
            }
            
            # Dataset-specific parameters
            dataset_params = {
                'config': hrdt_config,
                'image_transform': image_transform,
                'num_cameras': data_config.get('num_cameras', 1),
                'image_aug': data_config.get('image_aug', False),
                'upsample_rate': data_config.get('upsample_rate', 3),
                'use_precomp_lang_embed': data_config.get('use_precomp_lang_embed', True),
            }
            
            # Set dataset name based on type
            if data_type == 'hrdt_egodx':
                dataset_params.update({
                    'dataset_name': 'egodex',
                    'dataset_type': 'pretrain',
                })
            elif data_type == 'hrdt_robotwin':
                dataset_params.update({
                    'dataset_name': 'robotwin_agilex',
                    'dataset_type': 'finetune',
                    'task_name': data_config.get('task_name', 'open_laptop'),
                })
                # Adjust config for RobotWin
                hrdt_config["common"]["action_dim"] = data_config.get('action_dim', 14)
                hrdt_config["common"]["state_dim"] = data_config.get('state_dim', 14)
                hrdt_config["common"]["num_cameras"] = data_config.get('num_cameras', 3)
            elif data_type == 'hrdt_mix':
                # For mixed datasets, use the dataset_name from config
                dataset_params.update({
                    'dataset_name': data_config.get('dataset_name', 'egodex'),
                    'dataset_type': data_config.get('dataset_type', 'pretrain'),
                })
                if 'task_name' in data_config:
                    dataset_params['task_name'] = data_config['task_name']
            
            # Create train and eval datasets
            train_dataset = dataset_cls(val=False, **dataset_params)  # Use training data
            eval_dataset = dataset_cls(val=True, **dataset_params)    # Use test data
        else:
            # Standard dataset creation
            train_dataset = dataset_cls(split='train', **data_config)
            eval_dataset = dataset_cls(split='val', **data_config)
    
    return train_dataset, eval_dataset
