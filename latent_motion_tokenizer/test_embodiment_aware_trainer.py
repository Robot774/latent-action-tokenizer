"""
Test EmbodimentAware Trainer
Tests the new trainer with action conditioning capabilities
"""
import torch
import sys
import os
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer')

from src.trainers.embodiment_aware_trainer import EmbodimentAwareLatentMotionTokenizer_Trainer


def create_mock_batch():
    """Create a mock batch for testing"""
    batch_size = 2
    
    batch = {
        # Visual data (HRDT format)
        'rgb_initial': torch.randn(batch_size, 1, 3, 224, 224),  # (B, 1, C, H, W)
        'rgb_future': torch.randn(batch_size, 1, 3, 224, 224),   # (B, 1, C, H, W)
        
        # Action data
        'actions': torch.randn(batch_size, 4, 48),               # (B, chunk_size, action_dim)
        'states': torch.randn(batch_size, 1, 48),                # (B, 1, state_dim)
        'embodiment_ids': torch.tensor([0, 0], dtype=torch.long) # RobotWin embodiment
    }
    
    return batch


def create_mock_model():
    """Create a mock EmbodimentAware model for testing"""
    from src.models.embodiment_aware_action_encoder import EmbodimentAwareActionEncoder, EmbodimentAwareActionDecoder
    from src.models.visual_action_fusion import VisualActionFusion
    from src.models.a_former import AFormer
    from transformers import ViTConfig
    
    # Mock model that mimics EmbodimentAwareLatentMotionTokenizer interface
    class MockEmbodimentAwareModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enable_action_conditioning = True
            
            # Mock components
            self.mock_layer = torch.nn.Linear(100, 100)
            
        def forward(self, cond_pixel_values, target_pixel_values, 
                   actions=None, states=None, embodiment_ids=None,
                   return_recons_only=False, **kwargs):
            
            batch_size = cond_pixel_values.shape[0]
            
            # Mock outputs
            outputs = {
                'recons_pixel_values': torch.randn_like(target_pixel_values),
                'indices': torch.randint(0, 100, (batch_size, 8)),
                'loss': torch.tensor(0.5),
                'commit_loss': torch.tensor(0.1),
                'recons_loss': torch.tensor(0.3),
                'perceptual_loss': torch.tensor(0.1),
            }
            
            # Add action-related outputs if actions provided
            if actions is not None:
                outputs.update({
                    'action_recons_loss': torch.tensor(0.2),
                    'action_features': torch.randn(batch_size, 4, 768),
                    'state_features': torch.randn(batch_size, 1, 768),
                    'action_tokens': torch.randn(batch_size, 8, 768),
                    'decoded_actions': torch.randn_like(actions)
                })
            
            if return_recons_only:
                return {k: v for k, v in outputs.items() if k in ['recons_pixel_values', 'indices']}
            
            return outputs
        
        def get_state_dict_to_save(self):
            return self.state_dict()
        
        @property 
        def config(self):
            return {'model_type': 'mock_embodiment_aware'}
    
    return MockEmbodimentAwareModel()


def create_mock_rgb_preprocessor():
    """Create a mock RGB preprocessor"""
    class MockRGBPreprocessor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            
        def post_process(self, x):
            # Mock post-processing: just clamp to [0, 1]
            return torch.clamp(x, 0, 1)
    
    return MockRGBPreprocessor()


def create_mock_dataloaders():
    """Create mock dataloaders"""
    class MockDataLoader:
        def __init__(self, batch):
            self.batch = batch
            self.dataset = [batch] * 10  # Mock dataset with 10 samples
            
        def __len__(self):
            return len(self.dataset)
        
        def __iter__(self):
            return iter([self.batch])
    
    batch = create_mock_batch()
    train_loader = MockDataLoader(batch)
    eval_loader = MockDataLoader(batch)
    
    return train_loader, eval_loader


def create_mock_prefetcher():
    """Create a mock data prefetcher"""
    class MockPrefetcher:
        def __init__(self, batch):
            self.batch = batch
            self.called_count = 0
            
        def next(self):
            if self.called_count < 5:  # Return batch 5 times, then None
                self.called_count += 1
                return self.batch, 0.01  # batch, load_time
            return None, 0
        
        def next_without_none(self):
            return self.batch, 0.01
        
        def __len__(self):
            return 10
    
    batch = create_mock_batch()
    return MockPrefetcher(batch)


def test_trainer_initialization():
    """Test trainer initialization"""
    print("🧪 Testing EmbodimentAware Trainer Initialization")
    print("=" * 50)
    
    # Create mock components
    model = create_mock_model()
    rgb_preprocessor = create_mock_rgb_preprocessor()
    train_loader, eval_loader = create_mock_dataloaders()
    
    try:
        # Initialize trainer (simplified for testing)
        trainer = EmbodimentAwareLatentMotionTokenizer_Trainer(
            latent_motion_tokenizer=model,
            rgb_preprocessor=rgb_preprocessor,
            train_dataloader=train_loader,
            eval_dataloader=eval_loader,
            save_path='tmp/test_trainer',
            num_epochs=1,
            action_eval_probability=0.7
        )
        
        print("✅ Trainer initialization successful")
        print(f"   - Action conditioning support: {trainer.supports_action_conditioning}")
        print(f"   - Action eval probability: {trainer.action_eval_probability}")
        
        return trainer
        
    except Exception as e:
        print(f"❌ Trainer initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_batch_adaptation():
    """Test batch adaptation for action conditioning"""
    print("\n🧪 Testing Batch Adaptation")
    print("=" * 30)
    
    trainer = test_trainer_initialization()
    if trainer is None:
        return
    
    # Create test batch
    batch = create_mock_batch()
    
    try:
        # Test batch adaptation
        adapted_batch = trainer._adapt_batch_for_action_conditioning(batch)
        
        print("✅ Batch adaptation successful")
        print(f"   - Original keys: {list(batch.keys())}")
        print(f"   - Adapted keys: {list(adapted_batch.keys())}")
        
        # Check action data
        if 'actions' in adapted_batch and adapted_batch['actions'] is not None:
            print(f"   - Actions shape: {adapted_batch['actions'].shape}")
            print(f"   - States shape: {adapted_batch['states'].shape}")
            print(f"   - Embodiment IDs: {adapted_batch['embodiment_ids']}")
        
    except Exception as e:
        print(f"❌ Batch adaptation failed: {e}")
        import traceback
        traceback.print_exc()


def test_evaluation_modes():
    """Test evaluation mode sampling"""
    print("\n🧪 Testing Evaluation Mode Sampling")
    print("=" * 35)
    
    trainer = test_trainer_initialization()
    if trainer is None:
        return
    
    # Sample multiple modes to check distribution
    modes = []
    for _ in range(100):
        mode = trainer._sample_evaluation_mode()
        modes.append(mode)
    
    # Count occurrences
    from collections import Counter
    mode_counts = Counter(modes)
    
    print("✅ Evaluation mode sampling successful")
    print(f"   - Mode distribution (100 samples):")
    for mode, count in mode_counts.items():
        print(f"     • {mode}: {count}% ({count}/100)")


def test_loss_calculation():
    """Test loss calculation with action conditioning"""
    print("\n🧪 Testing Loss Calculation")
    print("=" * 28)
    
    trainer = test_trainer_initialization()
    if trainer is None:
        return
    
    # Mock the prefetcher
    trainer.train_prefetcher = create_mock_prefetcher()
    trainer.eval_prefetcher = create_mock_prefetcher()
    
    batch = create_mock_batch()
    
    try:
        # Test loss calculation
        loss_dict = trainer.calculate_loss(batch, train=True)
        
        print("✅ Loss calculation successful")
        print(f"   - Loss keys: {list(loss_dict.keys())}")
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:  # Only print scalar tensors
                    print(f"   - {key}: {value.item():.4f}")
                else:
                    print(f"   - {key}: Tensor{list(value.shape)}")
            else:
                print(f"   - {key}: {value}")
        
    except Exception as e:
        print(f"❌ Loss calculation failed: {e}")
        import traceback
        traceback.print_exc()


def test_action_visualization():
    """Test action reconstruction visualization"""
    print("\n🧪 Testing Action Visualization")
    print("=" * 32)
    
    trainer = test_trainer_initialization()
    if trainer is None:
        return
    
    # Mock the prefetcher
    trainer.train_prefetcher = create_mock_prefetcher()
    trainer.eval_prefetcher = create_mock_prefetcher()
    
    try:
        # Create test directory
        test_dir = 'tmp/test_action_viz'
        os.makedirs(test_dir, exist_ok=True)
        
        # Test action visualization
        gt_actions = torch.randn(4, 14)  # (chunk_size, 14_dims)
        decoded_actions = gt_actions + 0.1 * torch.randn(4, 14)  # Add some noise
        
        trainer._visualize_action_reconstruction(
            gt_actions=gt_actions,
            decoded_actions=decoded_actions,
            embodiment_name="RobotWin",
            effective_dims=14,
            save_path=os.path.join(test_dir, 'test_action_plot.png')
        )
        
        print("✅ Action visualization successful")
        print(f"   - Plot saved to: {test_dir}/test_action_plot.png")
        
    except Exception as e:
        print(f"❌ Action visualization failed: {e}")
        import traceback
        traceback.print_exc()


def test_full_evaluation():
    """Test full evaluation pipeline"""
    print("\n🧪 Testing Full Evaluation Pipeline")
    print("=" * 36)
    
    trainer = test_trainer_initialization()
    if trainer is None:
        return
    
    # Mock the prefetcher
    trainer.train_prefetcher = create_mock_prefetcher()
    trainer.eval_prefetcher = create_mock_prefetcher()
    
    try:
        # Create test directory
        test_dir = 'tmp/test_full_eval'
        os.makedirs(test_dir, exist_ok=True)
        
        # Test full evaluation
        trainer.eval_latent_motion_reconstruction(test_dir)
        
        print("✅ Full evaluation pipeline successful")
        print(f"   - Results saved to: {test_dir}")
        
        # List generated files
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), test_dir)
                print(f"   - Generated: {rel_path}")
        
    except Exception as e:
        print(f"❌ Full evaluation pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 EmbodimentAware Trainer Test Suite")
    print("=" * 40)
    
    # Run all tests
    test_trainer_initialization()
    test_batch_adaptation()  
    test_evaluation_modes()
    test_loss_calculation()
    test_action_visualization()
    test_full_evaluation()
    
    print(f"\n✅ All tests completed!")
    print(f"📝 Check tmp/test_* directories for generated outputs")
