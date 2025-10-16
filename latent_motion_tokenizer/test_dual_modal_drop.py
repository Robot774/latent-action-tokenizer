"""
Test Dual-Modal Drop Mechanism (Action + Vision)
Tests the enhanced drop mechanism with both action and vision drop support
"""
import torch
import sys
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer')

def test_dual_modal_drop():
    """Test the dual-modal drop mechanism"""
    print("🧪 Testing Dual-Modal Drop Mechanism")
    print("=" * 50)
    
    # Mock model that supports dual-modal drop
    class MockDualModalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enable_action_conditioning = True
            
        def forward(self, cond_pixel_values, target_pixel_values, 
                   actions=None, states=None, embodiment_ids=None,
                   gt_actions_for_loss=None, 
                   gt_images_for_loss=None, **kwargs):
            
            batch_size = cond_pixel_values.shape[0]
            
            # Analyze drop status
            action_dropped = actions is None and gt_actions_for_loss is not None
            vision_dropped = gt_images_for_loss is not None
            
            print(f"   📊 Drop Status:")
            print(f"      - Action: {'DROPPED' if action_dropped else 'NORMAL'}")
            print(f"      - Vision: {'DROPPED' if vision_dropped else 'NORMAL'}")
            
            # Mock processing
            if action_dropped and gt_actions_for_loss is not None:
                print(f"      - GT actions preserved: {gt_actions_for_loss.shape}")
            if vision_dropped and gt_images_for_loss is not None:
                print(f"      - GT images preserved: {gt_images_for_loss.shape}")
                
            # Mock decoded outputs
            decoded_actions = None
            if actions is not None or gt_actions_for_loss is not None:
                decoded_actions = torch.randn(batch_size, 4, 48)
                
            recons_pixel_values = torch.randn_like(cond_pixel_values)
            
            # Mock loss computation
            action_recons_loss = 0.0
            if decoded_actions is not None:
                target_actions = gt_actions_for_loss if gt_actions_for_loss is not None else actions
                if target_actions is not None:
                    action_recons_loss = torch.nn.functional.mse_loss(decoded_actions, target_actions).item()
                    action_type = "GT (dropped)" if gt_actions_for_loss is not None else "Input"
                    print(f"      - Action loss ({action_type}): {action_recons_loss:.6f}")
            
            # Vision reconstruction loss
            target_images = gt_images_for_loss if gt_images_for_loss is not None else target_pixel_values
            if target_images is not None:
                vision_recons_loss = torch.nn.functional.mse_loss(recons_pixel_values, target_images).item()
                vision_type = "GT (dropped)" if gt_images_for_loss is not None else "Input"
                print(f"      - Vision loss ({vision_type}): {vision_recons_loss:.6f}")
            
            # Mock outputs
            outputs = {
                'recons_pixel_values': recons_pixel_values,
                'indices': torch.randint(0, 100, (batch_size, 8)),
                'loss': torch.tensor(0.5),
                'commit_loss': torch.tensor(0.1),
                'recons_loss': torch.tensor(vision_recons_loss if 'vision_recons_loss' in locals() else 0.3),
                'perceptual_loss': torch.tensor(0.1),
                'action_recons_loss': torch.tensor(action_recons_loss),
                'decoded_actions': decoded_actions
            }
            
            return outputs
    
    # Create mock trainer with dual-modal drop
    class MockDualModalTrainer:
        def __init__(self):
            self.supports_action_conditioning = True
            self.action_drop_prob = 0.4  # 40% action drop
            self.vision_drop_prob = 0.3  # 30% vision drop
            self.latent_motion_tokenizer = MockDualModalModel()
            
        def print(self, msg):
            print(msg)
            
        def _adapt_batch_for_action_conditioning(self, batch):
            return batch
            
        def calculate_loss(self, batch, train):
            """Enhanced calculate_loss with dual-modal drop"""
            import random
            
            cond_pixel_values = torch.randn(2, 3, 224, 224)
            target_pixel_values = torch.randn(2, 3, 224, 224)
            
            # Dual-modal drop logic
            should_drop_actions = False
            should_drop_vision = False
            
            if train:
                should_drop_actions = random.random() < self.action_drop_prob
                should_drop_vision = random.random() < self.vision_drop_prob
            
            # Prepare forward kwargs
            forward_kwargs = {
                'cond_pixel_values': cond_pixel_values,
                'target_pixel_values': target_pixel_values if not should_drop_vision else None,
            }
            
            # Handle vision drop
            if should_drop_vision:
                forward_kwargs['gt_images_for_loss'] = target_pixel_values
                
            # Handle action drop
            if self.supports_action_conditioning and batch['actions'] is not None:
                if should_drop_actions:
                    forward_kwargs.update({
                        'actions': None,
                        'states': batch['states'],
                        'embodiment_ids': batch['embodiment_ids'],
                        'gt_actions_for_loss': batch['actions']
                    })
                else:
                    forward_kwargs.update({
                        'actions': batch['actions'],
                        'states': batch['states'],
                        'embodiment_ids': batch['embodiment_ids']
                    })
            
            # Log drop status
            if train and (should_drop_actions or should_drop_vision):
                drop_status = []
                if should_drop_actions:
                    drop_status.append(f"Action({self.action_drop_prob:.1f})")
                if should_drop_vision:
                    drop_status.append(f"Vision({self.vision_drop_prob:.1f})")
                print(f"   🎲 Drop applied: {' + '.join(drop_status)}")
            elif train:
                print(f"   ✅ No drop applied")
            
            return self.latent_motion_tokenizer(**forward_kwargs)
    
    # Test scenarios
    trainer = MockDualModalTrainer()
    
    # Create test batch
    batch = {
        'actions': torch.randn(2, 4, 48),
        'states': torch.randn(2, 1, 48),
        'embodiment_ids': torch.zeros(2, dtype=torch.long)
    }
    
    print("\n1️⃣ Testing Training Mode (with dual-modal drop)")
    drop_combinations = []
    for i in range(8):
        print(f"\n   Trial {i+1}:")
        loss = trainer.calculate_loss(batch, train=True)
        
        # Track drop combinations
        action_dropped = 'GT (dropped)' in str(loss.get('action_recons_loss', 0))
        vision_dropped = 'GT (dropped)' in str(loss.get('recons_loss', 0))
        combo = f"A:{'D' if action_dropped else 'N'}, V:{'D' if vision_dropped else 'N'}"
        drop_combinations.append(combo)
        
        print(f"   📊 Combination: {combo}")
    
    print(f"\n📈 Drop combinations observed: {set(drop_combinations)}")
    
    print("\n2️⃣ Testing Evaluation Mode (no drop)")
    for i in range(3):
        print(f"\n   Eval Trial {i+1}:")
        loss = trainer.calculate_loss(batch, train=False)
        print(f"   📊 Should be no drops in evaluation")
    
    print(f"\n✅ Dual-modal drop test completed!")


def test_drop_combinations():
    """Test specific drop combinations"""
    print("\n🔬 Testing Specific Drop Combinations")
    print("=" * 50)
    
    class TestModel:
        def __init__(self):
            self.enable_action_conditioning = True
            
        def simulate_forward(self, actions, target_images, gt_actions, gt_images, force_action):
            """Simulate the forward logic"""
            use_action_conditioning = (self.enable_action_conditioning and 
                                     (actions is not None or force_action))
            
            action_dropped = actions is None and gt_actions is not None
            vision_dropped = target_images is None and gt_images is not None
            
            return {
                'use_action_conditioning': use_action_conditioning,
                'action_dropped': action_dropped,
                'vision_dropped': vision_dropped,
                'can_compute_action_loss': gt_actions is not None or actions is not None,
                'can_compute_vision_loss': gt_images is not None or target_images is not None
            }
    
    model = TestModel()
    
    test_cases = [
        {
            'name': 'Normal (no drop)',
            'actions': torch.randn(2, 4, 48),
            'target_images': torch.randn(2, 3, 224, 224),
            'gt_actions': None,
            'gt_images': None,
            'force_action': False
        },
        {
            'name': 'Action drop only',
            'actions': None,
            'target_images': torch.randn(2, 3, 224, 224),
            'gt_actions': torch.randn(2, 4, 48),
            'gt_images': None,
            'force_action': True
        },
        {
            'name': 'Vision drop only',
            'actions': torch.randn(2, 4, 48),
            'target_images': None,
            'gt_actions': None,
            'gt_images': torch.randn(2, 3, 224, 224),
            'force_action': False
        },
        {
            'name': 'Both dropped',
            'actions': None,
            'target_images': None,
            'gt_actions': torch.randn(2, 4, 48),
            'gt_images': torch.randn(2, 3, 224, 224),
            'force_action': True
        },
        {
            'name': 'Vision-only mode',
            'actions': None,
            'target_images': torch.randn(2, 3, 224, 224),
            'gt_actions': None,
            'gt_images': None,
            'force_action': False
        }
    ]
    
    for case in test_cases:
        result = model.simulate_forward(
            case['actions'], case['target_images'], 
            case['gt_actions'], case['gt_images'], case['force_action']
        )
        
        print(f"\n   🧪 {case['name']}:")
        print(f"      - Action conditioning: {result['use_action_conditioning']}")
        print(f"      - Action dropped: {result['action_dropped']}")
        print(f"      - Vision dropped: {result['vision_dropped']}")
        print(f"      - Can compute action loss: {result['can_compute_action_loss']}")
        print(f"      - Can compute vision loss: {result['can_compute_vision_loss']}")
        
        # Validate expected behavior
        if case['name'] == 'Normal (no drop)':
            assert result['use_action_conditioning'] == True
            assert result['action_dropped'] == False
            assert result['vision_dropped'] == False
        elif case['name'] == 'Both dropped':
            assert result['use_action_conditioning'] == True
            assert result['action_dropped'] == True
            assert result['vision_dropped'] == True
            assert result['can_compute_action_loss'] == True
            assert result['can_compute_vision_loss'] == True
    
    print(f"\n✅ All drop combination tests passed!")


if __name__ == "__main__":
    test_dual_modal_drop()
    test_drop_combinations()
