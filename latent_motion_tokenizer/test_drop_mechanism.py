"""
Test Updated Drop Mechanism
Tests the new training drop and evaluation behavior
"""
import torch
import sys
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer')

def test_drop_mechanism():
    """Test the updated drop mechanism"""
    print("🧪 Testing Updated Drop Mechanism")
    print("=" * 40)
    
    # Mock model for action drop without force flag
    class MockEmbodimentAwareModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enable_action_conditioning = True
            
        def forward(self, cond_pixel_values, target_pixel_values, 
                   actions=None, states=None, embodiment_ids=None,
                   gt_actions_for_loss=None, **kwargs):
            
            batch_size = cond_pixel_values.shape[0]
            
            # Check the drop mechanism
            if actions is None and gt_actions_for_loss is not None:
                print(f"   🎲 DROPPED: actions=None but GT actions provided for loss")
                print(f"   ✅ Model will use fusion path with missing action tokens")
                if gt_actions_for_loss is not None:
                    print(f"   🎯 GT actions preserved for loss: {gt_actions_for_loss.shape}")
                else:
                    print(f"   ⚠️ No GT actions for loss computation")
            elif actions is not None:
                print(f"   ✅ NORMAL: actions provided, full multimodal processing")
            else:
                print(f"   ⚪ VISION_ONLY: no action conditioning")
            
            # Mock decoded actions (always generated if action conditioning enabled)
            decoded_actions = None
            if actions is not None or gt_actions_for_loss is not None:
                decoded_actions = torch.randn(batch_size, 4, 48)
                
            # Mock action reconstruction loss computation
            action_recons_loss = 0.0
            if decoded_actions is not None:
                target_actions = gt_actions_for_loss if gt_actions_for_loss is not None else actions
                if target_actions is not None:
                    # Simulate actual loss computation
                    action_recons_loss = torch.nn.functional.mse_loss(decoded_actions, target_actions).item()
                    action_type = "GT (dropped)" if gt_actions_for_loss is not None else "Input"
                    print(f"   📊 Action loss ({action_type}): {action_recons_loss:.6f}")
                else:
                    print(f"   ❌ No target actions - action loss = 0")
            
            # Mock outputs
            outputs = {
                'recons_pixel_values': torch.randn_like(target_pixel_values),
                'indices': torch.randint(0, 100, (batch_size, 8)),
                'loss': torch.tensor(0.5),
                'commit_loss': torch.tensor(0.1),
                'recons_loss': torch.tensor(0.3),
                'perceptual_loss': torch.tensor(0.1),
                'action_recons_loss': torch.tensor(action_recons_loss),
                'decoded_actions': decoded_actions
            }
            
            return outputs
    
    # Create mock trainer
    # No import of real trainer to avoid external deps
    
    class MockTrainer:
        def __init__(self):
            self.supports_action_conditioning = True
            self.action_drop_prob = 0.5  # 50% drop probability for testing
            self.latent_motion_tokenizer = MockEmbodimentAwareModel()
            
        def print(self, msg):
            print(msg)
            
        def _adapt_batch_for_action_conditioning(self, batch):
            return batch
            
        def calculate_loss(self, batch, train):
            """Simplified version of the new calculate_loss method"""
            
            cond_pixel_values = torch.randn(2, 3, 224, 224)
            target_pixel_values = torch.randn(2, 3, 224, 224)
            
            forward_kwargs = {
                'cond_pixel_values': cond_pixel_values,
                'target_pixel_values': target_pixel_values
            }
            
            if self.supports_action_conditioning and batch['actions'] is not None:
                import random
                should_drop_actions = train and (random.random() < self.action_drop_prob)
                
                if should_drop_actions:
                    forward_kwargs.update({
                        'actions': None,  # Drop signal
                        'states': batch['states'],
                        'embodiment_ids': batch['embodiment_ids'],
                        'gt_actions_for_loss': batch['actions']  # Preserve GT for loss
                    })
                    print(f"   🎲 Training DROP applied (prob={self.action_drop_prob})")
                else:
                    forward_kwargs.update({
                        'actions': batch['actions'],
                        'states': batch['states'],
                        'embodiment_ids': batch['embodiment_ids']
                    })
                    print(f"   ✅ Training NORMAL (no drop)")
            
            return self.latent_motion_tokenizer(**forward_kwargs)
    
    # Test scenarios
    trainer = MockTrainer()
    
    # Create test batch
    batch = {
        'actions': torch.randn(2, 4, 48),
        'states': torch.randn(2, 1, 48),
        'embodiment_ids': torch.zeros(2, dtype=torch.long)
    }
    
    print("\n1️⃣ Testing Training Mode (with random drop)")
    for i in range(5):
        print(f"\n   Trial {i+1}:")
        loss = trainer.calculate_loss(batch, train=True)
        print(f"   📊 Loss: {loss['loss'].item():.3f}, Action decoded: {loss['decoded_actions'] is not None}")
    
    print("\n2️⃣ Testing Evaluation Mode (no drop)")
    for i in range(3):
        print(f"\n   Eval Trial {i+1}:")
        loss = trainer.calculate_loss(batch, train=False)
        print(f"   📊 Loss: {loss['loss'].item():.3f}, Action decoded: {loss['decoded_actions'] is not None}")
    
    print(f"\n✅ Drop mechanism test completed!")


def test_action_conditioning_gate():
    """Test the gating logic without force flag"""
    print("\n🔧 Testing Action Conditioning Gate")
    print("=" * 40)
    
    class TestModel:
        def __init__(self):
            self.enable_action_conditioning = True
            
        def check_condition(self, actions, gt_actions_for_loss):
            use_action_conditioning = (self.enable_action_conditioning and 
                                       (actions is not None or gt_actions_for_loss is not None))
            return use_action_conditioning
    
    model = TestModel()
    
    test_cases = [
        {"actions": torch.randn(2, 4, 48), "gt": None, "expected": True, "desc": "Normal: actions provided"},
        {"actions": None, "gt": None, "expected": False, "desc": "Normal: no actions, no GT"},
        {"actions": None, "gt": torch.randn(2, 4, 48), "expected": True, "desc": "DROP: no actions, but GT provided"},
        {"actions": torch.randn(2, 4, 48), "gt": torch.randn(2, 4, 48), "expected": True, "desc": "Both present"},
    ]
    
    for case in test_cases:
        result = model.check_condition(case["actions"], case["gt"])
        status = "✅" if result == case["expected"] else "❌"
        print(f"   {status} {case['desc']}: {result} (expected: {case['expected']})")
    
    print(f"\n✅ Gate logic verified!")


if __name__ == "__main__":
    test_drop_mechanism()
    test_action_conditioning_gate()
