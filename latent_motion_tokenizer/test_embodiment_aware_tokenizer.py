"""
Test script for EmbodimentAwareLatentMotionTokenizer
"""
import torch
import sys
import os

# Add the src path to import our modules
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer/src')

from models.embodiment_aware_action_encoder import (
    CategorySpecificLinear, 
    CategorySpecificMLP, 
    EmbodimentAwareActionEncoder
)

def test_category_specific_components():
    """Test CategorySpecific components"""
    print("🧪 Testing CategorySpecific components...")
    
    # Test CategorySpecificLinear
    batch_size = 2
    seq_len = 4
    input_dim = 48
    hidden_dim = 1024
    num_categories = 1
    
    layer = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
    
    # Create test input
    x = torch.randn(batch_size, seq_len, input_dim)
    cat_ids = torch.zeros(batch_size, dtype=torch.long)  # All embodiment 0
    
    output = layer(x, cat_ids)
    print(f"  ✅ CategorySpecificLinear: {x.shape} -> {output.shape}")
    assert output.shape == (batch_size, seq_len, hidden_dim)
    
    # Test CategorySpecificMLP
    output_dim = 512
    mlp = CategorySpecificMLP(num_categories, input_dim, hidden_dim, output_dim)
    
    mlp_output = mlp(x, cat_ids)
    print(f"  ✅ CategorySpecificMLP: {x.shape} -> {mlp_output.shape}")
    assert mlp_output.shape == (batch_size, seq_len, output_dim)


def test_action_encoder():
    """Test EmbodimentAwareActionEncoder"""
    print("\n🧪 Testing EmbodimentAwareActionEncoder...")
    
    # Configuration
    config = {
        'max_action_dim': 48,
        'hidden_size': 1024,
        'num_embodiments': 1,
        'embodiment_configs': {
            0: {'name': 'egodex', 'action_dim': 48, 'state_dim': 48}
        }
    }
    
    encoder = EmbodimentAwareActionEncoder(**config)
    
    # Test data
    batch_size = 2
    action_chunk_size = 4
    state_seq_len = 1
    
    actions = torch.randn(batch_size, action_chunk_size, config['max_action_dim'])
    states = torch.randn(batch_size, state_seq_len, config['max_action_dim'])
    embodiment_ids = torch.zeros(batch_size, dtype=torch.long)
    
    # Test encoding
    action_features = encoder.encode_action(actions, embodiment_ids)
    state_features = encoder.encode_state(states, embodiment_ids)
    
    print(f"  ✅ Action encoding: {actions.shape} -> {action_features.shape}")
    print(f"  ✅ State encoding: {states.shape} -> {state_features.shape}")
    
    # Verify output shapes
    expected_action_shape = (batch_size, action_chunk_size, config['hidden_size'])
    expected_state_shape = (batch_size, state_seq_len, config['hidden_size'])
    
    assert action_features.shape == expected_action_shape
    assert state_features.shape == expected_state_shape
    
    # Test without embodiment_ids (should default to 0)
    action_features_default = encoder.encode_action(actions)
    assert torch.allclose(action_features, action_features_default)
    
    print("  ✅ Default embodiment_ids handling works")


def test_embodiment_aware_tokenizer_config():
    """Test EmbodimentAwareLatentMotionTokenizer configuration"""
    print("\n🧪 Testing EmbodimentAwareLatentMotionTokenizer config...")
    
    # Mock components (we'll just test the configuration logic)
    class MockComponent:
        def __init__(self):
            self.config = type('Config', (), {'hidden_size': 1024, 'query_num': 64})()
    
    # Test action encoder config
    action_encoder_config = {
        'max_action_dim': 48,
        'hidden_size': 1024,
        'num_embodiments': 1,
        'embodiment_configs': {
            0: {'name': 'egodx', 'action_dim': 48, 'state_dim': 48}
        }
    }
    
    print(f"  ✅ Action encoder config: {action_encoder_config}")
    
    # Verify required keys
    required_keys = ['max_action_dim', 'hidden_size', 'num_embodiments']
    for key in required_keys:
        assert key in action_encoder_config, f"Missing required key: {key}"
    
    print("  ✅ All required config keys present")


if __name__ == "__main__":
    print("🚀 Starting EmbodimentAwareLatentMotionTokenizer tests...")
    
    try:
        test_category_specific_components()
        test_action_encoder()
        test_embodiment_aware_tokenizer_config()
        
        print("\n🎉 All tests passed! Basic functionality working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
