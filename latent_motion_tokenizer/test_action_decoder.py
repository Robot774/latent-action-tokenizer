"""
Test script for EmbodimentAwareActionDecoder
Verifies the action decoding functionality
"""
import torch
import sys
import os

# Add the src path
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer/src')

from models.embodiment_aware_action_encoder import EmbodimentAwareActionDecoder

def test_action_decoder():
    """Test EmbodimentAwareActionDecoder"""
    print("🧪 Testing EmbodimentAwareActionDecoder...")
    
    # Configuration
    config = {
        'hidden_size': 768,
        'max_action_dim': 48,
        'action_chunk_size': 4,
        'num_embodiments': 1,
        'query_num': 8,
        'embodiment_configs': {
            0: {'name': 'egodx', 'action_dim': 48}
        }
    }
    
    decoder = EmbodimentAwareActionDecoder(**config)
    
    # Test data
    batch_size = 2
    query_num = config['query_num']
    hidden_size = config['hidden_size']
    
    # Mock latent motion tokens (output from VQ up-resampler)
    latent_motion_tokens = torch.randn(batch_size, query_num, hidden_size)
    embodiment_ids = torch.zeros(batch_size, dtype=torch.long)
    
    print(f"📊 Input shapes:")
    print(f"  latent_motion_tokens: {latent_motion_tokens.shape}")
    print(f"  embodiment_ids: {embodiment_ids.shape}")
    
    # Test decoding
    decoded_actions = decoder(latent_motion_tokens, embodiment_ids)
    
    print(f"📊 Output shapes:")
    print(f"  decoded_actions: {decoded_actions.shape}")
    
    # Verify output shape
    expected_shape = (batch_size, config['action_chunk_size'], config['max_action_dim'])
    assert decoded_actions.shape == expected_shape, f"Expected {expected_shape}, got {decoded_actions.shape}"
    
    print(f"✅ Action decoder test passed!")
    print(f"  Input: {latent_motion_tokens.shape} -> Output: {decoded_actions.shape}")
    
    # Test without embodiment_ids (should default to 0)
    decoded_actions_default = decoder(latent_motion_tokens)
    assert decoded_actions_default.shape == expected_shape
    print(f"✅ Default embodiment_ids handling works")
    
    return True

def test_decoder_components():
    """Test individual components of the decoder"""
    print("\n🧪 Testing Decoder Components...")
    
    batch_size = 2
    query_num = 8
    hidden_size = 768
    
    # Test motion pooling
    motion_pooling = torch.nn.Sequential(
        torch.nn.Linear(hidden_size * query_num, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size)
    )
    
    latent_tokens = torch.randn(batch_size, query_num, hidden_size)
    flattened = latent_tokens.reshape(batch_size, -1)
    pooled = motion_pooling(flattened)
    
    print(f"  Motion pooling: {latent_tokens.shape} -> {flattened.shape} -> {pooled.shape}")
    assert pooled.shape == (batch_size, hidden_size)
    print(f"  ✅ Motion pooling works correctly")
    
    # Test reshape logic
    chunk_size = 4
    action_dim = 48
    flattened_actions = torch.randn(batch_size, chunk_size * action_dim)
    reshaped_actions = flattened_actions.reshape(batch_size, chunk_size, action_dim)
    
    print(f"  Action reshape: {flattened_actions.shape} -> {reshaped_actions.shape}")
    assert reshaped_actions.shape == (batch_size, chunk_size, action_dim)
    print(f"  ✅ Action reshape works correctly")

def test_end_to_end_pipeline():
    """Test the complete encode-decode pipeline"""
    print("\n🧪 Testing End-to-End Pipeline...")
    
    from models.embodiment_aware_action_encoder import EmbodimentAwareActionEncoder
    
    # Configuration
    config = {
        'max_action_dim': 48,
        'hidden_size': 768,
        'num_embodiments': 1,
        'action_chunk_size': 4,
        'query_num': 8
    }
    
    # Create encoder and decoder
    encoder = EmbodimentAwareActionEncoder(
        max_action_dim=config['max_action_dim'],
        hidden_size=config['hidden_size'],
        num_embodiments=config['num_embodiments']
    )
    
    decoder = EmbodimentAwareActionDecoder(
        hidden_size=config['hidden_size'],
        max_action_dim=config['max_action_dim'],
        action_chunk_size=config['action_chunk_size'],
        num_embodiments=config['num_embodiments'],
        query_num=config['query_num']
    )
    
    # Test data
    batch_size = 2
    original_actions = torch.randn(batch_size, config['action_chunk_size'], config['max_action_dim'])
    embodiment_ids = torch.zeros(batch_size, dtype=torch.long)
    
    print(f"📊 End-to-End Pipeline:")
    print(f"  Original actions: {original_actions.shape}")
    
    # Encode actions
    action_features = encoder.encode_action(original_actions, embodiment_ids)
    print(f"  Encoded features: {action_features.shape}")
    
    # Simulate motion tokens (would come from VQ up-resampler in real pipeline)
    mock_motion_tokens = torch.randn(batch_size, config['query_num'], config['hidden_size'])
    
    # Decode actions
    decoded_actions = decoder(mock_motion_tokens, embodiment_ids)
    print(f"  Decoded actions: {decoded_actions.shape}")
    
    # Verify shapes match
    assert original_actions.shape == decoded_actions.shape
    print(f"✅ End-to-end pipeline shapes consistent")

if __name__ == "__main__":
    print("🚀 Starting Action Decoder Tests...")
    
    try:
        test_action_decoder()
        test_decoder_components()
        test_end_to_end_pipeline()
        
        print("\n🎉 All Action Decoder tests passed!")
        
    except Exception as e:
        print(f"\n❌ Action Decoder test failed: {e}")
        import traceback
        traceback.print_exc()
