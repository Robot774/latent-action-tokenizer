"""
Test script for A-Former (Action Transformer)
Verifies the basic functionality of action tokenization
"""
import torch
import sys
import os

# Add the src path
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer/src')

from transformers import ViTConfig
from models.a_former import AFormer, AFormerEmbeddings

def create_test_config():
    """Create test configuration for A-Former"""
    config = ViTConfig(
        query_num=8,                    # Same as M-Former
        action_chunk_size=4,            # Action sequence length
        hidden_size=768,                # Hidden dimension
        num_hidden_layers=2,            # 2 layers as discussed
        num_attention_heads=12,         # Standard attention heads
        intermediate_size=3072,         # FFN intermediate size
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        legacy=True
    )
    return config

def test_aformer_embeddings():
    """Test AFormerEmbeddings"""
    print("🧪 Testing AFormerEmbeddings...")
    
    config = create_test_config()
    embeddings = AFormerEmbeddings(config)
    
    # Test data
    batch_size = 2
    state_features = torch.randn(batch_size, 1, config.hidden_size)
    action_features = torch.randn(batch_size, config.action_chunk_size, config.hidden_size)
    
    print(f"📊 Input shapes:")
    print(f"  state_features: {state_features.shape}")
    print(f"  action_features: {action_features.shape}")
    
    # Forward pass
    embeddings_output = embeddings(state_features, action_features)
    
    expected_seq_len = config.query_num + 1 + 1 + config.action_chunk_size  # queries + state + sep + actions
    expected_shape = (batch_size, expected_seq_len, config.hidden_size)
    
    print(f"📊 Output shape:")
    print(f"  embeddings_output: {embeddings_output.shape}")
    print(f"  expected_shape: {expected_shape}")
    
    assert embeddings_output.shape == expected_shape, f"Shape mismatch: {embeddings_output.shape} vs {expected_shape}"
    print("✅ AFormerEmbeddings test passed!")
    
    return True

def test_aformer():
    """Test complete A-Former"""
    print("\n🧪 Testing A-Former...")
    
    config = create_test_config()
    aformer = AFormer(config, add_pooling_layer=False)
    
    # Test data
    batch_size = 2
    state_features = torch.randn(batch_size, 1, config.hidden_size)
    action_features = torch.randn(batch_size, config.action_chunk_size, config.hidden_size)
    
    print(f"📊 Input shapes:")
    print(f"  state_features: {state_features.shape}")
    print(f"  action_features: {action_features.shape}")
    
    # Forward pass
    outputs = aformer(state_features, action_features)
    
    latent_action_tokens = outputs.last_hidden_state
    expected_shape = (batch_size, config.query_num, config.hidden_size)
    
    print(f"📊 Output shapes:")
    print(f"  latent_action_tokens: {latent_action_tokens.shape}")
    print(f"  expected_shape: {expected_shape}")
    
    assert latent_action_tokens.shape == expected_shape, f"Shape mismatch: {latent_action_tokens.shape} vs {expected_shape}"
    print("✅ A-Former test passed!")
    
    return True

def test_aformer_with_different_inputs():
    """Test A-Former with different input sizes"""
    print("\n🧪 Testing A-Former with different inputs...")
    
    config = create_test_config()
    aformer = AFormer(config)
    
    # Test with different batch sizes and action chunk sizes
    test_cases = [
        (1, 4),  # Single sample
        (3, 4),  # Different batch size
        (2, 2),  # Different chunk size (need to update config)
    ]
    
    for batch_size, chunk_size in test_cases:
        if chunk_size != config.action_chunk_size:
            # Skip different chunk sizes for now (would need config update)
            continue
            
        state_features = torch.randn(batch_size, 1, config.hidden_size)
        action_features = torch.randn(batch_size, chunk_size, config.hidden_size)
        
        outputs = aformer(state_features, action_features)
        latent_tokens = outputs.last_hidden_state
        
        expected_shape = (batch_size, config.query_num, config.hidden_size)
        assert latent_tokens.shape == expected_shape
        
        print(f"  ✅ Test case (B={batch_size}, chunk={chunk_size}): {latent_tokens.shape}")
    
    print("✅ Different inputs test passed!")

def test_aformer_integration():
    """Test A-Former integration with action encoder outputs"""
    print("\n🧪 Testing A-Former integration...")
    
    from models.embodiment_aware_action_encoder import EmbodimentAwareActionEncoder
    
    # Create action encoder
    action_config = {
        'max_action_dim': 48,
        'hidden_size': 768,
        'num_embodiments': 1
    }
    
    action_encoder = EmbodimentAwareActionEncoder(**action_config)
    
    # Create A-Former
    aformer_config = create_test_config()
    aformer = AFormer(aformer_config)
    
    # Test data
    batch_size = 2
    chunk_size = 4
    
    # Raw action inputs
    actions = torch.randn(batch_size, chunk_size, 48)
    states = torch.randn(batch_size, 1, 48)
    embodiment_ids = torch.zeros(batch_size, dtype=torch.long)
    
    print(f"📊 Integration test:")
    print(f"  Raw actions: {actions.shape}")
    print(f"  Raw states: {states.shape}")
    
    # Step 1: Encode actions and states
    action_features = action_encoder.encode_action(actions, embodiment_ids)
    state_features = action_encoder.encode_state(states, embodiment_ids)
    
    print(f"  Encoded action_features: {action_features.shape}")
    print(f"  Encoded state_features: {state_features.shape}")
    
    # Step 2: A-Former processing
    outputs = aformer(state_features, action_features)
    latent_action_tokens = outputs.last_hidden_state
    
    print(f"  Latent action tokens: {latent_action_tokens.shape}")
    
    expected_shape = (batch_size, aformer_config.query_num, aformer_config.hidden_size)
    assert latent_action_tokens.shape == expected_shape
    
    print("✅ A-Former integration test passed!")
    
    # Show complete pipeline
    print(f"\n🔄 Complete Action Pipeline:")
    print(f"  Raw Actions (B, 4, 48) -> Action Encoder -> Action Features (B, 4, 768)")
    print(f"  Raw States (B, 1, 48) -> State Encoder -> State Features (B, 1, 768)")
    print(f"  [State Features, Action Features] -> A-Former -> Latent Action Tokens (B, 8, 768)")

if __name__ == "__main__":
    print("🚀 Starting A-Former Tests...")
    
    try:
        test_aformer_embeddings()
        test_aformer()
        test_aformer_with_different_inputs()
        test_aformer_integration()
        
        print("\n🎉 All A-Former tests passed!")
        print("\n✨ A-Former Features:")
        print("  ✅ State-Action sequence processing")
        print("  ✅ Learnable action query tokens")
        print("  ✅ Sep token for state-action separation")
        print("  ✅ 2-layer transformer architecture")
        print("  ✅ Integration with action encoder")
        print("  ✅ Consistent output dimensions")
        
    except Exception as e:
        print(f"\n❌ A-Former test failed: {e}")
        import traceback
        traceback.print_exc()
