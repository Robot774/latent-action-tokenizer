"""
A-Former Usage Example
Demonstrates the complete action processing pipeline with A-Former
"""
import torch
import sys
import os

# Add the src path
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer/src')

from transformers import ViTConfig
from models.a_former import AFormer
from models.embodiment_aware_action_encoder import EmbodimentAwareActionEncoder

def create_aformer_config():
    """Create A-Former configuration"""
    config = ViTConfig(
        query_num=8,                    # Output 8 latent action tokens
        action_chunk_size=4,            # Process 4-frame action sequences
        hidden_size=768,                # Hidden dimension
        num_hidden_layers=2,            # 2-layer transformer (lightweight)
        num_attention_heads=12,         # Standard attention heads
        intermediate_size=3072,         # FFN intermediate size
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        legacy=True
    )
    return config

def demonstrate_action_pipeline():
    """Demonstrate the complete action processing pipeline"""
    print("🚀 A-Former Complete Action Pipeline Demo")
    print("="*50)
    
    # 1. Create components
    print("🔧 Setting up components...")
    
    # Action encoder configuration
    action_encoder_config = {
        'max_action_dim': 48,
        'hidden_size': 768,
        'num_embodiments': 1,
        'embodiment_configs': {
            0: {'name': 'egodx', 'action_dim': 48, 'state_dim': 48}
        }
    }
    
    # Create action encoder and A-Former
    action_encoder = EmbodimentAwareActionEncoder(**action_encoder_config)
    aformer_config = create_aformer_config()
    aformer = AFormer(aformer_config)
    
    print(f"✅ Action Encoder: {action_encoder_config['max_action_dim']}D -> {action_encoder_config['hidden_size']}D")
    print(f"✅ A-Former: 2-layer transformer with {aformer_config.query_num} queries")
    
    # 2. Prepare test data
    print(f"\n📊 Preparing test data...")
    batch_size = 3
    chunk_size = 4
    action_dim = 48
    
    # Raw action and state data (as would come from dataset)
    raw_actions = torch.randn(batch_size, chunk_size, action_dim)
    raw_states = torch.randn(batch_size, 1, action_dim)
    embodiment_ids = torch.zeros(batch_size, dtype=torch.long)  # All EgoDx
    
    print(f"  Raw actions: {raw_actions.shape}")
    print(f"  Raw states: {raw_states.shape}")
    print(f"  Embodiment IDs: {embodiment_ids.shape}")
    
    # 3. Step-by-step processing
    print(f"\n🔄 Step-by-step processing...")
    
    # Step 1: Action Encoding
    print("  Step 1: Action & State Encoding")
    action_features = action_encoder.encode_action(raw_actions, embodiment_ids)
    state_features = action_encoder.encode_state(raw_states, embodiment_ids)
    
    print(f"    Actions {raw_actions.shape} -> {action_features.shape}")
    print(f"    States {raw_states.shape} -> {state_features.shape}")
    
    # Step 2: A-Former Processing
    print("  Step 2: A-Former Processing")
    outputs = aformer(state_features, action_features)
    latent_action_tokens = outputs.last_hidden_state
    
    print(f"    [State + Action Features] -> A-Former -> {latent_action_tokens.shape}")
    
    # 4. Results analysis
    print(f"\n📋 Results Analysis:")
    print(f"  Input sequence length: {chunk_size + 1} (4 actions + 1 state)")
    print(f"  A-Former internal sequence: {aformer_config.query_num + 1 + 1 + chunk_size} tokens")
    print(f"    - {aformer_config.query_num} learnable queries")
    print(f"    - 1 state token") 
    print(f"    - 1 separator token")
    print(f"    - {chunk_size} action tokens")
    print(f"  Output: {latent_action_tokens.shape[1]} latent action tokens")
    
    # 5. Compare with visual tokens
    print(f"\n🔗 Integration with Visual Pipeline:")
    print(f"  Visual M-Former output: (B, 8, 768) - latent motion tokens")
    print(f"  Action A-Former output: (B, 8, 768) - latent action tokens")
    print(f"  -> Ready for fusion or parallel processing!")
    
    return {
        'raw_actions': raw_actions,
        'raw_states': raw_states,
        'action_features': action_features,
        'state_features': state_features,
        'latent_action_tokens': latent_action_tokens
    }

def show_aformer_architecture():
    """Show A-Former architecture details"""
    print(f"\n🏗️ A-Former Architecture:")
    print(f"```")
    print(f"Input:")
    print(f"  state_features: (B, 1, 768)")
    print(f"  action_features: (B, 4, 768)")
    print(f"")
    print(f"AFormerEmbeddings:")
    print(f"  1. Learnable Queries: (B, 8, 768)")
    print(f"  2. State Features: (B, 1, 768)")
    print(f"  3. Separator Token: (B, 1, 768)")
    print(f"  4. Action Features: (B, 4, 768)")
    print(f"  -> Combined: (B, 14, 768)")
    print(f"  + Position Embeddings")
    print(f"  + Token Type Embeddings (state vs action)")
    print(f"")
    print(f"Transformer (2 layers):")
    print(f"  - Multi-head attention (12 heads)")
    print(f"  - Feed-forward networks")
    print(f"  - Layer normalization")
    print(f"")
    print(f"Output:")
    print(f"  latent_action_tokens: (B, 8, 768)  # First 8 tokens")
    print(f"```")

def show_integration_possibilities():
    """Show how A-Former integrates with the larger system"""
    print(f"\n🔗 Integration Possibilities:")
    
    print(f"\n1. 🎯 Current Integration (Phase 1):")
    print(f"   Visual: Images -> M-Former -> Visual Tokens (B, 8, 768)")
    print(f"   Action: Actions -> A-Former -> Action Tokens (B, 8, 768)")
    print(f"   -> Separate processing, no fusion yet")
    
    print(f"\n2. 🔄 Future Integration (Phase 2):")
    print(f"   Option A - Concatenation:")
    print(f"     Combined = concat([Visual Tokens, Action Tokens], dim=1)")
    print(f"     Shape: (B, 16, 768)")
    
    print(f"\n   Option B - Cross-Attention:")
    print(f"     Fused = CrossAttention(Visual Tokens, Action Tokens)")
    print(f"     Shape: (B, 8, 768)")
    
    print(f"\n   Option C - Element-wise Fusion:")
    print(f"     Fused = Visual Tokens + Action Tokens")
    print(f"     Shape: (B, 8, 768)")
    
    print(f"\n3. 🎓 Training Strategy:")
    print(f"   - Joint training with visual reconstruction loss")
    print(f"   - Action reconstruction loss via action decoder")
    print(f"   - Cross-modal alignment through shared representations")

def main():
    """Main demonstration"""
    # Run the complete pipeline demo
    results = demonstrate_action_pipeline()
    
    # Show architecture details
    show_aformer_architecture()
    
    # Show integration possibilities
    show_integration_possibilities()
    
    print(f"\n✨ A-Former Summary:")
    print(f"  ✅ Converts action sequences to latent tokens")
    print(f"  ✅ Matches visual token dimensions for easy fusion")
    print(f"  ✅ Lightweight 2-layer design")
    print(f"  ✅ Embodiment-aware through action encoder")
    print(f"  ✅ Ready for multi-modal training")
    
    print(f"\n🚀 Next Steps:")
    print(f"  1. Integrate A-Former into EmbodimentAwareLatentMotionTokenizer")
    print(f"  2. Implement fusion strategies")
    print(f"  3. Add action reconstruction supervision")
    print(f"  4. Train end-to-end with visual+action data")

if __name__ == "__main__":
    main()
