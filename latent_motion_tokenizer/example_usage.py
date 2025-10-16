"""
Usage example for EmbodimentAwareLatentMotionTokenizer
Demonstrates how to integrate action conditioning into latent motion tokenization
"""
import torch
import sys
import os

# Add the src path
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer/src')

def create_example_config():
    """Create example configuration for action encoder and decoder"""
    action_encoder_config = {
        'max_action_dim': 48,           # 统一动作维度 (支持EgoDex 48D)
        'hidden_size': 768,             # 与m_former保持一致 (updated to 768)
        'num_embodiments': 1,           # 暂时只支持1个embodiment
        'action_chunk_size': 4,         # 🆕 Action sequence length for decoder
        'embodiment_configs': {
            0: {
                'name': 'egodx', 
                'action_dim': 48, 
                'state_dim': 48,
                'description': 'EgoDex 48D hand action representation'
            }
        }
    }
    
    return action_encoder_config

def example_usage():
    """
    Example usage of EmbodimentAwareLatentMotionTokenizer
    """
    print("📋 EmbodimentAwareLatentMotionTokenizer Usage Example")
    print("=" * 60)
    
    # 1. Configuration
    config = create_example_config()
    print(f"🔧 Action Encoder Config:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    # 2. Mock input data (simulate real data shapes)
    batch_size = 2
    action_chunk_size = 4  # 4帧action sequence
    
    print(f"\n📊 Input Data Shapes:")
    print(f"   Batch size: {batch_size}")
    print(f"   Action chunk size: {action_chunk_size}")
    
    # Visual inputs (mock)
    cond_pixel_values = torch.randn(batch_size, 3, 224, 224)
    target_pixel_values = torch.randn(batch_size, 3, 224, 224)
    
    # Action inputs (mock - already padded to unified dimension)
    actions = torch.randn(batch_size, action_chunk_size, config['max_action_dim'])
    states = torch.randn(batch_size, 1, config['max_action_dim'])  # Current state
    embodiment_ids = torch.zeros(batch_size, dtype=torch.long)    # All EgoDex
    
    print(f"   cond_pixel_values: {cond_pixel_values.shape}")
    print(f"   target_pixel_values: {target_pixel_values.shape}")
    print(f"   actions: {actions.shape}")
    print(f"   states: {states.shape}")
    print(f"   embodiment_ids: {embodiment_ids.shape}")
    
    # 3. Usage patterns
    print(f"\n🎯 Usage Patterns:")
    
    print("\n   Pattern 1: Action conditioning with encoder & decoder")
    print("   ```python")
    print("   tokenizer = EmbodimentAwareLatentMotionTokenizer(")
    print("       image_encoder=image_encoder,")
    print("       m_former=m_former,") 
    print("       vector_quantizer=vector_quantizer,")
    print("       decoder=decoder,")
    print("       action_encoder_config=action_encoder_config  # Enables both encoder & decoder")
    print("   )")
    print("   ")
    print("   # Forward with action conditioning")
    print("   outputs = tokenizer(")
    print("       cond_pixel_values=cond_images,")
    print("       target_pixel_values=target_images,")
    print("       actions=actions,              # (B, chunk_size, 48) - Input actions")
    print("       states=states,                # (B, 1, 48)")
    print("       embodiment_ids=embodiment_ids # (B,)")
    print("   )")
    print("   ")
    print("   # 🆕 Access encoded and decoded actions")
    print("   input_actions = actions                        # (B, 4, 48) - Ground truth")
    print("   action_features = outputs['action_features']   # (B, 4, 768) - Encoded features")
    print("   decoded_actions = outputs['decoded_actions']   # (B, 4, 48) - Reconstructed actions")
    print("   ")
    print("   # 🆕 Compute action reconstruction loss")
    print("   import torch.nn.functional as F")
    print("   action_recon_loss = F.mse_loss(decoded_actions, input_actions)")
    print("   ```")
    
    print("\n   Pattern 2: Backward compatibility (no action)")
    print("   ```python")
    print("   # Without action inputs - behaves like original tokenizer")
    print("   outputs = tokenizer(")
    print("       cond_pixel_values=cond_images,")
    print("       target_pixel_values=target_images")
    print("   )")
    print("   ```")
    
    print("\n   Pattern 3: Action encoder disabled")
    print("   ```python")
    print("   tokenizer = EmbodimentAwareLatentMotionTokenizer(")
    print("       image_encoder=image_encoder,")
    print("       m_former=m_former,")
    print("       vector_quantizer=vector_quantizer,")
    print("       decoder=decoder")
    print("       # action_encoder_config=None (default)")
    print("   )")
    print("   ```")
    
    # 4. Expected outputs
    print(f"\n📤 Expected Outputs:")
    print("   outputs = {")
    print("       'recons_pixel_values': Tensor,  # Reconstructed images")
    print("       'indices': Tensor,              # Motion token indices") 
    print("       'loss': Tensor,                 # Total loss")
    print("       'commit_loss': Tensor,          # VQ commitment loss")
    print("       'recons_loss': Tensor,          # Reconstruction loss")
    print("       'perceptual_loss': Tensor,      # Perceptual loss")
    print("       'action_recons_loss': Tensor,   # 🆕 Action reconstruction loss")
    print("       'action_features': Tensor,      # 🆕 Encoded action features (B, chunk_size, hidden_size)")
    print("       'state_features': Tensor,       # 🆕 Encoded state features (B, 1, hidden_size)")
    print("       'action_tokens': Tensor,        # 🆕 A-Former output tokens (B, query_num, hidden_size)")
    print("       'decoded_actions': Tensor,      # 🆕 Decoded action sequences (B, chunk_size, action_dim)")
    print("   }")
    print("")
    print("   🔄 Data Flow:")
    print("   Input Actions (B, 4, 48) -> Action Encoder -> Action Features (B, 4, 768)")
    print("   Input States (B, 1, 48) -> State Encoder -> State Features (B, 1, 768)")
    print("                                     ↓")
    print("   Action Features + State Features -> A-Former -> Action Tokens (B, 8, 768)")
    print("                                     ↓")
    print("   Visual Motion -> M-Former -> Visual Tokens (B, 8, 768)")
    print("                                     ↓")
    print("   🆕 Visual-Action Fusion -> Fused Tokens (B, 8, 768) -> VQ -> Quantized Tokens")
    print("                                     ↓")
    print("   Quantized Tokens -> Action Decoder -> Decoded Actions (B, 4, 48)")
    
    # 5. Fusion details
    print(f"\n🔀 Visual-Action Fusion Details:")
    print("   • Handles missing modalities with learnable tokens")
    print("   • Cross-attention for information complementarity:")
    print("     - Visual attend to Action: enhanced visual features")
    print("     - Action attend to Visual: enhanced action features") 
    print("   • Gated fusion with presence awareness:")
    print("     - pv: visual presence flag (0=missing, 1=present)")
    print("     - pa: action presence flag (0=missing, 1=present)")
    print("     - Gate network considers presence patterns")
    print("   • Supports 4 scenarios: both, visual-only, action-only, neither")
    
    # 6. Integration points
    print(f"\n🔗 Integration Points:")
    print("   1. Collate函数需要传递actions, states, embodiment_ids")
    print("   2. Trainer需要处理新的输出字段") 
    print("   3. 配置文件需要添加action_encoder_config section")
    print("   4. ✅ Fusion逻辑已实现，支持模态缺失处理")
    
    # 7. Next steps
    print(f"\n🚀 Next Steps:")
    print("   ✅ Phase 1: Basic action encoder integration (DONE)")
    print("   ✅ Phase 2: Action decoder integration (DONE)")
    print("   ✅ Phase 3: Visual-Action fusion (DONE)")
    print("   ✅ Phase 4: Action reconstruction loss (DONE)")
    print("   ✅ Phase 5: A-Former integration (DONE)")
    print("   🔄 Phase 6: Multi-embodiment support (TODO)")
    print("   🔄 Phase 7: Full model training integration (TODO)")
    
    # 8. Loss components breakdown
    print(f"\n💰 Loss Components:")
    print("   📊 Total Loss = commit_loss_w × commit_loss")
    print("                 + recon_loss_w × recons_loss") 
    print("                 + perceptual_loss_w × perceptual_loss")
    print("                 + action_recons_loss_w × action_recons_loss  # 🆕")
    print("   🎯 Action Reconstruction Loss:")
    print("      • MSE(decoded_actions, original_actions)")
    print("      • 促进action-conditioned视觉表征学习")
    print("      • 提高动作预测准确性")
    print("      • 权重可通过配置文件调节")

if __name__ == "__main__":
    example_usage()
