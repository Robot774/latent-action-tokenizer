"""
Example: Loading EmbodimentAwareLatentMotionTokenizer from config
Demonstrates how to use the new configuration file to initialize the model
"""
import sys
import os
import yaml
import torch
from pathlib import Path

# Add the src path
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer/src')

def load_config_example():
    """Example of loading model from configuration"""
    print("📋 Loading EmbodimentAware Model from Configuration...")
    
    # Load configuration
    config_path = "/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer/configs/models/embodiment_aware_dinov2_action_encoder.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Loaded config: {Path(config_path).name}")
    
    # Display key configuration details
    print(f"\n📊 Configuration Summary:")
    print(f"  Target class: EmbodimentAwareLatentMotionTokenizer")
    print(f"  Vision encoder: DinoV2 Large")
    print(f"  Action dimension: {config['action_encoder_config']['max_action_dim']}")
    print(f"  Hidden size: {config['action_encoder_config']['hidden_size']}")
    print(f"  Embodiments: {config['action_encoder_config']['num_embodiments']}")
    print(f"  Action chunk size: {config['action_encoder_config']['action_chunk_size']}")
    
    # Show embodiment details
    embodiment_configs = config['action_encoder_config']['embodiment_configs']
    print(f"\n🤖 Supported Embodiments:")
    for emb_id, emb_config in embodiment_configs.items():
        print(f"  ID {emb_id}: {emb_config['name']} ({emb_config['action_dim']}D actions)")
    
    return config

def show_usage_pattern():
    """Show how to use the configuration in practice"""
    print(f"\n🎯 Usage Pattern:")
    print(f"```python")
    print(f"# 1. Load configuration")
    print(f"import yaml")
    print(f"with open('configs/models/embodiment_aware_dinov2_action_encoder.yaml') as f:")
    print(f"    config = yaml.safe_load(f)")
    print(f"")
    print(f"# 2. Initialize model (using hydra or manual instantiation)")
    print(f"from hydra.utils import instantiate")
    print(f"tokenizer = instantiate(config)")
    print(f"")
    print(f"# 3. Prepare data")
    print(f"cond_images = torch.randn(2, 3, 224, 224)")
    print(f"target_images = torch.randn(2, 3, 224, 224)")
    print(f"actions = torch.randn(2, 4, 48)  # (batch, chunk_size, action_dim)")
    print(f"states = torch.randn(2, 1, 48)   # (batch, 1, state_dim)")
    print(f"embodiment_ids = torch.zeros(2, dtype=torch.long)  # EgoDex")
    print(f"")
    print(f"# 4. Forward pass with action conditioning")
    print(f"outputs = tokenizer(")
    print(f"    cond_pixel_values=cond_images,")
    print(f"    target_pixel_values=target_images,")
    print(f"    actions=actions,")
    print(f"    states=states,")
    print(f"    embodiment_ids=embodiment_ids")
    print(f")")
    print(f"")
    print(f"# 5. Access results (including action decoder outputs)")
    print(f"recons = outputs['recons_pixel_values']        # Reconstructed images")
    print(f"motion_tokens = outputs['indices']             # Motion token indices")
    print(f"action_features = outputs['action_features']   # Encoded action features")
    print(f"decoded_actions = outputs['decoded_actions']   # 🆕 Decoded action sequences")
    print(f"")
    print(f"# 6. 🆕 Action reconstruction loss (automatically computed!)")
    print(f"# The model now automatically computes action reconstruction loss")
    print(f"outputs = model(cond_images, target_images, actions=actions, states=states)")
    print(f"")
    print(f"# 📊 Loss breakdown (all included in outputs['loss']):")
    print(f"total_loss = outputs['loss']  # Already includes:")
    print(f"#   = commit_loss_w × commit_loss")
    print(f"#   + recon_loss_w × recons_loss")
    print(f"#   + perceptual_loss_w × perceptual_loss")
    print(f"#   + action_recons_loss_w × action_recons_loss  # 🆕 NEW!")
    print(f"")
    print(f"# 🔍 Individual components available:")
    print(f"action_recons_loss = outputs['action_recons_loss']  # 🆕 MSE(decoded, original)")
    print(f"decoded_actions = outputs['decoded_actions']        # 🆕 (B, chunk_size, action_dim)")
    print(f"```")

def main():
    """Main example function"""
    print("🚀 EmbodimentAware Configuration Example")
    print("="*50)
    
    # Load and display config
    config = load_config_example()
    
    # Show usage pattern
    show_usage_pattern()
    
    print(f"\n✨ Key Benefits:")
    print(f"  ✅ Unified 48D action space (supports EgoDex)")
    print(f"  ✅ Embodiment-aware encoding (GR00T style)")
    print(f"  ✅ Action encoder: actions -> features")
    print(f"  ✅ Action decoder: motion tokens -> actions")
    print(f"  ✅ Visual-Action fusion with presence handling")
    print(f"  ✅ Action reconstruction loss (automatic)")
    print(f"  ✅ End-to-end action-conditioned learning")
    print(f"  ✅ Backward compatibility (works without actions)")
    print(f"  ✅ Configurable via YAML files")
    print(f"  ✅ Ready for multi-embodiment expansion")
    
    print(f"\n🔗 Integration Points:")
    print(f"  1. Update trainer to use this config")
    print(f"  2. Modify data collator to pass action parameters")
    print(f"  3. Add action data to training pipeline")
    print(f"  4. ✅ Action reconstruction loss automatically handled")
    print(f"  5. Configure loss weights in YAML: action_recons_loss_w")

if __name__ == "__main__":
    main()
