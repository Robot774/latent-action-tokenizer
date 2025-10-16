"""
Simple test for DinoV2 EmbodimentAware configuration
"""
import sys
import os
import yaml
from pathlib import Path

# Add the src path
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer/src')

def test_dinov2_config():
    """Test DinoV2 configuration loading and validation"""
    print("🧪 Testing DinoV2 EmbodimentAware Configuration...")
    
    config_path = "/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer/configs/models/embodiment_aware_dinov2_action_encoder.yaml"
    
    try:
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✅ Successfully loaded config from {Path(config_path).name}")
        
        # Check target class
        target = config['_target_']
        print(f"📋 Target class: {target}")
        assert 'EmbodimentAwareLatentMotionTokenizer' in target
        print("✅ Correct target class")
        
        # Check action encoder config
        action_config = config['action_encoder_config']
        print(f"📋 Action encoder config keys: {list(action_config.keys())}")
        
        required_keys = ['max_action_dim', 'hidden_size', 'num_embodiments']
        for key in required_keys:
            assert key in action_config
            print(f"  ✅ {key}: {action_config[key]}")
        
        # Check embodiment config
        embodiment_configs = action_config['embodiment_configs']
        print(f"📋 Embodiment configs: {embodiment_configs}")
        
        # Check dimension consistency
        m_former_hidden = config['m_former']['config']['hidden_size']
        action_hidden = action_config['hidden_size']
        decoder_hidden = config['decoder']['config']['hidden_size']
        
        print(f"📏 Dimension check:")
        print(f"  M-Former: {m_former_hidden}")
        print(f"  Action Encoder: {action_hidden}")
        print(f"  Decoder: {decoder_hidden}")
        
        assert m_former_hidden == action_hidden == decoder_hidden
        print("✅ All dimensions consistent")
        
        print("\n🎉 DinoV2 configuration test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dinov2_config()
    if success:
        print("\n🚀 Ready to use DinoV2 EmbodimentAware configuration!")
    else:
        print("\n⚠️  Configuration needs fixing before use.")
