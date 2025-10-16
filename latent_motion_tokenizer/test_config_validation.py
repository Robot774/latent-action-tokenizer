"""
Test script to validate EmbodimentAware config loading
Verifies that the new configuration files can properly initialize the model
"""
import sys
import os
import yaml
from pathlib import Path

# Add the src path
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer/src')

def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def validate_config_structure(config):
    """Validate the configuration structure"""
    print("🔍 Validating configuration structure...")
    
    # Check required top-level keys
    required_keys = ['_target_', 'image_encoder', 'm_former', 'vector_quantizer', 'decoder']
    for key in required_keys:
        assert key in config, f"Missing required key: {key}"
        print(f"  ✅ Found required key: {key}")
    
    # Check if action_encoder_config is present
    if 'action_encoder_config' in config:
        print("  ✅ Found action_encoder_config")
        
        action_config = config['action_encoder_config']
        required_action_keys = ['max_action_dim', 'hidden_size', 'num_embodiments']
        for key in required_action_keys:
            assert key in action_config, f"Missing required action config key: {key}"
            print(f"    ✅ Found action config key: {key}")
            
        # Validate embodiment configs
        if 'embodiment_configs' in action_config:
            print("    ✅ Found embodiment_configs")
            for emb_id, emb_config in action_config['embodiment_configs'].items():
                required_emb_keys = ['name', 'action_dim', 'state_dim']
                for key in required_emb_keys:
                    assert key in emb_config, f"Missing embodiment config key: {key}"
                print(f"      ✅ Embodiment {emb_id}: {emb_config['name']}")
    else:
        print("  ⚠️  No action_encoder_config found (original tokenizer mode)")

def validate_target_class(config):
    """Validate that the target class is correct"""
    print("\n🎯 Validating target class...")
    
    target = config['_target_']
    print(f"  Target class: {target}")
    
    if 'EmbodimentAware' in target:
        print("  ✅ Using EmbodimentAwareLatentMotionTokenizer")
        assert 'action_encoder_config' in config, "EmbodimentAware tokenizer requires action_encoder_config"
    else:
        print("  ℹ️  Using original LatentMotionTokenizer")

def validate_dimension_consistency(config):
    """Validate dimension consistency between components"""
    print("\n📏 Validating dimension consistency...")
    
    # Get m_former hidden size
    m_former_hidden_size = config['m_former']['config']['hidden_size']
    print(f"  M-Former hidden_size: {m_former_hidden_size}")
    
    # Get decoder hidden size
    decoder_hidden_size = config['decoder']['config']['hidden_size']
    print(f"  Decoder hidden_size: {decoder_hidden_size}")
    
    # Check consistency
    assert m_former_hidden_size == decoder_hidden_size, \
        f"Dimension mismatch: m_former ({m_former_hidden_size}) != decoder ({decoder_hidden_size})"
    print("  ✅ M-Former and Decoder dimensions consistent")
    
    # Check action encoder dimensions if present
    if 'action_encoder_config' in config:
        action_hidden_size = config['action_encoder_config']['hidden_size']
        print(f"  Action Encoder hidden_size: {action_hidden_size}")
        
        assert action_hidden_size == m_former_hidden_size, \
            f"Dimension mismatch: action_encoder ({action_hidden_size}) != m_former ({m_former_hidden_size})"
        print("  ✅ Action Encoder and M-Former dimensions consistent")

def test_config_file(config_path):
    """Test a single configuration file"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing configuration: {Path(config_path).name}")
    print(f"{'='*60}")
    
    try:
        # Load configuration
        config = load_config(config_path)
        print(f"✅ Successfully loaded config from {config_path}")
        
        # Validate structure
        validate_config_structure(config)
        
        # Validate target class
        validate_target_class(config)
        
        # Validate dimensions
        validate_dimension_consistency(config)
        
        # Print action encoder config details if present
        if 'action_encoder_config' in config:
            print(f"\n📋 Action Encoder Configuration Details:")
            action_config = config['action_encoder_config']
            for key, value in action_config.items():
                if key != 'embodiment_configs':
                    print(f"  {key}: {value}")
            
            if 'embodiment_configs' in action_config:
                print(f"  embodiment_configs:")
                for emb_id, emb_config in action_config['embodiment_configs'].items():
                    print(f"    {emb_id}: {emb_config}")
        
        print(f"\n🎉 Configuration validation passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Configuration validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting EmbodimentAware Configuration Validation...")
    
    # Configuration file paths
    config_dir = Path("/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer/configs/models")
    
    config_files = [
        config_dir / "embodiment_aware_dinov2_action_encoder.yaml",
    ]
    
    # Test each configuration
    results = []
    for config_path in config_files:
        if config_path.exists():
            success = test_config_file(config_path)
            results.append((config_path.name, success))
        else:
            print(f"⚠️  Configuration file not found: {config_path}")
            results.append((config_path.name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Summary:")
    print(f"{'='*60}")
    
    for config_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {config_name}: {status}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, success in results if success)
    
    print(f"\nOverall: {passed_tests}/{total_tests} configurations passed")
    
    if passed_tests == total_tests:
        print("🎉 All configuration tests passed!")
    else:
        print("⚠️  Some configuration tests failed")

if __name__ == "__main__":
    main()
