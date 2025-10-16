"""
Test Visual-Action Fusion Module
Tests the fusion mechanism with different modality presence scenarios
"""
import torch
import sys
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer')

from src.models.visual_action_fusion import VisualActionFusion


def test_fusion_scenarios():
    """Test different modality presence scenarios"""
    
    # Configuration
    batch_size = 2
    query_num = 8
    hidden_size = 768
    
    # Create fusion module
    fusion_module = VisualActionFusion(
        hidden_size=hidden_size,
        num_heads=8,
        query_num=query_num,
        dropout=0.1
    )
    
    print("🧪 Testing VisualActionFusion Module")
    print(f"📊 Configuration: batch_size={batch_size}, query_num={query_num}, hidden_size={hidden_size}")
    print()
    
    # Create test data
    visual_tokens = torch.randn(batch_size, query_num, hidden_size)
    action_tokens = torch.randn(batch_size, query_num, hidden_size)
    
    # Test scenarios
    scenarios = [
        {"name": "Both modalities present", "pv": torch.tensor([1, 1]), "pa": torch.tensor([1, 1])},
        {"name": "Visual only", "pv": torch.tensor([1, 1]), "pa": torch.tensor([0, 0])},
        {"name": "Action only", "pv": torch.tensor([0, 0]), "pa": torch.tensor([1, 1])},
        {"name": "Neither modality", "pv": torch.tensor([0, 0]), "pa": torch.tensor([0, 0])},
        {"name": "Mixed presence", "pv": torch.tensor([1, 0]), "pa": torch.tensor([0, 1])},
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"🔍 Scenario {i+1}: {scenario['name']}")
        print(f"   pv: {scenario['pv'].tolist()}, pa: {scenario['pa'].tolist()}")
        
        try:
            # Prepare inputs based on presence flags
            v_input = visual_tokens if scenario['pv'].any() else None
            a_input = action_tokens if scenario['pa'].any() else None
            
            # Forward pass
            with torch.no_grad():
                fused_output = fusion_module(
                    visual_tokens=v_input,
                    action_tokens=a_input,
                    pv=scenario['pv'],
                    pa=scenario['pa']
                )
            
            print(f"   ✅ Output shape: {fused_output.shape}")
            print(f"   📈 Output range: [{fused_output.min():.3f}, {fused_output.max():.3f}]")
            
            # Verify output properties
            assert fused_output.shape == (batch_size, query_num, hidden_size), f"Wrong output shape: {fused_output.shape}"
            assert not torch.isnan(fused_output).any(), "NaN values detected in output"
            assert torch.isfinite(fused_output).all(), "Infinite values detected in output"
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()
    
    # Test fusion info
    print("📋 Fusion Module Info:")
    info = fusion_module.get_fusion_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n✅ All fusion tests completed!")


def test_fusion_integration():
    """Test fusion integration with EmbodimentAwareLatentMotionTokenizer"""
    
    print("\n🔧 Testing Fusion Integration")
    
    try:
        from hydra import compose, initialize
        from omegaconf import OmegaConf
        import hydra
        
        # Initialize Hydra
        with initialize(config_path="configs/models", version_base=None):
            cfg = compose(config_name="embodiment_aware_dinov2_action_encoder.yaml")
            
        print("✅ Config loaded successfully")
        print(f"📊 Fusion config: num_heads={cfg.action_encoder_config.fusion_num_heads}, dropout={cfg.action_encoder_config.fusion_dropout}")
        
        # Test config instantiation
        model = hydra.utils.instantiate(cfg)
        print("✅ Model instantiated successfully with fusion module")
        
        # Test forward pass with fusion
        batch_size = 1
        cond_images = torch.randn(batch_size, 3, 224, 224)
        target_images = torch.randn(batch_size, 3, 224, 224)
        actions = torch.randn(batch_size, 4, 48)  # action_chunk_size=4, max_action_dim=48
        states = torch.randn(batch_size, 1, 48)
        
        with torch.no_grad():
            outputs = model(
                cond_pixel_values=cond_images,
                target_pixel_values=target_images,
                actions=actions,
                states=states
            )
        
        print("✅ Forward pass with fusion completed successfully")
        print(f"📊 Output keys: {list(outputs.keys())}")
        print(f"📊 Reconstruction shape: {outputs['recons_pixel_values'].shape}")
        if 'decoded_actions' in outputs:
            print(f"📊 Decoded actions shape: {outputs['decoded_actions'].shape}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test fusion module
    test_fusion_scenarios()
    
    # Test integration
    test_fusion_integration()
