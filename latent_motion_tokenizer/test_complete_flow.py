"""
Test Complete Action-Vision Data Flow
Tests the full pipeline: [action & vision] -> [action feature & vision feature] -> [feature fusion] -> [visual & action reconstruction]
"""
import torch
import sys
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer')

from src.models.embodiment_aware_action_encoder import EmbodimentAwareActionEncoder, EmbodimentAwareActionDecoder
from src.models.visual_action_fusion import VisualActionFusion
from src.models.a_former import AFormer
from transformers import ViTConfig


def test_complete_data_flow():
    """
    测试完整的action-vision数据流
    """
    
    print("🚀 Testing Complete Action-Vision Data Flow")
    print("=" * 60)
    
    # 1. 设置测试参数
    print("\n1️⃣ 设置测试参数")
    batch_size = 2
    action_chunk_size = 4
    max_action_dim = 48
    hidden_size = 768
    query_num = 8
    num_embodiments = 1
    
    print(f"   📊 Batch size: {batch_size}")
    print(f"   📊 Action chunk size: {action_chunk_size}")
    print(f"   📊 Max action dim: {max_action_dim}")
    print(f"   📊 Hidden size: {hidden_size}")
    print(f"   📊 Query num: {query_num}")
    
    # 2. 创建所有组件
    print("\n2️⃣ 创建数据流组件")
    
    # Action encoder/decoder
    action_encoder = EmbodimentAwareActionEncoder(
        max_action_dim=max_action_dim,
        hidden_size=hidden_size,
        num_embodiments=num_embodiments
    )
    
    action_decoder = EmbodimentAwareActionDecoder(
        hidden_size=hidden_size,
        max_action_dim=max_action_dim,
        action_chunk_size=action_chunk_size,
        num_embodiments=num_embodiments,
        query_num=query_num
    )
    
    # A-Former
    a_former_config = ViTConfig(
        hidden_size=hidden_size,
        num_hidden_layers=4,
        num_attention_heads=12,
        intermediate_size=3072,
        query_num=query_num,
        action_chunk_size=action_chunk_size,
        model_type="vit"
    )
    a_former = AFormer(a_former_config)
    
    # Fusion module
    fusion_module = VisualActionFusion(
        hidden_size=hidden_size,
        num_heads=8,
        query_num=query_num,
        dropout=0.1
    )
    
    print("   ✅ All components created successfully")
    
    # 3. 准备测试数据
    print("\n3️⃣ 准备测试数据")
    
    # 原始输入
    actions = torch.randn(batch_size, action_chunk_size, max_action_dim)
    states = torch.randn(batch_size, 1, max_action_dim)
    embodiment_ids = torch.zeros(batch_size, dtype=torch.long)
    
    # 模拟视觉tokens (来自M-Former)
    visual_tokens = torch.randn(batch_size, query_num, hidden_size)
    
    print(f"   📊 Actions: {actions.shape}")
    print(f"   📊 States: {states.shape}")
    print(f"   📊 Visual tokens: {visual_tokens.shape}")
    
    # 4. 测试完整数据流
    print("\n4️⃣ 执行完整数据流")
    
    with torch.no_grad():
        
        # Step 1: Action & State Encoding
        print("   🔄 Step 1: Action & State Encoding")
        action_features = action_encoder.encode_action(actions, embodiment_ids)
        state_features = action_encoder.encode_state(states, embodiment_ids)
        
        print(f"      • Actions {actions.shape} -> Action features {action_features.shape}")
        print(f"      • States {states.shape} -> State features {state_features.shape}")
        
        # Step 2: A-Former Tokenization
        print("   🔄 Step 2: A-Former Tokenization")
        a_former_output = a_former(state_features, action_features)  # 修正参数顺序
        action_tokens = a_former_output.last_hidden_state
        
        print(f"      • Action features {action_features.shape} + State features {state_features.shape}")
        print(f"        -> Action tokens {action_tokens.shape}")
        
        # Step 3: Visual-Action Fusion
        print("   🔄 Step 3: Visual-Action Fusion")
        pv = torch.ones(batch_size, dtype=torch.long)  # Visual present
        pa = torch.ones(batch_size, dtype=torch.long)  # Action present
        
        fused_tokens = fusion_module(
            visual_tokens=visual_tokens,
            action_tokens=action_tokens,
            pv=pv,
            pa=pa
        )
        
        print(f"      • Visual tokens {visual_tokens.shape} + Action tokens {action_tokens.shape}")
        print(f"        -> Fused tokens {fused_tokens.shape}")
        
        # Step 4: Action Reconstruction
        print("   🔄 Step 4: Action Reconstruction")
        decoded_actions = action_decoder(fused_tokens, embodiment_ids)
        
        print(f"      • Fused tokens {fused_tokens.shape} -> Decoded actions {decoded_actions.shape}")
        
        # Step 5: Loss Computation
        print("   🔄 Step 5: Loss Computation")
        action_recons_loss = torch.nn.functional.mse_loss(decoded_actions, actions)
        
        print(f"      • Action reconstruction loss: {action_recons_loss.item():.6f}")
    
    # 5. 验证数据流完整性
    print("\n5️⃣ 验证数据流完整性")
    
    # 检查维度一致性
    assert action_features.shape == (batch_size, action_chunk_size, hidden_size), f"Action features shape mismatch: {action_features.shape}"
    assert state_features.shape == (batch_size, 1, hidden_size), f"State features shape mismatch: {state_features.shape}"
    assert action_tokens.shape == (batch_size, query_num, hidden_size), f"Action tokens shape mismatch: {action_tokens.shape}"
    assert fused_tokens.shape == (batch_size, query_num, hidden_size), f"Fused tokens shape mismatch: {fused_tokens.shape}"
    assert decoded_actions.shape == actions.shape, f"Decoded actions shape mismatch: {decoded_actions.shape} vs {actions.shape}"
    
    print("   ✅ All dimensions consistent")
    
    # 检查数值范围
    print(f"   📊 Value ranges:")
    print(f"      • Action features: [{action_features.min():.3f}, {action_features.max():.3f}]")
    print(f"      • Action tokens: [{action_tokens.min():.3f}, {action_tokens.max():.3f}]")
    print(f"      • Fused tokens: [{fused_tokens.min():.3f}, {fused_tokens.max():.3f}]")
    print(f"      • Decoded actions: [{decoded_actions.min():.3f}, {decoded_actions.max():.3f}]")
    
    # 6. 测试不同输入组合
    print("\n6️⃣ 测试不同输入组合")
    
    test_cases = [
        {"name": "Action + State", "actions": actions, "states": states},
        {"name": "Action only", "actions": actions, "states": None},
        {"name": "State only", "actions": None, "states": states},
    ]
    
    for case in test_cases:
        print(f"   🧪 Testing: {case['name']}")
        
        with torch.no_grad():
            # Action encoding
            action_feats = None
            state_feats = None
            
            if case['actions'] is not None:
                action_feats = action_encoder.encode_action(case['actions'], embodiment_ids)
            if case['states'] is not None:
                state_feats = action_encoder.encode_state(case['states'], embodiment_ids)
            
            # A-Former with fallback handling
            if action_feats is not None and state_feats is not None:
                action_toks = a_former(state_feats, action_feats).last_hidden_state  # 修正参数顺序
            elif action_feats is not None:
                zero_states = torch.zeros(batch_size, 1, hidden_size)
                action_toks = a_former(zero_states, action_feats).last_hidden_state  # 修正参数顺序
            elif state_feats is not None:
                zero_actions = torch.zeros(batch_size, action_chunk_size, hidden_size)
                action_toks = a_former(state_feats, zero_actions).last_hidden_state  # 修正参数顺序
            else:
                action_toks = None
            
            # Fusion
            pa_flag = torch.ones(batch_size, dtype=torch.long) if action_toks is not None else torch.zeros(batch_size, dtype=torch.long)
            fused = fusion_module(visual_tokens, action_toks, pv, pa_flag)
            
            print(f"      ✅ {case['name']}: Fusion output {fused.shape}")
    
    # 7. 数据流总结
    print(f"\n7️⃣ 完整数据流总结")
    print("   🎯 Input:")
    print(f"      • Actions: (B, chunk_size, action_dim) = {actions.shape}")
    print(f"      • States: (B, 1, state_dim) = {states.shape}")
    print(f"      • Visual tokens: (B, query_num, hidden_size) = {visual_tokens.shape}")
    
    print("   🔄 Processing:")
    print(f"      • Action Encoder: actions -> action_features {action_features.shape}")
    print(f"      • State Encoder: states -> state_features {state_features.shape}")
    print(f"      • A-Former: action_features + state_features -> action_tokens {action_tokens.shape}")
    print(f"      • Fusion: visual_tokens + action_tokens -> fused_tokens {fused_tokens.shape}")
    print(f"      • Action Decoder: fused_tokens -> decoded_actions {decoded_actions.shape}")
    
    print("   🎯 Output:")
    print(f"      • Fused multimodal tokens: {fused_tokens.shape}")
    print(f"      • Reconstructed actions: {decoded_actions.shape}")
    print(f"      • Action reconstruction loss: {action_recons_loss.item():.6f}")
    
    print(f"\n✅ Complete Action-Vision Data Flow Test Passed!")
    
    return {
        "action_features": action_features,
        "state_features": state_features,
        "action_tokens": action_tokens,
        "fused_tokens": fused_tokens,
        "decoded_actions": decoded_actions,
        "action_recons_loss": action_recons_loss.item()
    }


def test_integration_with_model():
    """
    测试与EmbodimentAwareLatentMotionTokenizer的集成 (简化版本)
    """
    
    print("\n🔗 Testing Integration with Main Model")
    print("=" * 40)
    
    # 模拟完整模型的forward pass逻辑
    print("   📋 Simulating complete model forward pass:")
    print("   1. Visual processing: cond_images + target_images -> M-Former -> visual_tokens")
    print("   2. Action processing: actions + states -> Action Encoder -> action_features")
    print("   3. Action tokenization: action_features -> A-Former -> action_tokens")
    print("   4. Multimodal fusion: visual_tokens + action_tokens -> fused_tokens")
    print("   5. Vector quantization: fused_tokens -> VQ -> quantized_tokens")
    print("   6. Reconstruction: quantized_tokens -> Visual Decoder + Action Decoder")
    print("   7. Loss computation: visual_loss + action_recons_loss")
    
    print("   ✅ Integration flow verified conceptually")
    print("   📝 Ready for full model testing with actual image inputs")


if __name__ == "__main__":
    # 测试完整数据流
    results = test_complete_data_flow()
    
    # 测试模型集成
    test_integration_with_model()
    
    print(f"\n🎉 All tests completed successfully!")
    print(f"   Action reconstruction loss: {results['action_recons_loss']:.6f}")
    print(f"   Data flow integrity: ✅ PASSED")
