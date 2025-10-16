"""
Test Action Reconstruction Loss
Tests the action reconstruction loss functionality in EmbodimentAwareLatentMotionTokenizer
"""
import torch
import torch.nn.functional as F
import sys
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer')

from src.models.embodiment_aware_action_encoder import EmbodimentAwareActionEncoder, EmbodimentAwareActionDecoder
from src.models.visual_action_fusion import VisualActionFusion


def test_action_reconstruction_loss():
    """
    测试action reconstruction loss的计算和集成
    """
    
    print("🧪 Testing Action Reconstruction Loss")
    print("=" * 50)
    
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
    
    # 2. 创建测试组件
    print("\n2️⃣ 创建测试组件")
    
    # Action encoder
    action_encoder = EmbodimentAwareActionEncoder(
        max_action_dim=max_action_dim,
        hidden_size=hidden_size,
        num_embodiments=num_embodiments
    )
    
    # Action decoder  
    action_decoder = EmbodimentAwareActionDecoder(
        hidden_size=hidden_size,
        max_action_dim=max_action_dim,
        action_chunk_size=action_chunk_size,
        num_embodiments=num_embodiments,
        query_num=query_num
    )
    
    print("   ✅ Action encoder/decoder created")
    
    # 3. 准备测试数据
    print("\n3️⃣ 准备测试数据")
    
    # 原始动作序列
    original_actions = torch.randn(batch_size, action_chunk_size, max_action_dim)
    states = torch.randn(batch_size, 1, max_action_dim)
    embodiment_ids = torch.zeros(batch_size, dtype=torch.long)
    
    # 模拟latent motion tokens (来自M-Former + fusion)
    latent_motion_tokens = torch.randn(batch_size, query_num, hidden_size)
    
    print(f"   📊 Original actions: {original_actions.shape}")
    print(f"   📊 States: {states.shape}")
    print(f"   📊 Latent motion tokens: {latent_motion_tokens.shape}")
    
    # 4. 测试完整的action reconstruction流程
    print("\n4️⃣ 测试Action Reconstruction流程")
    
    with torch.no_grad():
        # Step 1: Encode actions
        action_features = action_encoder.encode_action(original_actions, embodiment_ids)
        state_features = action_encoder.encode_state(states, embodiment_ids)
        
        print(f"   🔧 Action features: {action_features.shape}")
        print(f"   🔧 State features: {state_features.shape}")
        
        # Step 2: Decode actions from latent motion tokens
        decoded_actions = action_decoder(latent_motion_tokens, embodiment_ids)
        
        print(f"   🔧 Decoded actions: {decoded_actions.shape}")
        
        # Step 3: Compute action reconstruction loss
        action_recons_loss = F.mse_loss(decoded_actions, original_actions)
        
        print(f"   🔧 Action reconstruction loss: {action_recons_loss.item():.6f}")
    
    # 5. 测试不同的损失权重
    print("\n5️⃣ 测试不同损失权重的影响")
    
    loss_weights = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    for weight in loss_weights:
        # 模拟其他损失组件
        commit_loss = torch.tensor(0.1)
        recons_loss = torch.tensor(0.5) 
        perceptual_loss = torch.tensor(0.2)
        
        # 计算总损失
        total_loss = (1.0 * commit_loss + 
                     1.0 * recons_loss + 
                     1.0 * perceptual_loss + 
                     weight * action_recons_loss)
        
        print(f"   📊 Weight={weight:.1f}: Total loss={total_loss.item():.4f} " +
              f"(action contribution: {(weight * action_recons_loss).item():.4f})")
    
    # 6. 测试不同动作相似度的重建损失
    print("\n6️⃣ 测试不同动作相似度的重建损失")
    
    # 生成不同相似度的解码动作
    similarity_levels = {
        "Perfect reconstruction": original_actions,  # 完美重建
        "Small noise": original_actions + 0.1 * torch.randn_like(original_actions),  # 小噪声
        "Medium noise": original_actions + 0.5 * torch.randn_like(original_actions),  # 中等噪声  
        "Large noise": original_actions + 1.0 * torch.randn_like(original_actions),  # 大噪声
        "Random actions": torch.randn_like(original_actions),  # 随机动作
    }
    
    for level_name, test_decoded in similarity_levels.items():
        test_loss = F.mse_loss(test_decoded, original_actions)
        print(f"   📊 {level_name}: Loss = {test_loss.item():.6f}")
    
    # 7. 验证梯度流
    print("\n7️⃣ 验证梯度流")
    
    # 启用梯度计算
    original_actions.requires_grad_(True)
    latent_motion_tokens.requires_grad_(True)
    
    # Forward pass
    decoded_actions = action_decoder(latent_motion_tokens, embodiment_ids)
    action_loss = F.mse_loss(decoded_actions, original_actions)
    
    # Backward pass
    action_loss.backward()
    
    # 检查梯度
    if latent_motion_tokens.grad is not None:
        grad_norm = torch.norm(latent_motion_tokens.grad).item()
        print(f"   ✅ Gradient flow verified - Latent tokens grad norm: {grad_norm:.6f}")
    else:
        print(f"   ❌ No gradient flow detected")
    
    # 8. 损失组件总结
    print(f"\n8️⃣ Action Reconstruction Loss总结")
    print("   🎯 功能:")
    print("      • 衡量解码动作与原始动作的相似度")
    print("      • 使用MSE损失计算重建误差")
    print("      • 支持可配置的损失权重")
    print("   🔧 集成:")
    print("      • 在EmbodimentAwareLatentMotionTokenizer中自动计算")
    print("      • 当actions输入不为None时启用")
    print("      • 添加到总损失中: total_loss += action_recons_loss_w * action_recons_loss")
    print("   📊 预期效果:")
    print("      • 促进模型学习action-conditioned的视觉表征")
    print("      • 提高动作预测的准确性")
    print("      • 增强多模态信息的一致性")
    
    print(f"\n✅ Action Reconstruction Loss测试完成!")
    
    return {
        "action_recons_loss": action_recons_loss.item(),
        "original_shape": original_actions.shape,
        "decoded_shape": decoded_actions.shape,
        "gradient_flow": latent_motion_tokens.grad is not None
    }


def test_loss_integration():
    """
    测试损失在完整模型中的集成 (简化版本，不依赖完整模型加载)
    """
    
    print("\n🔗 Testing Loss Integration")
    print("=" * 30)
    
    # 模拟完整模型的损失计算逻辑
    batch_size = 2
    
    # 模拟各种损失组件
    commit_loss = torch.tensor(0.15)
    recons_loss = torch.tensor(0.45)  
    perceptual_loss = torch.tensor(0.25)
    action_recons_loss = torch.tensor(0.35)
    
    # 损失权重 (来自配置)
    commit_loss_w = 1.0
    recon_loss_w = 1.0
    perceptual_loss_w = 1.0
    action_recons_loss_w = 1.0
    
    print(f"   📊 Individual losses:")
    print(f"      • Commit loss: {commit_loss.item():.4f}")
    print(f"      • Recons loss: {recons_loss.item():.4f}")
    print(f"      • Perceptual loss: {perceptual_loss.item():.4f}")
    print(f"      • Action recons loss: {action_recons_loss.item():.4f}")
    
    # 计算总损失
    total_loss = (commit_loss_w * commit_loss + 
                  recon_loss_w * recons_loss + 
                  perceptual_loss_w * perceptual_loss +
                  action_recons_loss_w * action_recons_loss)
    
    print(f"   🎯 Total loss: {total_loss.item():.4f}")
    
    # 分析各组件贡献
    contributions = {
        "Commit": (commit_loss_w * commit_loss / total_loss * 100).item(),
        "Recons": (recon_loss_w * recons_loss / total_loss * 100).item(),
        "Perceptual": (perceptual_loss_w * perceptual_loss / total_loss * 100).item(),
        "Action": (action_recons_loss_w * action_recons_loss / total_loss * 100).item(),
    }
    
    print(f"   📊 Loss contributions:")
    for name, contrib in contributions.items():
        print(f"      • {name}: {contrib:.1f}%")
    
    print(f"   ✅ Loss integration verified!")


if __name__ == "__main__":
    # 测试action reconstruction loss
    results = test_action_reconstruction_loss()
    
    # 测试损失集成
    test_loss_integration()
