"""
Visual-Action Fusion Usage Example
Demonstrates the fusion mechanism for multimodal processing
"""
import torch
import sys
sys.path.append('/dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer')

from src.models.visual_action_fusion import VisualActionFusion


def fusion_usage_example():
    """
    完整的Visual-Action Fusion使用示例
    展示不同模态存在情况下的融合处理
    """
    
    print("🔀 Visual-Action Fusion Usage Example")
    print("=" * 50)
    
    # 1. 创建fusion模块
    print("\n1️⃣ 创建Fusion模块")
    fusion_module = VisualActionFusion(
        hidden_size=768,     # 与M-Former输出维度一致
        num_heads=8,         # 多头注意力头数
        query_num=8,         # 查询token数量
        dropout=0.1          # Dropout率
    )
    print(f"   ✅ Fusion模块创建成功")
    
    # 2. 准备测试数据
    print("\n2️⃣ 准备测试数据")
    batch_size = 2
    query_num = 8
    hidden_size = 768
    
    # 模拟M-Former输出的visual tokens
    visual_tokens = torch.randn(batch_size, query_num, hidden_size)
    print(f"   📊 Visual tokens shape: {visual_tokens.shape}")
    
    # 模拟A-Former输出的action tokens (暂时用随机数据)
    action_tokens = torch.randn(batch_size, query_num, hidden_size) 
    print(f"   📊 Action tokens shape: {action_tokens.shape}")
    
    # 3. 测试不同的模态存在场景
    print("\n3️⃣ 测试不同模态存在场景")
    
    scenarios = [
        {
            "name": "🎯 完整多模态 (Both Present)",
            "description": "视觉和动作信息都存在",
            "pv": torch.tensor([1, 1]),  # 两个样本都有视觉
            "pa": torch.tensor([1, 1]),  # 两个样本都有动作
            "visual_input": visual_tokens,
            "action_input": action_tokens
        },
        {
            "name": "👁️ 仅视觉模态 (Visual Only)",
            "description": "只有视觉信息，缺少动作信息",
            "pv": torch.tensor([1, 1]),  # 两个样本都有视觉
            "pa": torch.tensor([0, 0]),  # 两个样本都缺少动作
            "visual_input": visual_tokens,
            "action_input": None  # 动作缺失
        },
        {
            "name": "🤖 仅动作模态 (Action Only)",
            "description": "只有动作信息，缺少视觉信息",
            "pv": torch.tensor([0, 0]),  # 两个样本都缺少视觉
            "pa": torch.tensor([1, 1]),  # 两个样本都有动作
            "visual_input": None,  # 视觉缺失
            "action_input": action_tokens
        },
        {
            "name": "🔀 混合场景 (Mixed)",
            "description": "第一个样本有视觉，第二个样本有动作",
            "pv": torch.tensor([1, 0]),  # 第一个有视觉，第二个没有
            "pa": torch.tensor([0, 1]),  # 第一个没动作，第二个有
            "visual_input": visual_tokens,
            "action_input": action_tokens
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        print(f"\n   {scenario['name']}")
        print(f"   📝 {scenario['description']}")
        print(f"   🏷️ pv={scenario['pv'].tolist()}, pa={scenario['pa'].tolist()}")
        
        with torch.no_grad():
            fused_output = fusion_module(
                visual_tokens=scenario['visual_input'],
                action_tokens=scenario['action_input'], 
                pv=scenario['pv'],
                pa=scenario['pa']
            )
        
        print(f"   ✅ 输出形状: {fused_output.shape}")
        print(f"   📊 数值范围: [{fused_output.min():.3f}, {fused_output.max():.3f}]")
        
        # 分析不同样本的融合结果
        if batch_size == 2:
            sample1_norm = torch.norm(fused_output[0]).item()
            sample2_norm = torch.norm(fused_output[1]).item()
            print(f"   🔍 样本范数: Sample1={sample1_norm:.3f}, Sample2={sample2_norm:.3f}")
    
    # 4. 融合机制原理说明
    print(f"\n4️⃣ 融合机制原理")
    print("   🧠 Cross-Attention增强:")
    print("      • Visual → Action: 视觉信息增强动作表征")
    print("      • Action → Visual: 动作信息增强视觉表征")
    print("   🚪 门控融合:")
    print("      • 基于模态存在标志计算融合权重")
    print("      • 自适应平衡不同模态贡献")
    print("   🎭 缺失处理:")
    print("      • 可学习的缺失token替代缺失模态")
    print("      • 保证融合过程的连续性")
    
    # 5. 在实际模型中的使用
    print(f"\n5️⃣ 在EmbodimentAwareLatentMotionTokenizer中的集成")
    print("   📥 输入:")
    print("      • visual_tokens: M-Former输出 (B, 8, 768)")  
    print("      • action_tokens: A-Former输出 (B, 8, 768) [待实现]")
    print("      • pv, pa: 模态存在标志 (B,)")
    print("   📤 输出:")
    print("      • fused_tokens: 融合后的多模态表征 (B, 8, 768)")
    print("   🔄 后续处理:")
    print("      • fused_tokens → VQ → 量化 → Decoder")
    
    print(f"\n✅ Fusion使用示例完成!")
    return fusion_module


if __name__ == "__main__":
    fusion_usage_example()
