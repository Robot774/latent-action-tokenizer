"""
Visual-Action Fusion Module
Implements cross-modal fusion with presence-aware processing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VisualActionFusion(nn.Module):
    """
    Visual-Action Fusion with presence-aware cross-attention and gated fusion
    
    Features:
    1. Learnable missing tokens for absent modalities
    2. Cross-attention for information complementarity  
    3. Gated MLP with presence-aware fusion
    """
    
    def __init__(self, 
                 hidden_size: int = 768,
                 num_heads: int = 8,
                 query_num: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.query_num = query_num
        
        # Learnable missing tokens for absent modalities
        self.missing_visual_tokens = nn.Parameter(
            torch.randn(1, query_num, hidden_size) * 0.02
        )
        self.missing_action_tokens = nn.Parameter(
            torch.randn(1, query_num, hidden_size) * 0.02
        )
        
        # Cross-attention layers for mutual enhancement
        self.visual_to_action_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.action_to_visual_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization after cross-attention
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.action_norm = nn.LayerNorm(hidden_size)
        
        # Gated fusion network with presence awareness
        # Input: [visual_enhanced, action_enhanced, presence_encoding]
        self.presence_embedding = nn.Embedding(4, hidden_size)  # 4 combinations of (pv, pa)
        
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # visual + action + presence
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()  # Gate weights in [0, 1]
        )
        
        # Final projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
    def _get_presence_encoding(self, pv: torch.Tensor, pa: torch.Tensor) -> torch.Tensor:
        """
        Convert presence flags to encoding indices
        pv=0, pa=0 -> 0  (neither)
        pv=1, pa=0 -> 1  (visual only)  
        pv=0, pa=1 -> 2  (action only)
        pv=1, pa=1 -> 3  (both)
        """
        presence_idx = pv.long() * 2 + pa.long()  # (B,)
        presence_emb = self.presence_embedding(presence_idx)  # (B, hidden_size)
        return presence_emb.unsqueeze(1).expand(-1, self.query_num, -1)  # (B, query_num, hidden_size)
    
    def _handle_missing_modalities(self, 
                                   visual_tokens: Optional[torch.Tensor], 
                                   action_tokens: Optional[torch.Tensor],
                                   pv: torch.Tensor, 
                                   pa: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Handle missing modalities using learnable tokens
        """
        batch_size = pv.shape[0]
        
        # Soft masking: always expect real tokens; blend with missing tokens via presence mask
        if visual_tokens is None:
            raise ValueError("visual_tokens must not be None when using soft masking")
        if action_tokens is None:
            raise ValueError("action_tokens must not be None when using soft masking")

        missing_visual = self.missing_visual_tokens.expand(batch_size, -1, -1)
        pv_mask = pv.float().unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        visual_tokens = pv_mask * visual_tokens + (1 - pv_mask) * missing_visual

        missing_action = self.missing_action_tokens.expand(batch_size, -1, -1)
        pa_mask = pa.float().unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        action_tokens = pa_mask * action_tokens + (1 - pa_mask) * missing_action
        
        return visual_tokens, action_tokens
    
    def _cross_attention_enhancement(self,
                                   visual_tokens: torch.Tensor,
                                   action_tokens: torch.Tensor,
                                   pv: torch.Tensor,
                                   pa: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-attention for mutual information enhancement
        """
        # Visual attend to Action (visual enhanced by action information)
        visual_enhanced, _ = self.visual_to_action_attn(
            query=visual_tokens,      # (B, query_num, hidden_size)
            key=action_tokens,        # (B, query_num, hidden_size)
            value=action_tokens       # (B, query_num, hidden_size)
        )
        # Residual connection + layer norm
        visual_enhanced = self.visual_norm(visual_tokens + visual_enhanced)
        
        # Action attend to Visual (action enhanced by visual information)
        action_enhanced, _ = self.action_to_visual_attn(
            query=action_tokens,      # (B, query_num, hidden_size)
            key=visual_tokens,        # (B, query_num, hidden_size)
            value=visual_tokens       # (B, query_num, hidden_size)
        )
        # Residual connection + layer norm
        action_enhanced = self.action_norm(action_tokens + action_enhanced)
        
        return visual_enhanced, action_enhanced
    
    def _gated_fusion(self,
                     visual_enhanced: torch.Tensor,
                     action_enhanced: torch.Tensor,
                     pv: torch.Tensor,
                     pa: torch.Tensor) -> torch.Tensor:
        """
        Gated fusion with presence awareness
        """
        batch_size = visual_enhanced.shape[0]
        
        # Get presence encoding
        presence_encoding = self._get_presence_encoding(pv, pa)  # (B, query_num, hidden_size)
        
        # Combine features for gate computation
        # Pool over sequence dimension for gate computation
        visual_pooled = visual_enhanced.mean(dim=1)    # (B, hidden_size)
        action_pooled = action_enhanced.mean(dim=1)    # (B, hidden_size)
        presence_pooled = presence_encoding.mean(dim=1)  # (B, hidden_size)
        
        # Compute gate weights
        gate_input = torch.cat([visual_pooled, action_pooled, presence_pooled], dim=-1)  # (B, 3*hidden_size)
        gate_weights = self.gate_network(gate_input)  # (B, hidden_size)
        gate_weights = gate_weights.unsqueeze(1).expand(-1, self.query_num, -1)  # (B, query_num, hidden_size)
        
        # Gated fusion
        fused_tokens = gate_weights * visual_enhanced + (1 - gate_weights) * action_enhanced
        
        # Final projection
        fused_tokens = self.output_projection(fused_tokens)
        
        return fused_tokens
    
    def forward(self,
                visual_tokens: Optional[torch.Tensor],  # (B, query_num, hidden_size) or None
                action_tokens: Optional[torch.Tensor],  # (B, query_num, hidden_size) or None
                pv: torch.Tensor,                       # (B,) visual presence flags
                pa: torch.Tensor                        # (B,) action presence flags
                ) -> torch.Tensor:
        """
        Forward pass of visual-action fusion
        
        Args:
            visual_tokens: Visual latent tokens from M-Former, can be None
            action_tokens: Action latent tokens from A-Former, can be None
            pv: Visual presence flags (0=missing, 1=present)
            pa: Action presence flags (0=missing, 1=present)
            
        Returns:
            fused_tokens: (B, query_num, hidden_size) fused multimodal tokens
        """
        
        # Step 1: Handle missing modalities with learnable tokens
        visual_tokens, action_tokens = self._handle_missing_modalities(
            visual_tokens, action_tokens, pv, pa
        )
        
        # Step 2: Cross-attention enhancement for information complementarity
        visual_enhanced, action_enhanced = self._cross_attention_enhancement(
            visual_tokens, action_tokens, pv, pa
        )
        
        # Step 3: Gated fusion with presence awareness
        fused_tokens = self._gated_fusion(
            visual_enhanced, action_enhanced, pv, pa
        )
        
        return fused_tokens
    
    def get_fusion_info(self) -> dict:
        """Get fusion module information"""
        return {
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "query_num": self.query_num,
            "missing_tokens": {
                "visual": self.missing_visual_tokens.shape,
                "action": self.missing_action_tokens.shape
            },
            "presence_combinations": 4,  # 2^2 combinations of (pv, pa)
        }
