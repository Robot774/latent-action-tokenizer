"""
Embodiment-aware Action Encoder and Decoder for Latent Motion Tokenizer
Based on Isaac-GR00T's CategorySpecific design
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CategorySpecificLinear(nn.Module):
    """Category-specific linear layer that maintains separate weights for each embodiment"""
    
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        """
        Args:
            x: (B, T, input_dim) - input features
            cat_ids: (B,) - category/embodiment IDs
        Returns:
            output: (B, T, hidden_dim)
        """
        # Ensure dtype alignment (avoid Double vs Float mismatches)
        if x.dtype != self.W.dtype:
            x = x.to(self.W.dtype)
        selected_W = self.W[cat_ids]  # (B, input_dim, hidden_dim)
        selected_b = self.b[cat_ids]  # (B, hidden_dim)
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    """Category-specific MLP with two layers"""
    
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        """
        Args:
            x: (B, T, input_dim) - input features
            cat_ids: (B,) - category/embodiment IDs
        Returns:
            output: (B, T, output_dim)
        """
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class EmbodimentAwareActionEncoder(nn.Module):
    """
    Embodiment-aware action encoder that can handle different action/state dimensions
    for different embodiments using category-specific transformations
    """
    
    def __init__(self, 
                 max_action_dim=48,
                 hidden_size=1024, 
                 num_embodiments=1,
                 embodiment_configs=None):
        super().__init__()
        
        self.max_action_dim = max_action_dim
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments
        
        # Embodiment配置映射
        self.embodiment_configs = embodiment_configs or {
            0: {"name": "default", "action_dim": max_action_dim, "state_dim": max_action_dim}
        }
        
        # Category-specific encoders following GR00T design
        self.action_encoder = CategorySpecificMLP(
            num_categories=num_embodiments,
            input_dim=max_action_dim,
            hidden_dim=hidden_size,
            output_dim=hidden_size
        )
        
        self.state_encoder = CategorySpecificMLP(
            num_categories=num_embodiments,
            input_dim=max_action_dim,
            hidden_dim=hidden_size, 
            output_dim=hidden_size
        )
        
    def encode_action(self, actions, embodiment_ids=None):
        """
        Encode action sequences
        Args:
            actions: (B, T, max_action_dim) - action sequences (already padded to unified dimension)
            embodiment_ids: (B,) - embodiment IDs for each sample
        Returns:
            action_features: (B, T, hidden_size)
        """
        # Ensure float dtype for computation
        if actions.dtype != self.action_encoder.layer1.W.dtype:
            actions = actions.to(self.action_encoder.layer1.W.dtype)
        if embodiment_ids is None:
            # Default to embodiment 0 if not provided
            embodiment_ids = torch.zeros(actions.shape[0], dtype=torch.long, device=actions.device)
            
        return self.action_encoder(actions, embodiment_ids)
    
    def encode_state(self, states, embodiment_ids=None):
        """
        Encode state sequences
        Args:
            states: (B, T, max_action_dim) - state sequences (already padded to unified dimension)  
            embodiment_ids: (B,) - embodiment IDs for each sample
        Returns:
            state_features: (B, T, hidden_size)
        """
        # Ensure float dtype for computation
        if states.dtype != self.state_encoder.layer1.W.dtype:
            states = states.to(self.state_encoder.layer1.W.dtype)
        if embodiment_ids is None:
            # Default to embodiment 0 if not provided
            embodiment_ids = torch.zeros(states.shape[0], dtype=torch.long, device=states.device)
            
        return self.state_encoder(states, embodiment_ids)
    
    def get_embodiment_info(self, embodiment_id):
        """Get embodiment configuration info"""
        return self.embodiment_configs.get(embodiment_id, self.embodiment_configs[0])


class EmbodimentAwareActionDecoder(nn.Module):
    """
    Embodiment-aware action decoder that decodes latent motion tokens back to action sequences
    Supports different action dimensions for different embodiments
    """
    
    def __init__(self,
                 hidden_size=768,
                 max_action_dim=48,
                 action_chunk_size=4,
                 num_embodiments=1,
                 embodiment_configs=None,
                 query_num=8):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.max_action_dim = max_action_dim
        self.action_chunk_size = action_chunk_size
        self.num_embodiments = num_embodiments
        self.query_num = query_num
        
        # Embodiment配置映射
        self.embodiment_configs = embodiment_configs or {
            0: {"name": "default", "action_dim": max_action_dim}
        }
        
        # Motion tokens pooling layer (query_num -> 1)
        self.motion_pooling = nn.Sequential(
            nn.Linear(hidden_size * query_num, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Category-specific action decoder
        # Input: pooled motion features (B, hidden_size)
        # Output: flattened actions (B, chunk_size * max_action_dim)
        output_size = action_chunk_size * max_action_dim
        
        self.action_decoder = CategorySpecificMLP(
            num_categories=num_embodiments,
            input_dim=hidden_size,
            hidden_dim=hidden_size,
            output_dim=output_size
        )
        
    def forward(self, latent_motion_tokens, embodiment_ids=None):
        """
        Decode latent motion tokens to action sequences
        
        Args:
            latent_motion_tokens: (B, query_num, hidden_size) - VQ后的motion tokens
            embodiment_ids: (B,) - embodiment IDs for each sample
            
        Returns:
            decoded_actions: (B, chunk_size, max_action_dim) - 解码的动作序列
        """
        if embodiment_ids is None:
            # Default to embodiment 0 if not provided
            embodiment_ids = torch.zeros(latent_motion_tokens.shape[0], dtype=torch.long, 
                                       device=latent_motion_tokens.device)
        
        B, query_num, hidden_size = latent_motion_tokens.shape
        
        # 1. Pool motion tokens: (B, query_num, hidden_size) -> (B, hidden_size)
        flattened_tokens = latent_motion_tokens.reshape(B, -1)  # (B, query_num * hidden_size)
        pooled_features = self.motion_pooling(flattened_tokens)  # (B, hidden_size)
        
        # 2. Embodiment-specific decoding: (B, hidden_size) -> (B, chunk_size * max_action_dim)
        flattened_actions = self.action_decoder(pooled_features.unsqueeze(1), embodiment_ids)  # (B, 1, output_size)
        flattened_actions = flattened_actions.squeeze(1)  # (B, chunk_size * max_action_dim)
        
        # 3. Reshape to action sequences: (B, chunk_size * max_action_dim) -> (B, chunk_size, max_action_dim)
        decoded_actions = flattened_actions.reshape(B, self.action_chunk_size, self.max_action_dim)
        
        return decoded_actions
    
    def get_embodiment_info(self, embodiment_id):
        """Get embodiment configuration info"""
        return self.embodiment_configs.get(embodiment_id, self.embodiment_configs[0])
