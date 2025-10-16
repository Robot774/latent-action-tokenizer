"""
A-Former: Action Transformer for processing action and state sequences
Based on M-Former design but specialized for action tokenization
"""
from transformers.models.vit.modeling_vit import (
    ViTConfig,
    ViTPreTrainedModel,
    ViTEncoder
)
from torch import nn
import torch
from typing import Optional, Dict, List, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPooling


class AFormerEmbeddings(nn.Module):
    """
    A-Former embeddings for processing action and state sequences
    Similar to MFormerEmbeddings but specialized for action data
    """
    
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        query_num = config.query_num
        self.query_num = query_num
        self.action_chunk_size = getattr(config, 'action_chunk_size', 4)
        
        # Learnable action query tokens (similar to latent_motion_token)
        self.latent_action_token = nn.Parameter(torch.zeros(1, query_num, config.hidden_size))
        
        # Separator token between state and action features
        self.sep_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        
        # Position embeddings for [queries, state, sep, actions]
        # Total length: query_num + 1 (state) + 1 (sep) + action_chunk_size
        max_seq_len = query_num + 1 + 1 + self.action_chunk_size
        self.position_embeddings = nn.Parameter(torch.randn(1, max_seq_len, config.hidden_size))
        
        # Token type embeddings (state vs action)
        self.token_type_embeddings = nn.Parameter(torch.randn(2, config.hidden_size))
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
        
        # Legacy flag for compatibility
        if hasattr(config, "legacy"):
            self.legacy = config.legacy
        else:
            self.legacy = True

    def forward(
        self,
        state_features: torch.Tensor,      # (B, 1, hidden_size)
        action_features: torch.Tensor,     # (B, chunk_size, hidden_size)
    ) -> torch.Tensor:
        batch_size = state_features.shape[0]
        action_seq_length = action_features.shape[1]
        
        # 1. Prepare learnable query tokens
        latent_action_tokens = self.latent_action_token.expand(batch_size, -1, -1)
        
        # 2. Prepare separator token
        sep_tokens = self.sep_token.expand(batch_size, -1, -1)
        
        # 3. Combine sequence: [queries, state, sep, actions]
        embeddings = torch.cat([
            latent_action_tokens,  # (B, query_num, hidden_size)
            state_features,        # (B, 1, hidden_size)
            sep_tokens,           # (B, 1, hidden_size)
            action_features       # (B, chunk_size, hidden_size)
        ], dim=1)
        
        # 4. Add positional encodings
        seq_length = embeddings.shape[1]
        embeddings = embeddings + self.position_embeddings[:, :seq_length, :]
        
        # 5. Add token type encodings
        # Queries and state use type 0, actions use type 1
        query_state_sep_length = self.query_num + 1 + 1  # queries + state + sep
        
        query_state_type_embeddings = self.token_type_embeddings[0].expand(
            batch_size, query_state_sep_length, -1
        )
        action_type_embeddings = self.token_type_embeddings[1].expand(
            batch_size, action_seq_length, -1
        )
        
        token_type_embeddings = torch.cat([
            query_state_type_embeddings,  # For queries, state, and sep
            action_type_embeddings        # For actions
        ], dim=1)
        
        embeddings = embeddings + token_type_embeddings
        
        # 6. Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings


class AFormer(ViTPreTrainedModel):
    """
    A-Former: Action Transformer for processing action sequences
    Generates latent action tokens from state and action features
    """
    
    def __init__(self,
                 config: ViTConfig,
                 add_pooling_layer: bool = False):  # Default False for simplicity
        
        super().__init__(config)
        self.config = config
        self.query_num = config.query_num
        
        # Components
        self.embeddings = AFormerEmbeddings(config)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Optional pooler (disabled by default for simplicity)
        self.pooler = None
        if add_pooling_layer:
            from .m_former import ViTPooler  # Reuse ViTPooler from m_former
            self.pooler = ViTPooler(config)
        
        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights (similar to MFormer)"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, AFormerEmbeddings):
            # Initialize learnable parameters
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.token_type_embeddings.data = nn.init.trunc_normal_(
                module.token_type_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.token_type_embeddings.dtype)

            module.latent_action_token.data = nn.init.trunc_normal_(
                module.latent_action_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.latent_action_token.dtype)

            module.sep_token.data = nn.init.trunc_normal_(
                module.sep_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.sep_token.dtype)

    def forward(
        self,
        state_features: torch.Tensor,      # (B, 1, hidden_size)
        action_features: torch.Tensor,     # (B, chunk_size, hidden_size)
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """
        Forward pass of A-Former
        
        Args:
            state_features: (B, 1, hidden_size) - Current state features
            action_features: (B, chunk_size, hidden_size) - Action sequence features
            
        Returns:
            latent_action_tokens: (B, query_num, hidden_size) - Latent action representations
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else getattr(self.config, 'use_return_dict', True)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 1. Embedding layer
        embedding_output = self.embeddings(
            state_features=state_features,
            action_features=action_features
        )

        # 2. Transformer encoder
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # 3. Layer normalization
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        
        # 4. Optional pooling
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 5. Extract latent action tokens (first query_num tokens)
        latent_action_tokens = sequence_output[:, :self.query_num]  # (B, query_num, hidden_size)

        if not return_dict:
            head_outputs = (latent_action_tokens, pooled_output) if pooled_output is not None else (latent_action_tokens,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=latent_action_tokens,  # Return only the latent action tokens
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
