"""Latent Motion Tokenizer model."""
import torch
import torch.nn.functional as F
from torch import nn
import lpips
from einops import rearrange
from transformers import ViTMAEModel
from PIL import Image
from torchvision import transforms as T
import time
from collections import OrderedDict
from .embodiment_aware_action_encoder import EmbodimentAwareActionEncoder, EmbodimentAwareActionDecoder
from .visual_action_fusion import VisualActionFusion
from .a_former import AFormer


class LatentMotionTokenizer(nn.Module):
    def __init__(
            self,
            image_encoder,
            m_former,
            vector_quantizer,
            decoder,
            hidden_state_decoder=None,
            codebook_dim=32,
            commit_loss_w=1.,
            recon_loss_w=1.,
            recon_hidden_loss_w=1.,
            perceptual_loss_w=1.,
            action_recons_loss_w=1.,  # 🆕 Action reconstruction loss weight
            use_abs_recons_loss=False,
    ):
        super().__init__()

        codebook_embed_dim = codebook_dim
        decoder_hidden_size = decoder.config.hidden_size
        m_former_hidden_size = m_former.config.hidden_size

        # 检测编码器类型并适配
        if isinstance(image_encoder, ViTMAEModel):
            image_encoder.config.mask_ratio = 0.0
            self.encoder_type = "mae"
        elif hasattr(image_encoder, 'dino_featurizer') and hasattr(image_encoder, 'siglip_featurizer'):
            self.encoder_type = "dinosiglip"
        else:
            self.encoder_type = "dino"  # 默认为DINO
            # print(f"Warning: Assuming DINO encoder type for: {type(image_encoder)}")

        self.image_encoder = image_encoder.requires_grad_(False).eval()

        self.m_former = m_former

        self.vector_quantizer = vector_quantizer
        self.vq_down_resampler = nn.Sequential(
            nn.Linear(m_former_hidden_size, decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(decoder_hidden_size, codebook_embed_dim)
        )
        self.vq_up_resampler = nn.Sequential(
            nn.Linear(codebook_embed_dim, codebook_embed_dim),
            nn.Tanh(),
            nn.Linear(codebook_embed_dim, decoder_hidden_size)
        )

        self.decoder = decoder
        self.hidden_state_decoder = hidden_state_decoder

        self.commit_loss_w = commit_loss_w
        self.recon_loss_w = recon_loss_w
        self.recon_hidden_loss_w = recon_hidden_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        self.action_recons_loss_w = action_recons_loss_w  # 🆕 Action reconstruction loss weight
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()
        self.use_abs_recons_loss = use_abs_recons_loss

    @property
    def device(self):
        return next(self.parameters()).device


    def get_state_dict_to_save(self):
        modules_to_exclude = ['loss_fn_lpips', 'image_encoder']
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict


    @torch.no_grad()
    def decode_image(self, cond_pixel_values, given_motion_token_ids):
        quant = self.vector_quantizer.get_codebook_entry(given_motion_token_ids)
        latent_motion_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(cond_input=cond_pixel_values, latent_motion_tokens=latent_motion_tokens_up)
        return  {
            "recons_pixel_values": recons_pixel_values,
        }


    @torch.no_grad()
    def embed(self, cond_pixel_values, target_pixel_values, pool=False, before_vq=False, avg=False):
        quant, *_ = self.tokenize(cond_pixel_values, target_pixel_values, before_vq=before_vq)
        if pool:
            latent_motion_tokens_up = self.vq_up_resampler(quant)
            flat_latent_motion_tokens_up = latent_motion_tokens_up.reshape(latent_motion_tokens_up.shape[0], -1)
            pooled_embeddings = self.decoder.transformer.embeddings.query_pooling_layer(flat_latent_motion_tokens_up)
            return pooled_embeddings
        elif avg:
            return quant.mean(dim=1)
        else:
            return quant.reshape(quant.shape[0], -1)

    def tokenize(self, cond_pixel_values, target_pixel_values, before_vq=False):
        with torch.no_grad():
            cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state

        query_num = self.m_former.query_num
        latent_motion_tokens = self.m_former(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num]

        if before_vq:
            return latent_motion_tokens, None, None
        else:
            latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
            quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
            return quant, indices, commit_loss


    def forward(self, cond_pixel_values, target_pixel_values,
                return_recons_only=False, 
                return_motion_token_ids_only=False): 

        # Tokenization
        with torch.no_grad():
            cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state

        query_num = self.m_former.query_num
        latent_motion_tokens = self.m_former(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num]

        latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
        quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)
        
        # quant, indices, commit_loss = self.tokenize(cond_pixel_values, target_pixel_values)

        if return_motion_token_ids_only:
            return indices # (bs, motion_query_num)

        # Detokenization
        latent_motion_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(
            cond_input=cond_pixel_values,
            latent_motion_tokens=latent_motion_tokens_up
        )
            
        if return_recons_only:
            return {
                "recons_pixel_values": recons_pixel_values,
                "indices": indices
            }

        if self.hidden_state_decoder is not None:
            recons_hidden_states = self.hidden_state_decoder(
                cond_input = cond_hidden_states,
                latent_motion_tokens=latent_motion_tokens_up
            )

        # Compute loss
        outputs = {
            "loss": torch.zeros_like(commit_loss),
            "commit_loss": commit_loss,
            "recons_loss": torch.zeros_like(commit_loss),
            "recons_hidden_loss": torch.zeros_like(commit_loss),
            "perceptual_loss": torch.zeros_like(commit_loss)
        }

        if self.use_abs_recons_loss:
            recons_loss = torch.abs(recons_pixel_values - target_pixel_values).mean()
        else:
            recons_loss = F.mse_loss(target_pixel_values, recons_pixel_values)
        outputs["recons_loss"] = recons_loss

        if self.perceptual_loss_w > 0:
            with torch.no_grad():
                perceptual_loss = self.loss_fn_lpips.forward(
                    target_pixel_values, recons_pixel_values, normalize=True).mean()
        else:
            perceptual_loss = torch.zeros_like(recons_loss)
        outputs["perceptual_loss"] = perceptual_loss

        loss =  self.commit_loss_w * outputs["commit_loss"] + self.recon_loss_w * outputs["recons_loss"] + \
                self.perceptual_loss_w * outputs["perceptual_loss"]
        
        if self.hidden_state_decoder is not None:
            recon_hidden_loss = F.mse_loss(target_hidden_states, recons_hidden_states)
            outputs['recons_hidden_loss'] = recon_hidden_loss
            loss += self.recon_hidden_loss_w * outputs['recons_hidden_loss']

        outputs["loss"] = loss

        # active_code_num = torch.tensor(len(set(indices.long().reshape(-1).cpu().numpy().tolist()))).float().to(loss.device)
        active_code_num = torch.tensor(torch.unique(indices).shape[0]).float().to(loss.device)
        outputs["active_code_num"] = active_code_num

        return outputs


class EmbodimentAwareLatentMotionTokenizer(LatentMotionTokenizer):
    """
    Extended Latent Motion Tokenizer with embodiment-aware action conditioning
    Inherits from LatentMotionTokenizer and adds action/state encoding capabilities
    """
    
    def __init__(self,
                 image_encoder,
                 m_former,
                 vector_quantizer,
                 decoder,
                 action_encoder_config=None,
                 **kwargs):
        """
        Args:
            image_encoder: Visual encoder (frozen)
            m_former: Motion transformer
            vector_quantizer: Vector quantizer for motion tokens
            decoder: Motion decoder
            action_encoder_config: Configuration for action encoder
                {
                    'max_action_dim': 48,
                    'hidden_size': 1024, 
                    'num_embodiments': 1,
                    'embodiment_configs': {...}
                }
            **kwargs: Other arguments passed to parent class
        """
        super().__init__(image_encoder, m_former, vector_quantizer, decoder, **kwargs)
        
        # Action encoder configuration
        self.enable_action_conditioning = action_encoder_config is not None
        if self.enable_action_conditioning:
            self.action_encoder_config = action_encoder_config
            self._build_action_encoder()
        else:
            self.action_encoder = None
    
    def _build_action_encoder(self):
        """Build embodiment-aware action encoder and decoder"""
        config = self.action_encoder_config
        
        # Validate config
        required_keys = ['max_action_dim', 'hidden_size', 'num_embodiments']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Get m_former query_num for decoder
        query_num = self.m_former.query_num
        
        # Create action encoder
        self.action_encoder = EmbodimentAwareActionEncoder(
            max_action_dim=config['max_action_dim'],
            hidden_size=config['hidden_size'],
            num_embodiments=config['num_embodiments'],
            embodiment_configs=config.get('embodiment_configs', None)
        )
        
        # 🆕 Create action decoder
        self.action_decoder = EmbodimentAwareActionDecoder(
            hidden_size=config['hidden_size'],
            max_action_dim=config['max_action_dim'],
            action_chunk_size=config.get('action_chunk_size', 4),
            num_embodiments=config['num_embodiments'],
            embodiment_configs=config.get('embodiment_configs', None),
            query_num=query_num
        )
        
        # 🆕 Create visual-action fusion module
        self.fusion_module = VisualActionFusion(
            hidden_size=config['hidden_size'],
            num_heads=config.get('fusion_num_heads', 8),
            query_num=query_num,
            dropout=config.get('fusion_dropout', 0.1)
        )
        
        # 🆕 Create A-Former for action tokenization
        from transformers import ViTConfig
        a_former_config = ViTConfig(
            hidden_size=config['hidden_size'],
            num_hidden_layers=config.get('a_former_num_layers', 4),
            num_attention_heads=config.get('a_former_num_heads', 12),
            intermediate_size=config.get('a_former_intermediate_size', 3072),
            query_num=query_num,
            action_chunk_size=config.get('action_chunk_size', 4),
            model_type="vit"
        )
        self.a_former = AFormer(a_former_config)
        
        # print(f"✅ Built EmbodimentAwareActionEncoder with config: {config}")
        # print(f"✅ Built EmbodimentAwareActionDecoder with query_num: {query_num}")
        # print(f"✅ Built VisualActionFusion with hidden_size: {config['hidden_size']}")
        # print(f"✅ Built A-Former with query_num: {query_num}, layers: {a_former_config.num_hidden_layers}")
        
    
    def forward(self, 
                cond_pixel_values,
                target_pixel_values,
                actions=None,           # (B, action_chunk_size, max_action_dim)
                states=None,            # (B, 1, max_action_dim)
                embodiment_ids=None,    # (B,) - embodiment IDs
                gt_actions_for_loss=None,  # Ground truth actions for loss computation (when actions are dropped)
                gt_images_for_loss=None,   # Ground truth images for loss computation (when vision is dropped)
                return_recons_only=False,
                return_motion_token_ids_only=False,
                **kwargs):
        """
        Forward pass with optional dual-modal conditioning and drop support
        
        Args:
            cond_pixel_values: Condition images (B, C, H, W)
            target_pixel_values: Target images (B, C, H, W), can be None if dropped
            actions: Action sequences (B, chunk_size, max_action_dim), optional
            states: Current states (B, 1, max_action_dim), optional
            embodiment_ids: Embodiment IDs (B,), optional
            gt_actions_for_loss: Ground truth actions for loss computation (when actions are dropped)
            gt_images_for_loss: Ground truth images for loss computation (when vision is dropped)
            return_recons_only: Whether to return only reconstruction
            return_motion_token_ids_only: Whether to return only motion token IDs
            
        Returns:
            Dictionary containing reconstruction results and optional action features
        """
        
        # Use action conditioning if enabled AND (actions provided OR GT actions provided)
        use_action_conditioning = (self.enable_action_conditioning and 
                                   (actions is not None or gt_actions_for_loss is not None))
        
        # If no action conditioning, use parent class behavior
        if not use_action_conditioning:
            return super().forward(
                cond_pixel_values, 
                target_pixel_values,
                return_recons_only=return_recons_only,
                return_motion_token_ids_only=return_motion_token_ids_only,
                **kwargs
            )
        
        # 1. Visual tokenization (always compute; masking handled in fusion)
        query_num = self.m_former.query_num
        with torch.no_grad():
            cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state
        latent_motion_tokens = self.m_former(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states
        ).last_hidden_state[:, :query_num]

        # 2. 🆕 Action encoding and tokenization
        action_features = None
        state_features = None
        action_tokens = None  # A-Former output
        
        if actions is not None:
            action_features = self.action_encoder.encode_action(actions, embodiment_ids)
            # print(f"🔧 Encoded actions: {actions.shape} -> {action_features.shape}")
            
        if states is not None:
            state_features = self.action_encoder.encode_state(states, embodiment_ids)
            # print(f"🔧 Encoded states: {states.shape} -> {state_features.shape}")
            
        # 🆕 A-Former integration for action tokenization
        if action_features is not None and state_features is not None:
            action_tokens = self.a_former(state_features, action_features).last_hidden_state
            # print(f"🔧 A-Former tokenization: state_features {state_features.shape} + action_features {action_features.shape} -> action_tokens {action_tokens.shape}")
        elif action_features is not None:
            # If only actions available, use zero states
            batch_size = action_features.shape[0]
            zero_states = torch.zeros(batch_size, 1, action_features.shape[-1], device=action_features.device)
            action_tokens = self.a_former(zero_states, action_features).last_hidden_state
            # print(f"🔧 A-Former tokenization (action only): {action_features.shape} -> {action_tokens.shape}")
        elif state_features is not None:
            # If only states available, use zero actions
            batch_size = state_features.shape[0]
            action_chunk_size = self.action_encoder_config.get('action_chunk_size', 4)
            zero_actions = torch.zeros(batch_size, action_chunk_size, state_features.shape[-1], device=state_features.device)
            action_tokens = self.a_former(state_features, zero_actions).last_hidden_state
            # print(f"🔧 A-Former tokenization (state only): {state_features.shape} -> {action_tokens.shape}")
        
        # 3. 🆕 Visual-Action Fusion
        batch_size = cond_pixel_values.shape[0]
        device = cond_pixel_values.device
        
        # Presence flags: prefer external inputs (pv/pa), fallback to ones
        pv = kwargs.get('pv', None)
        pa = kwargs.get('pa', None)
        if pv is None:
            pv = torch.ones(batch_size, dtype=torch.long, device=device)
        if pa is None:
            pa = torch.ones(batch_size, dtype=torch.long, device=device if action_tokens is None else action_tokens.device)
        
        # Apply fusion with A-Former output
        fused_tokens = self.fusion_module(
            visual_tokens=latent_motion_tokens,  # (B, query_num, hidden_size) or None
            action_tokens=action_tokens,         # (B, query_num, hidden_size) or None
            pv=pv,                              # (B,) visual presence flags
            pa=pa                               # (B,) action presence flags  
        )
        
        visual_shape = latent_motion_tokens.shape if latent_motion_tokens is not None else 'None'
        action_shape = action_tokens.shape if action_tokens is not None else 'None'
        # print(f"🔧 Fusion result: visual {visual_shape} + action {action_shape} -> {fused_tokens.shape}")
        
        # Use fused tokens for subsequent processing
        latent_motion_tokens = fused_tokens
        
        # 4. Vector quantization (original logic)
        latent_motion_tokens_down = self.vq_down_resampler(latent_motion_tokens)
        quant, indices, commit_loss = self.vector_quantizer(latent_motion_tokens_down)

        if return_motion_token_ids_only:
            return indices

        # 5. Decoding (original logic)
        latent_motion_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(
            cond_input=cond_pixel_values,
            latent_motion_tokens=latent_motion_tokens_up
        )
        
        # 🆕 6. Action decoding (new functionality)
        decoded_actions = None
        if self.enable_action_conditioning:
            decoded_actions = self.action_decoder(latent_motion_tokens_up, embodiment_ids)
            # print(f"🔧 Decoded actions: {latent_motion_tokens_up.shape} -> {decoded_actions.shape}")
            
        if return_recons_only:
            return {
                "recons_pixel_values": recons_pixel_values,
                "indices": indices,
                "action_features": action_features,    # 🆕 调试信息
                "state_features": state_features,      # 🆕 调试信息
                "decoded_actions": decoded_actions,    # 🆕 解码的动作
            }

        # 6. Loss computation with action reconstruction
        if self.hidden_state_decoder is not None:
            recons_hidden_states = self.hidden_state_decoder(
                cond_input=cond_hidden_states,
                latent_motion_tokens=latent_motion_tokens_up
            )

        # Initialize outputs with all loss components
        outputs = {
            "recons_pixel_values": recons_pixel_values,
            "loss": torch.zeros_like(commit_loss),
            "commit_loss": commit_loss,
            "recons_loss": torch.zeros_like(commit_loss),
            "recons_hidden_loss": torch.zeros_like(commit_loss),
            "perceptual_loss": torch.zeros_like(commit_loss),
            "action_recons_loss": torch.zeros_like(commit_loss),  # 🆕 Action reconstruction loss
            # Action pipeline outputs
            "action_features": action_features,      # Raw action features from encoder
            "state_features": state_features,        # Raw state features from encoder
            "action_tokens": action_tokens,          # 🆕 Action tokens from A-Former
            "decoded_actions": decoded_actions,      # Decoded actions from action decoder
        }

        # Visual reconstruction loss
        if self.use_abs_recons_loss:
            # 🆕 Use GT images if provided (for vision drop), otherwise use target images
            target_images_for_loss = gt_images_for_loss if gt_images_for_loss is not None else target_pixel_values
            # ensure float32
            target_images_for_loss = target_images_for_loss.float()
            recons_loss = torch.abs(recons_pixel_values.float() - target_images_for_loss).mean()
        else:
            # 🆕 Use GT images if provided (for vision drop), otherwise use target images
            target_images_for_loss = gt_images_for_loss if gt_images_for_loss is not None else target_pixel_values
            # ensure float32
            target_images_for_loss = target_images_for_loss.float()
            recons_loss = F.mse_loss(target_images_for_loss, recons_pixel_values.float())
        outputs["recons_loss"] = recons_loss
        
        # 🆕 Debug info for vision reconstruction
        # if gt_images_for_loss is not None:
        #     print(f"🔧 Vision reconstruction loss (GT - dropped): {recons_loss.item():.6f}")
        # else:
        #     print(f"🔧 Vision reconstruction loss (Input): {recons_loss.item():.6f}")

        # Perceptual loss
        if self.perceptual_loss_w > 0:
            with torch.no_grad():
                # Use GT images for perceptual loss if available and force float32
                target_images_for_perceptual = gt_images_for_loss if gt_images_for_loss is not None else target_pixel_values
                target_images_for_perceptual = target_images_for_perceptual.float()
                recons_for_perceptual = recons_pixel_values.float()
                perceptual_loss = self.loss_fn_lpips.forward(
                    target_images_for_perceptual, recons_for_perceptual, normalize=True).mean()
        else:
            perceptual_loss = torch.zeros_like(recons_loss)
        outputs["perceptual_loss"] = perceptual_loss
        
        # 🆕 Action reconstruction loss
        action_recons_loss = torch.zeros_like(recons_loss)
        if self.enable_action_conditioning and decoded_actions is not None:
            # 🆕 Use gt_actions_for_loss if provided (for dropped actions), otherwise use actions
            target_actions = gt_actions_for_loss if gt_actions_for_loss is not None else actions
            
            if target_actions is not None:
                # Compute MSE loss between target actions and decoded actions
                action_recons_loss = F.mse_loss(decoded_actions.float(), target_actions.float())
                
                # Debug info
                action_type = "GT (dropped)" if gt_actions_for_loss is not None else "Input"
                # print(f"🔧 Action reconstruction loss ({action_type}): {action_recons_loss.item():.6f}")
                # print(f"🔧 Action shapes - Target: {target_actions.shape}, Decoded: {decoded_actions.shape}")
            # else:
            #     print(f"🔧 Action reconstruction loss: No target actions available")
        outputs["action_recons_loss"] = action_recons_loss

        # Total loss computation
        loss = (self.commit_loss_w * outputs["commit_loss"] + 
                self.recon_loss_w * outputs["recons_loss"] + 
                self.perceptual_loss_w * outputs["perceptual_loss"] +
                self.action_recons_loss_w * outputs["action_recons_loss"])  # 🆕 Add action loss
        
        # Hidden state reconstruction loss (if enabled)
        if self.hidden_state_decoder is not None:
            recon_hidden_loss = F.mse_loss(target_hidden_states, recons_hidden_states)
            outputs['recons_hidden_loss'] = recon_hidden_loss
            loss += self.recon_hidden_loss_w * outputs['recons_hidden_loss']

        outputs["loss"] = loss

        # Active code number (original logic)
        active_code_num = torch.tensor(torch.unique(indices).shape[0]).float().to(loss.device)
        outputs["active_code_num"] = active_code_num

        return outputs
    
    def get_action_encoder_info(self):
        """Get action encoder configuration info"""
        if not self.enable_action_conditioning:
            return "Action conditioning disabled"
        
        return {
            "config": self.action_encoder_config,
            "embodiment_configs": self.action_encoder.embodiment_configs if self.action_encoder else None
        }
