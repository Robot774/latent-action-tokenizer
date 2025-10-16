"""
EmbodimentAware Latent Motion Tokenizer Trainer
Extends the base trainer to support action conditioning and evaluation
"""
import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from .latent_motion_tokenizer_trainer import LatentMotionTokenizer_Trainer
from .trainer_utils import visualize_latent_motion_reconstruction
import random
import torch.distributed as dist


class EmbodimentAwareLatentMotionTokenizer_Trainer(LatentMotionTokenizer_Trainer):
    """
    Trainer for EmbodimentAware Latent Motion Tokenizer
    Supports action conditioning, multimodal evaluation, and action reconstruction visualization
    """
    
    def __init__(self, 
                 *args,
                 action_eval_probability=0.7,  # Probability of including actions during evaluation (DEPRECATED)
                 action_drop_prob=0.3,         # 🆕 Probability of dropping actions during training
                 vision_drop_prob=0.2,         # 🆕 Probability of dropping vision during training
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        # Drop configuration for training
        self.action_drop_prob = action_drop_prob
        self.vision_drop_prob = vision_drop_prob  # 🆕 Vision drop probability
        self.drop_warmup_steps = 50
        self._train_batches = 0
        
        # Keep eval probability for backward compatibility (but not used)
        self.action_eval_probability = action_eval_probability
        
        # Check if model supports action conditioning
        unwrapped_model = self.accelerator.unwrap_model(self.latent_motion_tokenizer)
        self.supports_action_conditioning = hasattr(unwrapped_model, 'enable_action_conditioning')
        
        if self.supports_action_conditioning:
            self.print(f"✅ EmbodimentAware model detected with action conditioning support")
            self.print(f"📊 Training drop probabilities - Action: {action_drop_prob}, Vision: {vision_drop_prob}")
        else:
            self.print(f"⚠️ Model does not support action conditioning - will run in vision-only mode")

    def _adapt_batch_for_action_conditioning(self, batch):
        """
        Adapt batch to include action conditioning data
        
        Args:
            batch: Input batch with potential action data
            
        Returns:
            dict: Batch with action conditioning fields
        """
        # First apply the base HRDT adaptation
        batch = self._adapt_hrdt_batch(batch)
        
        # Add action conditioning fields if available
        if self.supports_action_conditioning:
            # Extract action data from batch if available
            actions = batch.get('actions', None)  # (B, chunk_size, action_dim)
            states = batch.get('states', None)    # (B, 1, state_dim)
            embodiment_ids = batch.get('embodiment_ids', None)  # (B,)
            
            # Log action data availability (only once to avoid spam)
            if not hasattr(self, '_action_data_logged'):
                if actions is not None:
                    self.print(f"✅ Action data detected:")
                    self.print(f"   - actions shape: {actions.shape}")
                    if states is not None:
                        self.print(f"   - states shape: {states.shape}")
                    if embodiment_ids is not None:
                        self.print(f"   - embodiment_ids shape: {embodiment_ids.shape}")
                        unique_embodiments = torch.unique(embodiment_ids).tolist()
                        self.print(f"   - unique embodiments: {unique_embodiments}")
                else:
                    self.print(f"⚠️ No action data found in batch")
                self._action_data_logged = True
            
            # Add to batch
            batch['actions'] = actions
            batch['states'] = states  
            batch['embodiment_ids'] = embodiment_ids
            
        return batch

    def calculate_loss(self, batch, train):
        """
        Calculate loss with dual-modal drop support (action + vision)
        
        Args:
            batch: Input batch
            train: Whether in training mode
            
        Returns:
            dict: Loss dictionary
        """
        # Adapt batch for action conditioning
        batch = self._adapt_batch_for_action_conditioning(batch)
        
        # Image preprocessing (same as base class)
        rgb_seq = torch.cat([batch['rgb_initial'], batch['rgb_future']], dim=1)
        cond_pixel_values = rgb_seq[:, 0]
        target_pixel_values = rgb_seq[:, 1]
        
        # 🆕 Dual-modal drop logic during training
        should_drop_actions = False
        should_drop_vision = False
        
        if train:
            # Synchronize drop decisions across ranks via broadcast
            if dist.is_available() and dist.is_initialized():
                if self.is_main:
                    r = torch.rand(1, device=self.device)
                else:
                    r = torch.empty(1, device=self.device)
                dist.broadcast(r, src=0)
            else:
                r = torch.rand(1, device=self.device)
            
            # Three-segment partitioning: [0, action_prob) -> action_drop, [action_prob, action_prob+vision_prob) -> vision_drop, [action_prob+vision_prob, 1] -> both
            rand_val = r[0].item()
            if rand_val < self.action_drop_prob:
                should_drop_actions = True
                should_drop_vision = False
            elif rand_val < self.action_drop_prob + self.vision_drop_prob:
                should_drop_actions = False
                should_drop_vision = True
            else:
                should_drop_actions = False
                should_drop_vision = False
        
        # Prepare base forward kwargs (always pass real inputs)
        forward_kwargs = {
            'cond_pixel_values': cond_pixel_values,
            'target_pixel_values': target_pixel_values,
            'actions': batch['actions'] if (self.supports_action_conditioning and batch['actions'] is not None) else None,
            'states': batch['states'] if (self.supports_action_conditioning and batch['states'] is not None) else None,
            'embodiment_ids': batch.get('embodiment_ids', None),
        }

        # Presence flags for soft drop
        B = cond_pixel_values.shape[0]
        device = cond_pixel_values.device
        pv = torch.ones(B, dtype=torch.long, device=device)
        pa = torch.ones(B, dtype=torch.long, device=device)
        if should_drop_vision:
            pv = torch.zeros(B, dtype=torch.long, device=device)
            forward_kwargs['gt_images_for_loss'] = target_pixel_values  # supervise with GT
        if self.supports_action_conditioning and batch['actions'] is not None and should_drop_actions:
            pa = torch.zeros(B, dtype=torch.long, device=device)
            forward_kwargs['gt_actions_for_loss'] = batch['actions']  # supervise with GT

        forward_kwargs['pv'] = pv
        forward_kwargs['pa'] = pa
        
        # 🆕 Log drop status during training
        if train and (should_drop_actions or should_drop_vision):
            drop_status = []
            if should_drop_actions:
                drop_status.append(f"Action({self.action_drop_prob:.1f})")
            if should_drop_vision:
                drop_status.append(f"Vision({self.vision_drop_prob:.1f})")
            
            if not hasattr(self, '_dual_drop_logged'):
                self.print(f"🎲 Dual-modal drop enabled: {' + '.join(drop_status)}")
                self._dual_drop_logged = True
        
        # Forward pass
        loss = self.latent_motion_tokenizer(**forward_kwargs)
        
        if train:
            self._train_batches += 1
        
        # Record the mode used for this call (for TB logging without extra forwards)
        mode = 'both'
        if 'pv' in forward_kwargs and 'pa' in forward_kwargs:
            if (forward_kwargs['pa'] == 0).any() and (forward_kwargs['pv'] == 1).any():
                mode = 'action_drop'
            elif (forward_kwargs['pv'] == 0).any() and (forward_kwargs['pa'] == 1).any():
                mode = 'vision_drop'
            else:
                mode = 'both'
        if self.latent_motion_tokenizer.training:
            self._last_train_mode = mode
        else:
            self._last_eval_mode = mode

        return loss

    @torch.no_grad()
    def eval_latent_motion_reconstruction(self, visualization_dir):
        """
        Extended evaluation with both visual and action reconstruction
        🆕 Always use complete multimodal data during evaluation (no random sampling)
        
        Args:
            visualization_dir: Directory to save visualization results
        """
        os.makedirs(visualization_dir, exist_ok=True)
        self.print(f"Saving visualization results to {visualization_dir} ...")
        
        # Get evaluation batch
        batch, _ = self.eval_prefetcher.next_without_none()
        batch = self._adapt_batch_for_action_conditioning(batch)
        
        # Prepare visual inputs
        orig_rgb_seq = torch.cat([batch['rgb_initial'], batch['rgb_future']], dim=1)
        rgb_seq = orig_rgb_seq
        
        self.latent_motion_tokenizer.eval()
        
        # 🆕 Three-mode side-by-side comparison and CSV metrics (both / action-drop / vision-drop)
        compare_dir = os.path.join(visualization_dir, 'eval_compare')
        os.makedirs(compare_dir, exist_ok=True)
        self._eval_three_modes_compare(compare_dir, rgb_seq, batch)

    def _eval_visual_reconstruction(self, visualization_dir, rgb_seq, batch):
        """
        Evaluate visual reconstruction (original functionality)
        """
        outputs = self.latent_motion_tokenizer(
            cond_pixel_values=rgb_seq[:,0],
            target_pixel_values=rgb_seq[:,1],
            return_recons_only=True
        )
        
        recons_rgb_future = self.rgb_preprocessor.post_process(outputs["recons_pixel_values"]).detach().cpu()
        gt_latent_motion_ids = outputs["indices"].detach().cpu()
        orig_rgb_seq_processed = self.rgb_preprocessor.post_process(rgb_seq).detach().cpu()
        
        # Save visual reconstruction results
        visual_dir = os.path.join(visualization_dir, 'visual_only')
        os.makedirs(visual_dir, exist_ok=True)
        
        for i in range(orig_rgb_seq_processed.shape[0]):
            visualize_latent_motion_reconstruction(
                initial_frame=orig_rgb_seq_processed[i,0],
                next_frame=orig_rgb_seq_processed[i,1],
                recons_next_frame=recons_rgb_future[i],
                latent_motion_ids=gt_latent_motion_ids[i],
                path=os.path.join(visual_dir, f"{self.process_index}-{i}.png")
            )

    def _eval_complete_multimodal(self, visualization_dir, rgb_seq, batch):
        """
        🆕 Evaluate complete multimodal model (always use both action and vision)
        
        Args:
            visualization_dir: Directory to save results
            rgb_seq: RGB sequence tensor
            batch: Batch with action data
        """
        actions = batch['actions']
        states = batch['states'] 
        embodiment_ids = batch['embodiment_ids']
        
        # Complete multimodal forward pass (no dropping)
        outputs = self.latent_motion_tokenizer(
            cond_pixel_values=rgb_seq[:,0],
            target_pixel_values=rgb_seq[:,1],
            actions=actions,                    # Always provide actions during eval
            states=states,                      # Always provide states during eval
            embodiment_ids=embodiment_ids,
            return_recons_only=False
        )
        
        # Save multimodal visual results
        multimodal_dir = os.path.join(visualization_dir, 'complete_multimodal')
        os.makedirs(multimodal_dir, exist_ok=True)
        
        recons_rgb_future = self.rgb_preprocessor.post_process(outputs["recons_pixel_values"]).detach().cpu()
        gt_latent_motion_ids = outputs["indices"].detach().cpu()
        orig_rgb_seq_processed = self.rgb_preprocessor.post_process(rgb_seq).detach().cpu()
        
        batch_size = rgb_seq.shape[0]
        for i in range(batch_size):
            visualize_latent_motion_reconstruction(
                initial_frame=orig_rgb_seq_processed[i,0],
                next_frame=orig_rgb_seq_processed[i,1],
                recons_next_frame=recons_rgb_future[i],
                latent_motion_ids=gt_latent_motion_ids[i],
                path=os.path.join(multimodal_dir, f"{self.process_index}-{i}.png")
            )
        
        # Action reconstruction evaluation
        if 'decoded_actions' in outputs and outputs['decoded_actions'] is not None:
            self._eval_action_reconstruction_simple(visualization_dir, outputs, actions, embodiment_ids)

    def _eval_three_modes_compare(self, out_dir, rgb_seq, batch):
        """
        Evaluate three input modes and save side-by-side image comparisons and CSV metrics.
        Five columns: original(initial) | GT(next) | both | action-drop | vision-drop
        Also plot action overlays for RobotWin (14D) with semi-transparent curves.
        """
        import csv
        images_dir = os.path.join(out_dir, 'images')
        actions_dir = os.path.join(out_dir, 'actions')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(actions_dir, exist_ok=True)
        
        cond = rgb_seq[:, 0]
        target = rgb_seq[:, 1]
        actions = batch.get('actions', None)
        states = batch.get('states', None)
        embodiment_ids = batch.get('embodiment_ids', None)
        
        # Build kwargs for three modes (soft drop via pv/pa)
        kwargs_both = dict(
            cond_pixel_values=cond,
            target_pixel_values=target,
            actions=actions,
            states=states,
            embodiment_ids=embodiment_ids,
            # Return full outputs to include perceptual_loss for LPIPS display
            return_recons_only=False,
            pv=torch.ones(cond.shape[0], dtype=torch.long, device=cond.device),
            pa=torch.ones(cond.shape[0], dtype=torch.long, device=cond.device),
        )
        kwargs_action_drop = dict(
            cond_pixel_values=cond,
            target_pixel_values=target,
            actions=actions,
            states=states,
            embodiment_ids=embodiment_ids,
            gt_actions_for_loss=actions,
            # Return full outputs to include perceptual_loss for LPIPS display
            return_recons_only=False,
            pv=torch.ones(cond.shape[0], dtype=torch.long, device=cond.device),
            pa=torch.zeros(cond.shape[0], dtype=torch.long, device=cond.device),
        )
        kwargs_vision_drop = dict(
            cond_pixel_values=cond,
            target_pixel_values=target,
            gt_images_for_loss=target,
            actions=actions,
            states=states,
            embodiment_ids=embodiment_ids,
            # Return full outputs to include perceptual_loss for LPIPS display
            return_recons_only=False,
            pv=torch.zeros(cond.shape[0], dtype=torch.long, device=cond.device),
            pa=torch.ones(cond.shape[0], dtype=torch.long, device=cond.device),
        )
        
        # Forward passes
        outputs_both = self.latent_motion_tokenizer(**kwargs_both)
        outputs_action_drop = self.latent_motion_tokenizer(**kwargs_action_drop)
        outputs_vision_drop = self.latent_motion_tokenizer(**kwargs_vision_drop)
        
        # Post-process images
        gt_initial = self.rgb_preprocessor.post_process(rgb_seq[:, 0]).detach().cpu()
        gt_next = self.rgb_preprocessor.post_process(rgb_seq[:, 1]).detach().cpu()
        re_both = self.rgb_preprocessor.post_process(outputs_both["recons_pixel_values"]).detach().cpu()
        re_a_drop = self.rgb_preprocessor.post_process(outputs_action_drop["recons_pixel_values"]).detach().cpu()
        re_v_drop = self.rgb_preprocessor.post_process(outputs_vision_drop["recons_pixel_values"]).detach().cpu()
        
        # Decoded actions (if any)
        dec_both = outputs_both.get('decoded_actions', None)
        dec_a_drop = outputs_action_drop.get('decoded_actions', None)
        dec_v_drop = outputs_vision_drop.get('decoded_actions', None)
        
        # LPIPS (batch-level scalar extracted from model outputs)
        def _to_scalar(x):
            if isinstance(x, torch.Tensor):
                return x.detach().float().mean().item()
            try:
                return float(x)
            except Exception:
                return 0.0

        lpips_both_scalar = _to_scalar(outputs_both.get('perceptual_loss', 0.0))
        lpips_a_drop_scalar = _to_scalar(outputs_action_drop.get('perceptual_loss', 0.0))
        lpips_v_drop_scalar = _to_scalar(outputs_vision_drop.get('perceptual_loss', 0.0))

        # CSV metrics (per-sample rows) and batch-mean accumulation
        csv_path = os.path.join(out_dir, 'metrics.csv')
        write_header = not os.path.exists(csv_path)
        # Accumulators for batch means
        vm_both_list, vm_adrop_list, vm_vdrop_list = [], [], []
        am_both_list, am_adrop_list, am_vdrop_list = [], [], []  # only append when action available
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['sample_idx', 'mode', 'vision_mse', 'vision_lpips', 'action_mse', 'action_mae'])
            
            bs = rgb_seq.shape[0]
            for i in range(bs):
                # Per-sample visual MSEs (vs GT next)
                mse_both = torch.mean((re_both[i] - gt_next[i])**2).item()
                mse_a_drop = torch.mean((re_a_drop[i] - gt_next[i])**2).item()
                mse_v_drop = torch.mean((re_v_drop[i] - gt_next[i])**2).item()

                vm_both_list.append(mse_both)
                vm_adrop_list.append(mse_a_drop)
                vm_vdrop_list.append(mse_v_drop)

                # Side-by-side image with labels: original | GT | both | action-drop | vision-drop
                titles = [
                    'initial',
                    'GT',
                    f"both | MSE={mse_both:.4f} LPIPS={lpips_both_scalar:.4f}",
                    f"action-drop | MSE={mse_a_drop:.4f} LPIPS={lpips_a_drop_scalar:.4f}",
                    f"vision-drop | MSE={mse_v_drop:.4f} LPIPS={lpips_v_drop_scalar:.4f}",
                ]
                strip = self._compose_side_by_side_with_labels([
                    gt_initial[i], gt_next[i], re_both[i], re_a_drop[i], re_v_drop[i]
                ], titles)
                strip.save(os.path.join(images_dir, f"{self.process_index}-{i}.png"))
                
                # Per-sample action metrics when available (RobotWin ID=0 and decoded actions exist)
                a_mse_both = a_mae_both = ''
                a_mse_adrop = a_mae_adrop = ''
                a_mse_vdrop = a_mae_vdrop = ''
                if actions is not None and embodiment_ids is not None and embodiment_ids[i].item() == 0:
                    eff = 14
                    gt_act = actions[i, :, :eff].detach().cpu()
                    if dec_both is not None:
                        pred = dec_both[i, :, :eff].detach().cpu()
                        a_mse = torch.mean((pred - gt_act)**2).item()
                        a_mae = torch.mean(torch.abs(pred - gt_act)).item()
                        a_mse_both = f"{a_mse:.6f}"
                        a_mae_both = f"{a_mae:.6f}"
                        am_both_list.append(a_mse)
                    if dec_a_drop is not None:
                        pred = dec_a_drop[i, :, :eff].detach().cpu()
                        a_mse = torch.mean((pred - gt_act)**2).item()
                        a_mae = torch.mean(torch.abs(pred - gt_act)).item()
                        a_mse_adrop = f"{a_mse:.6f}"
                        a_mae_adrop = f"{a_mae:.6f}"
                        am_adrop_list.append(a_mse)
                    if dec_v_drop is not None:
                        pred = dec_v_drop[i, :, :eff].detach().cpu()
                        a_mse = torch.mean((pred - gt_act)**2).item()
                        a_mae = torch.mean(torch.abs(pred - gt_act)).item()
                        a_mse_vdrop = f"{a_mse:.6f}"
                        a_mae_vdrop = f"{a_mae:.6f}"
                        am_vdrop_list.append(a_mse)

                # Combined per-sample rows (vision + action in one row per mode)
                writer.writerow([i, 'both', f"{mse_both:.6f}", f"{lpips_both_scalar:.6f}", a_mse_both, a_mae_both])
                writer.writerow([i, 'action_drop', f"{mse_a_drop:.6f}", f"{lpips_a_drop_scalar:.6f}", a_mse_adrop, a_mae_adrop])
                writer.writerow([i, 'vision_drop', f"{mse_v_drop:.6f}", f"{lpips_v_drop_scalar:.6f}", a_mse_vdrop, a_mae_vdrop])
                
                # Action overlays for RobotWin (ID=0)
                if actions is not None and embodiment_ids is not None and embodiment_ids[i].item() == 0:
                    eff = 14
                    gt_act = actions[i, :, :eff].detach().cpu()
                    curves = {'GT': gt_act}
                    if dec_both is not None:
                        curves['both'] = dec_both[i, :, :eff].detach().cpu()
                    if dec_a_drop is not None:
                        curves['action_drop'] = dec_a_drop[i, :, :eff].detach().cpu()
                    if dec_v_drop is not None:
                        curves['vision_drop'] = dec_v_drop[i, :, :eff].detach().cpu()
                    
                    # Save action overlay plot
                    act_path = os.path.join(out_dir, 'actions', f"{self.process_index}-{i}.png")
                    self._plot_action_overlays(curves, act_path)

        # After per-sample rows: write batch means and append to run-root aggregator metrics
        # out_dir = .../temp_epoch_X_step_Y/visualization/eval_compare
        # run_root_dir = parent of temp_epoch_X_step_Y
        import re
        temp_dir = os.path.dirname(os.path.dirname(out_dir))
        run_root_dir = os.path.dirname(temp_dir)
        base_name = os.path.basename(temp_dir)
        m_step = re.search(r'step_(\d+)', base_name)
        step_str = m_step.group(1) if m_step else ''
        m_epoch = re.search(r'epoch_(\d+)', base_name)
        epoch_str = m_epoch.group(1) if m_epoch else ''
        agg_csv = os.path.join(run_root_dir, 'metrics.csv')
        write_header_batch = not os.path.exists(agg_csv)
        def _mean_or_blank(x):
            return f"{(sum(x)/len(x)):.6f}" if (isinstance(x, list) and len(x) > 0) else ''
        with open(agg_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header_batch:
                writer.writerow(['epoch', 'step', 'mode', 'vision_mse', 'vision_lpips', 'action_mse'])
            writer.writerow([epoch_str, step_str, 'both', _mean_or_blank(vm_both_list), f"{lpips_both_scalar:.6f}", _mean_or_blank(am_both_list)])
            writer.writerow([epoch_str, step_str, 'action_drop', _mean_or_blank(vm_adrop_list), f"{lpips_a_drop_scalar:.6f}", _mean_or_blank(am_adrop_list)])
            writer.writerow([epoch_str, step_str, 'vision_drop', _mean_or_blank(vm_vdrop_list), f"{lpips_v_drop_scalar:.6f}", _mean_or_blank(am_vdrop_list)])

        # Note: TensorBoard logging for mode-wise metrics is intentionally done in log()

    def _eval_action_reconstruction_simple(self, visualization_dir, outputs, gt_actions, embodiment_ids):
        """
        🆕 Simple action reconstruction evaluation (no mode complexity)
        """
        action_dir = os.path.join(visualization_dir, 'action_reconstruction')
        os.makedirs(action_dir, exist_ok=True)
        
        decoded_actions = outputs['decoded_actions'].detach().cpu()
        gt_actions_cpu = gt_actions.detach().cpu()
        embodiment_ids_cpu = embodiment_ids.detach().cpu()
        
        # Process each sample in the batch
        batch_size = gt_actions_cpu.shape[0]
        for i in range(batch_size):
            embodiment_id = embodiment_ids_cpu[i].item()
            
            # Only process RobotWin (embodiment_id=0) for now
            if embodiment_id != 0:
                continue
                
            effective_dims = 14
            embodiment_name = "RobotWin"
            
            # Extract effective dimensions
            gt_action_sample = gt_actions_cpu[i, :, :effective_dims]
            decoded_action_sample = decoded_actions[i, :, :effective_dims]
            
            # Create action reconstruction visualization
            self._visualize_action_reconstruction(
                gt_actions=gt_action_sample,
                decoded_actions=decoded_action_sample,
                embodiment_name=embodiment_name,
                effective_dims=effective_dims,
                save_path=os.path.join(action_dir, f"multimodal_{self.process_index}-{i}.png")
            )
            
            # Compute and log metrics
            mse_loss = F.mse_loss(decoded_action_sample, gt_action_sample).item()
            mae_loss = F.l1_loss(decoded_action_sample, gt_action_sample).item()
            
            self.print(f"📊 Action reconstruction metrics (sample {i}):")
            self.print(f"   - Embodiment: {embodiment_name} (ID: {embodiment_id})")
            self.print(f"   - MSE: {mse_loss:.6f}")
            self.print(f"   - MAE: {mae_loss:.6f}")

    def _visualize_action_reconstruction(self, gt_actions, decoded_actions, embodiment_name, effective_dims, save_path):
        """
        Create action reconstruction visualization showing 14 action dimensions over time
        
        Args:
            gt_actions: Ground truth actions (chunk_size, effective_dims)
            decoded_actions: Decoded actions (chunk_size, effective_dims)
            embodiment_name: Name of the embodiment
            effective_dims: Number of effective action dimensions
            save_path: Path to save the plot
        """
        chunk_size = gt_actions.shape[0]
        time_steps = np.arange(chunk_size)
        
        # Create subplot grid for action dimensions
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))  # 4x4 grid for up to 16 dimensions
        fig.suptitle(f'{embodiment_name} Action Reconstruction Comparison', fontsize=16)
        
        axes = axes.flatten()
        
        for dim in range(effective_dims):
            ax = axes[dim]
            
            # Plot ground truth and decoded actions
            ax.plot(time_steps, gt_actions[:, dim].numpy(), 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
            ax.plot(time_steps, decoded_actions[:, dim].numpy(), 'r--', linewidth=2, label='Decoded', alpha=0.8)
            
            # Compute dimension-wise error
            dim_mse = F.mse_loss(decoded_actions[:, dim], gt_actions[:, dim]).item()
            
            ax.set_title(f'Action Dim {dim}\nMSE: {dim_mse:.4f}', fontsize=10)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Action Value')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Set consistent y-axis limits for better comparison
            all_values = torch.cat([gt_actions[:, dim], decoded_actions[:, dim]])
            y_min, y_max = all_values.min().item(), all_values.max().item()
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        # Hide unused subplots
        for dim in range(effective_dims, len(axes)):
            axes[dim].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.print(f"📊 Action reconstruction plot saved to: {save_path}")

    def _compose_side_by_side(self, images):
        """Compose a horizontal strip of images from CHW tensors in [0,1]."""
        from PIL import Image
        pil_imgs = []
        for t in images:
            arr = (t.clamp(0,1).permute(1,2,0).numpy()*255).astype('uint8')
            pil_imgs.append(Image.fromarray(arr))
        total_w = sum(im.width for im in pil_imgs)
        max_h = max(im.height for im in pil_imgs)
        strip = Image.new('RGB', (total_w, max_h))
        x = 0
        for im in pil_imgs:
            strip.paste(im, (x, 0))
            x += im.width
        return strip

    def _compose_side_by_side_with_labels(self, images, titles):
        """Compose a horizontal strip with a top text band for titles/metrics.
        images: list of CHW tensors in [0,1]
        titles: list of strings, same length as images
        """
        from PIL import Image, ImageDraw
        pil_imgs = []
        for t in images:
            arr = (t.clamp(0,1).permute(1,2,0).numpy()*255).astype('uint8')
            pil_imgs.append(Image.fromarray(arr))
        total_w = sum(im.width for im in pil_imgs)
        max_h = max(im.height for im in pil_imgs)
        title_band = 32  # pixels
        strip = Image.new('RGB', (total_w, max_h + title_band), color=(255, 255, 255))
        draw = ImageDraw.Draw(strip)
        x = 0
        for idx, im in enumerate(pil_imgs):
            # paste image below the title band
            strip.paste(im, (x, title_band))
            # draw title centered over this tile (truncate overly long text)
            txt = titles[idx] if idx < len(titles) else ''
            # simple centering using text length approximation
            # Note: ImageDraw.textsize may depend on font; use default font implicitly
            draw.text((x + 4, 4), txt, fill=(0, 0, 0))
            x += im.width
        return strip

    def _plot_action_overlays(self, curves_dict, save_path):
        """Plot overlays: GT (black), both (blue), action-drop (orange, alpha=0.5), vision-drop (green, alpha=0.5)."""
        import matplotlib.pyplot as plt
        import numpy as np
        gt = curves_dict['GT']
        T, D = gt.shape
        time = np.arange(T)
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        axes = axes.flatten()
        for d in range(14):
            ax = axes[d]
            ax.plot(time, gt[:, d].numpy(), color='k', linewidth=2, label='GT', alpha=1.0)
            if 'both' in curves_dict:
                ax.plot(time, curves_dict['both'][:, d].numpy(), color='b', linewidth=1.8, label='both', alpha=0.9)
            if 'action_drop' in curves_dict:
                ax.plot(time, curves_dict['action_drop'][:, d].numpy(), color='orange', linewidth=1.5, label='action-drop', alpha=0.5)
            if 'vision_drop' in curves_dict:
                ax.plot(time, curves_dict['vision_drop'][:, d].numpy(), color='g', linewidth=1.5, label='vision-drop', alpha=0.5)
            ax.grid(True, alpha=0.3)
            if d == 0:
                ax.legend(fontsize=8)
        for d in range(14, len(axes)):
            axes[d].set_visible(False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def log(self, log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step):
        """
        Extended logging with action-related metrics
        """
        # Call parent logging
        super().log(log_loss, eval_log_loss, cum_load_time, clock, epoch, batch_idx, step)
        
        # Additional logging for action-related losses
        if self.is_main and self.supports_action_conditioning:
            action_losses = ['action_recons_loss']
            for loss_name in action_losses:
                if loss_name in log_loss:
                    self.writer.add_scalar(f"train_{loss_name}", log_loss[loss_name], step)
                if loss_name in eval_log_loss:
                    self.writer.add_scalar(f"eval_{loss_name}", eval_log_loss[loss_name], step)

        # Mode-specific eval curves to separate subfolders (no extra forward)
        if self.is_main and hasattr(self, '_last_eval_mode'):
            mode = self._last_eval_mode
            writer = getattr(self, 'eval_mode_writers', {}).get(mode, None)
            if writer is not None:
                tag_key_pairs = [
                    ('eval/vision_mse', 'recons_loss'),
                    ('eval/vision_lpips', 'perceptual_loss'),
                    ('eval/action_mse', 'action_recons_loss'),
                    ('eval/commit_loss', 'commit_loss'),
                    ('eval/loss', 'loss'),
                ]
                for tag, key in tag_key_pairs:
                    if key in eval_log_loss:
                        writer.add_scalar(tag, eval_log_loss[key], step)
