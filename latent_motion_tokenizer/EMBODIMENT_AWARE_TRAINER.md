"""
EmbodimentAware Trainer Documentation
=====================================

This document explains the EmbodimentAwareLatentMotionTokenizer_Trainer implementation and usage.

## Overview

The `EmbodimentAwareLatentMotionTokenizer_Trainer` extends the base `LatentMotionTokenizer_Trainer` 
to support action conditioning and multimodal evaluation. It enables training and evaluation of 
models that can process both visual and action data simultaneously.

## 🆕 Key Features

### 1. **Dual-Modal Drop Mechanism**

Enhanced training regularization with independent action and vision drop:

```python
trainer = EmbodimentAwareLatentMotionTokenizer_Trainer(
    action_drop_prob=0.3,  # 30% probability to drop actions during training
    vision_drop_prob=0.2,  # 20% probability to drop vision during training
    # ... other parameters
)
```

#### Drop Strategy Details:

**Action Drop**:
- Randomly sets `actions=None` during training with probability `action_drop_prob`
- Preserves original actions via `gt_actions_for_loss` for loss computation
- Model handles missing actions through fusion module's missing action tokens

**Vision Drop**:
- Randomly sets `target_pixel_values=None` during training with probability `vision_drop_prob`
- Preserves original images via `gt_images_for_loss` for loss computation
- Model uses zero images for forward pass but GT images for loss calculation

**Combined Drop**:
- Action and vision drops are independent and can occur simultaneously
- Loss computation works as long as original data exists
- Supports all combinations: no drop, action-only drop, vision-only drop, dual drop

### 2. **Evaluation Strategy**

**Evaluation Behavior**:
- 🚫 No modal dropping during evaluation
- ✅ Always uses complete multimodal data (action + vision)
- 📊 Generates `complete_multimodal/` visualization directory
- 📈 Provides 14-dimensional action reconstruction visualization for RobotWin (embodiment_id=0)

### 3. **Action Conditioning Support**
- **Automatic Detection**: Checks if the model supports action conditioning via `enable_action_conditioning` attribute
- **Data Flow Integration**: Seamlessly passes action data (actions, states, embodiment_ids) to the model
- **Backward Compatibility**: Falls back to vision-only mode if action data is unavailable

## Architecture

```
Base Trainer (LatentMotionTokenizer_Trainer)
    ↓ (inherits)
EmbodimentAware Trainer
    ├── Action Data Adaptation
    ├── Probabilistic Evaluation Modes  
    ├── Action Reconstruction Metrics
    └── Visualization Generation
```

## Data Flow

### Training/Evaluation Pipeline:
```
1. Batch Input
   ├── Visual: rgb_initial, rgb_future
   └── Action: actions, states, embodiment_ids

2. Data Adaptation (_adapt_batch_for_action_conditioning)
   ├── HRDT format adaptation (base class)
   └── Action data extraction and validation

3. Loss Calculation (calculate_loss)
   ├── Visual preprocessing
   ├── Action conditioning (if available)
   └── Model forward pass

4. Evaluation (eval_latent_motion_reconstruction)
   ├── Visual-only evaluation (base functionality)
   └── Action conditioning evaluation (new)
```

### Action Evaluation Flow:
```
1. Mode Sampling
   └── Probabilistically sample: action_only | motion_only | both

2. Mode-Specific Forward Pass
   ├── Prepare mode-specific inputs
   ├── Model inference
   └── Save mode-specific visual results

3. Action Reconstruction Analysis
   ├── Extract effective dimensions (14D for RobotWin)
   ├── Compute MSE/MAE metrics
   └── Generate time-series plots
```

## Configuration

### Initialization Parameters:
```python
trainer = EmbodimentAwareLatentMotionTokenizer_Trainer(
    latent_motion_tokenizer=model,      # EmbodimentAware model
    rgb_preprocessor=preprocessor,       # Image preprocessing
    train_dataloader=train_loader,       # Training data
    eval_dataloader=eval_loader,         # Evaluation data  
    save_path="checkpoints/",            # Checkpoint directory
    action_eval_probability=0.7,         # Action evaluation probability
    # ... other base trainer parameters
)
```

### Action Evaluation Probability:
- Controls the likelihood of including actions during evaluation
- Default: 0.7 (70% chance of action conditioning)
- Distribution: action_only (23%) | motion_only (23%) | both (54%)

## Usage Example

### Basic Training:
```python
# Initialize trainer
trainer = EmbodimentAwareLatentMotionTokenizer_Trainer(
    latent_motion_tokenizer=embodiment_aware_model,
    rgb_preprocessor=preprocessor,
    train_dataloader=train_loader,
    eval_dataloader=eval_loader,
    save_path="experiments/embodiment_aware/",
    action_eval_probability=0.8
)

# Start training
trainer.train()
```

### Evaluation Only:
```python
# Run evaluation with visualization
trainer.eval_latent_motion_reconstruction("results/eval_epoch_10/")
```

## Output Structure

The trainer generates organized evaluation results:

```
visualization_dir/
├── visual_only/                    # Base visual reconstruction
│   ├── 0-0.png, 0-1.png, ...      # Visual reconstruction images
├── mode_action_only/               # Action-only mode results  
│   └── 0-*.png                     # Visual results for action-only samples
├── mode_motion_only/               # Motion-only mode results
│   └── 0-*.png                     # Visual results for motion-only samples  
├── mode_both/                      # Both modalities mode results
│   └── 0-*.png                     # Visual results for both-modality samples
└── action_reconstruction/          # Action analysis
    ├── both_0-0.png               # Action reconstruction plots (both mode)
    └── action_only_0-*.png        # Action reconstruction plots (action-only mode)
```

## Action Visualization Details

### Plot Structure (4x4 Grid):
- **14 Subplots**: One for each effective action dimension (RobotWin)
- **Time Series**: X-axis = time steps (chunk_size), Y-axis = action values
- **Comparison Lines**: Blue solid (ground truth) vs Red dashed (decoded)  
- **Per-Dimension MSE**: Displayed in subplot titles
- **Automatic Scaling**: Y-axis limits adjusted per dimension for clarity

### Metrics Logged:
- **MSE**: Mean Squared Error between ground truth and decoded actions
- **MAE**: Mean Absolute Error between ground truth and decoded actions
- **Per-Sample**: Metrics computed and logged for each evaluated sample

## Embodiment Support

### Current Implementation:
- **RobotWin (ID=0)**: Full support with 14 effective action dimensions
- **EgoDex (ID≠0)**: Skipped as requested (can be added later)

### Adding New Embodiments:
```python
# In _eval_action_reconstruction method:
if embodiment_id == 0:  # RobotWin
    effective_dims = 14
    embodiment_name = "RobotWin"
elif embodiment_id == 1:  # New embodiment
    effective_dims = 48  # Or other dimension count
    embodiment_name = "EgoDex"
```

## Integration with Base Trainer

### Inherited Functionality:
- ✅ Training loop and optimization
- ✅ Checkpoint saving and loading  
- ✅ Learning rate scheduling
- ✅ Distributed training support
- ✅ TensorBoard logging

### Extended Functionality:
- 🆕 Action data handling
- 🆕 Multimodal evaluation modes
- 🆕 Action reconstruction metrics
- 🆕 Action visualization plots
- 🆕 Enhanced logging for action losses

## Testing

Run the test suite to verify functionality:
```bash
cd /dataset_rc_mm/chenby10@xiaopeng.com/Moto_copy/latent_motion_tokenizer
python test_embodiment_aware_trainer.py
```

Expected outputs:
- ✅ Trainer initialization
- ✅ Batch adaptation with action data
- ✅ Evaluation mode sampling  
- ✅ Loss calculation with action conditioning
- ✅ Action visualization generation
- ✅ Full evaluation pipeline

## Notes

### Dependencies:
- Inherits all base trainer dependencies
- Additional: `matplotlib` for action visualization
- Compatible with existing HRDT data format

### Performance:
- Minimal overhead when action conditioning is disabled
- Efficient probabilistic mode sampling
- On-demand visualization generation

### Future Extensions:
- Multi-embodiment support expansion
- Additional action metrics (correlation, etc.)
- Real-time action reconstruction monitoring
- Custom evaluation mode configurations
"""
