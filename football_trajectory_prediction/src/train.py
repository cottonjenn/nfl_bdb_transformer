"""
Training pipeline for trajectory prediction transformer.
Key improvements:
1. Proper loss masking for variable-length outputs
2. Gradient clipping and learning rate warmup
3. Multiple loss components (position + velocity consistency)
4. Better logging and checkpointing
"""

import torch
import torch.nn as nn
import time
from pathlib import Path
import sys
sys.path.append('..')
import config
from src.model import TrajectoryTransformer


class TrajectoryLoss(nn.Module):
    """
    Combined loss function for trajectory prediction.
    Components:
    1. Position MSE: Direct coordinate prediction error
    2. Velocity MSE: Consistency of predicted movement
    3. Final position MSE: Extra weight on endpoint accuracy
    """
    
    def __init__(self, vel_weight=0.3, final_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.vel_weight = vel_weight
        self.final_weight = final_weight
        
    def forward(self, pred, target, mask):
        """
        Args:
            pred: [B, P, T, 2] predicted positions
            target: [B, P, T, 2] ground truth positions
            mask: [B, P, T] (True = valid frame)
        Returns:
            total_loss, loss_dict
        """
        B, P, T, _ = pred.shape
        
        # Check for NaN/Inf in inputs
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            print("ERROR: NaN/Inf in predictions passed to loss")
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.isnan(target).any() or torch.isinf(target).any():
            print("ERROR: NaN/Inf in targets passed to loss")
            target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Expand mask for coordinates
        mask_expanded = mask.unsqueeze(-1).expand_as(pred).float()
        
        # Ensure we have valid frames
        valid_frames = mask_expanded.sum()
        if valid_frames < 1e-8:
            # Fallback: return small loss if no valid frames
            zero_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
            return zero_loss, {
                'position': 0.0,
                'velocity': 0.0,
                'final_pos': 0.0,
                'total': 0.0
            }
        
        # 1. Position loss
        pos_loss = self.mse(pred, target)
        pos_loss = (pos_loss * mask_expanded).sum() / (valid_frames + 1e-8)
        
        # Check for NaN
        if torch.isnan(pos_loss):
            pos_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 2. Velocity loss (consistency of movement)
        if T > 1:
            pred_vel = pred[:, :, 1:] - pred[:, :, :-1]
            target_vel = target[:, :, 1:] - target[:, :, :-1]
            vel_mask = (mask[:, :, 1:] & mask[:, :, :-1]).unsqueeze(-1).expand_as(pred_vel).float()
            
            vel_valid = vel_mask.sum()
            if vel_valid > 1e-8:
                vel_loss = self.mse(pred_vel, target_vel)
                vel_loss = (vel_loss * vel_mask).sum() / (vel_valid + 1e-8)
            else:
                vel_loss = torch.tensor(0.0, device=pred.device)
        else:
            vel_loss = torch.tensor(0.0, device=pred.device)
        
        # Check for NaN
        if torch.isnan(vel_loss):
            vel_loss = torch.tensor(0.0, device=pred.device)
        
        # 3. Final position loss (where players end up)
        # Find last valid frame for each player
        last_valid_idx = self._get_last_valid_idx(mask)
        batch_idx = torch.arange(B, device=pred.device).unsqueeze(1).expand(B, P)
        player_idx = torch.arange(P, device=pred.device).unsqueeze(0).expand(B, P)
        
        pred_final = pred[batch_idx, player_idx, last_valid_idx]
        target_final = target[batch_idx, player_idx, last_valid_idx]
        player_valid = mask.any(dim=-1).float()
        
        valid_players = player_valid.sum()
        if valid_players > 1e-8:
            final_loss = self.mse(pred_final, target_final).mean(dim=-1)
            final_loss = (final_loss * player_valid).sum() / (valid_players + 1e-8)
        else:
            final_loss = torch.tensor(0.0, device=pred.device)
        
        # Check for NaN
        if torch.isnan(final_loss):
            final_loss = torch.tensor(0.0, device=pred.device)
        
        # Combined loss
        total = pos_loss + self.vel_weight * vel_loss + self.final_weight * final_loss
        
        # Final NaN check
        if torch.isnan(total) or torch.isinf(total):
            print("ERROR: NaN/Inf in total loss, returning zero")
            total = torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        return total, {
            'position': pos_loss.item() if isinstance(pos_loss, torch.Tensor) else pos_loss,
            'velocity': vel_loss.item() if isinstance(vel_loss, torch.Tensor) else vel_loss,
            'final_pos': final_loss.item() if isinstance(final_loss, torch.Tensor) else final_loss,
            'total': total.item() if isinstance(total, torch.Tensor) else total
        }
    
    def _get_last_valid_idx(self, mask):
        """Get index of last valid frame for each player."""
        B, P, T = mask.shape
        indices = torch.arange(T, device=mask.device).view(1, 1, T).expand(B, P, T)
        masked_indices = indices * mask.long()
        last_idx = masked_indices.max(dim=-1).values
        return last_idx.clamp(0, T - 1)


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine decay."""
    
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_step = 0
        
    def step(self):
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def _get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


import math

class Trainer:
    """Handles model training and validation."""
    
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function
        self.criterion = TrajectoryLoss(vel_weight=0.3, final_weight=0.5)
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0.01
        )
        
        # Calculate total steps for scheduler
        total_steps = len(train_loader) * config.NUM_EPOCHS
        warmup_steps = len(train_loader) * 2  # 2 epochs warmup
        
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, warmup_steps, total_steps
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_pos_loss': [], 'val_pos_loss': [],
            'train_vel_loss': [], 'val_vel_loss': [],
            'learning_rates': [], 'epoch_times': []
        }
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def _map_predictions_to_output_indices(self, predictions, batch):
        """Map predictions from input player indices to output player indices."""
        B, P_in, T, C = predictions.shape
        P_out = batch['output'].shape[1]
        
        mapped = torch.zeros(B, P_out, T, C, device=predictions.device, dtype=predictions.dtype)
        
        idx_mappings = batch.get('input_to_output_idx', [])
        for b in range(B):
            if b < len(idx_mappings):
                for in_idx, out_idx in idx_mappings[b].items():
                    if in_idx < P_in and out_idx < P_out:
                        mapped[b, out_idx] = predictions[b, in_idx]
        
        return mapped
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_pos_loss = 0
        total_vel_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            inputs = batch['input'].to(self.device)
            input_mask = batch['input_mask'].to(self.device)
            targets = batch['output'].to(self.device)
            output_mask = batch['output_mask'].to(self.device)
            ball_pos = batch['ball_position'].to(self.device)
            
            # Debug first batch
            if batch_idx == 0 and len(self.history['train_loss']) == 0:
                print(f"\n=== First Batch Shapes ===")
                print(f"inputs: {inputs.shape}, mask: {input_mask.shape}")
                print(f"targets: {targets.shape}, mask: {output_mask.shape}")
            
            self.optimizer.zero_grad()
            
            # Check for NaN/Inf in inputs
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print(f"ERROR: NaN/Inf detected in inputs at batch {batch_idx}")
                print(f"  NaN count: {torch.isnan(inputs).sum().item()}")
                print(f"  Inf count: {torch.isinf(inputs).sum().item()}")
                raise ValueError("NaN/Inf in inputs")
            
            # Forward pass
            predictions = self.model(inputs, input_mask, ball_pos)
            
            # Check for NaN/Inf in predictions
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                print(f"ERROR: NaN/Inf detected in model predictions at batch {batch_idx}")
                print(f"  NaN count: {torch.isnan(predictions).sum().item()}")
                print(f"  Inf count: {torch.isinf(predictions).sum().item()}")
                print(f"  Prediction stats: min={predictions.min().item():.4f}, max={predictions.max().item():.4f}, mean={predictions.mean().item():.4f}")
                raise ValueError("NaN/Inf in model predictions")
            
            # Map to output indices
            predictions = self._map_predictions_to_output_indices(predictions, batch)
            
            # Ensure shapes match
            if predictions.shape[2] != targets.shape[2]:
                min_frames = min(predictions.shape[2], targets.shape[2])
                predictions = predictions[:, :, :min_frames]
                targets = targets[:, :, :min_frames]
                output_mask = output_mask[:, :, :min_frames]
            
            # Calculate loss
            loss, loss_dict = self.criterion(predictions, targets, output_mask)
            
            # Check for NaN in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"ERROR: NaN/Inf detected in loss at batch {batch_idx}")
                print(f"  Loss value: {loss.item()}")
                print(f"  Loss dict: {loss_dict}")
                raise ValueError("NaN/Inf in loss")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            lr = self.scheduler.step()
            
            total_loss += loss_dict['total']
            total_pos_loss += loss_dict['position']
            total_vel_loss += loss_dict['velocity']
            num_batches += 1
            
            # Log periodically
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)} | "
                      f"Loss: {loss_dict['total']:.4f} | LR: {lr:.2e}")
        
        return {
            'total': total_loss / num_batches,
            'position': total_pos_loss / num_batches,
            'velocity': total_vel_loss / num_batches
        }
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        total_pos_loss = 0
        total_vel_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch['input'].to(self.device)
                input_mask = batch['input_mask'].to(self.device)
                targets = batch['output'].to(self.device)
                output_mask = batch['output_mask'].to(self.device)
                ball_pos = batch['ball_position'].to(self.device)
                
                predictions = self.model(inputs, input_mask, ball_pos)
                predictions = self._map_predictions_to_output_indices(predictions, batch)
                
                if predictions.shape[2] != targets.shape[2]:
                    min_frames = min(predictions.shape[2], targets.shape[2])
                    predictions = predictions[:, :, :min_frames]
                    targets = targets[:, :, :min_frames]
                    output_mask = output_mask[:, :, :min_frames]
                
                loss, loss_dict = self.criterion(predictions, targets, output_mask)
                
                total_loss += loss_dict['total']
                total_pos_loss += loss_dict['position']
                total_vel_loss += loss_dict['velocity']
                num_batches += 1
        
        return {
            'total': total_loss / num_batches,
            'position': total_pos_loss / num_batches,
            'velocity': total_vel_loss / num_batches
        }
    
    def train(self, num_epochs):
        """Train for specified number of epochs."""
        print("\n" + "=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print()
        
        try:
            for epoch in range(num_epochs):
                epoch_start = time.time()
                
                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print("-" * 40)
                
                # Train
                train_losses = self.train_epoch()
                
                # Validate
                val_losses = self.validate()
                
                epoch_time = time.time() - epoch_start
                
                # Store history
                self.history['train_loss'].append(train_losses['total'])
                self.history['val_loss'].append(val_losses['total'])
                self.history['train_pos_loss'].append(train_losses['position'])
                self.history['val_pos_loss'].append(val_losses['position'])
                self.history['train_vel_loss'].append(train_losses['velocity'])
                self.history['val_vel_loss'].append(val_losses['velocity'])
                self.history['epoch_times'].append(epoch_time)
                self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                
                # Print summary
                print(f"\nEpoch {epoch + 1} Summary:")
                print(f"  Train Loss: {train_losses['total']:.6f} "
                      f"(pos: {train_losses['position']:.4f}, vel: {train_losses['velocity']:.4f})")
                print(f"  Val Loss:   {val_losses['total']:.6f} "
                      f"(pos: {val_losses['position']:.4f}, vel: {val_losses['velocity']:.4f})")
                print(f"  Time: {epoch_time:.2f}s")
                
                # Save best model
                if val_losses['total'] < self.best_val_loss - config.MIN_DELTA:
                    self.best_val_loss = val_losses['total']
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(config.BEST_MODEL_PATH, is_best=True)
                    print(f"  ✓ New best model saved!")
                else:
                    self.epochs_without_improvement += 1
                    print(f"  No improvement for {self.epochs_without_improvement} epochs")
                
                # Regular checkpoint
                if (epoch + 1) % config.SAVE_EVERY_N_EPOCHS == 0:
                    self.save_checkpoint(config.CHECKPOINT_PATH)
                
                # Early stopping
                if self.epochs_without_improvement >= config.PATIENCE:
                    print(f"\nEarly stopping triggered!")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            if self.best_val_loss < float('inf'):
                print(f"Best model saved to: {config.BEST_MODEL_PATH}")
            raise
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        
    def save_checkpoint(self, path, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'config': {
                'd_model': config.D_MODEL,
                'nhead': config.NHEAD,
                'num_encoder_layers': config.NUM_ENCODER_LAYERS,
                'num_decoder_layers': config.NUM_DECODER_LAYERS,
                'dropout': config.DROPOUT,
                'max_output_frames': config.MAX_OUTPUT_FRAMES
            }
        }
        torch.save(checkpoint, path)