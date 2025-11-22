"""
Visualization tools for trajectory predictions.
Creates plots comparing predicted vs actual trajectories.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pickle
import pandas as pd

import sys
sys.path.append('..')
import config
from src.dataset import FootballTrajectoryDataset
from src.model import TrajectoryTransformer


class TrajectoryVisualizer:
    """Visualize predicted vs actual player trajectories."""
    
    def __init__(self, model, dataset, device, scaling_params_path):
        """
        Args:
            model: Trained transformer model
            dataset: FootballTrajectoryDataset instance
            device: 'cuda' or 'cpu'
            scaling_params_path: Path to scaling parameters
        """
        self.model = model.to(device)
        self.model.eval()
        self.dataset = dataset
        self.device = device
        
        # Load scaling parameters
        with open(scaling_params_path, 'rb') as f:
            self.scaling_params = pickle.load(f)
    
    def denormalize_coordinates(self, coords_normalized):
        """Convert normalized coordinates back to yards."""
        x_mean = self.scaling_params['x_norm']['mean']
        x_std = self.scaling_params['x_norm']['std']
        y_mean = self.scaling_params['y_norm']['mean']
        y_std = self.scaling_params['y_norm']['std']
        
        coords_yards = coords_normalized.clone()
        coords_yards[..., 0] = coords_normalized[..., 0] * x_std + x_mean
        coords_yards[..., 1] = coords_normalized[..., 1] * y_std + y_mean
        
        return coords_yards
    
    def draw_field(self, ax):
        """Draw football field on matplotlib axis."""
        # Field dimensions (yards)
        field_length = 120
        field_width = 53.3
        
        # Green field
        ax.add_patch(patches.Rectangle(
            (0, 0), field_length, field_width,
            facecolor='#2d5f2e', edgecolor='white', linewidth=2
        ))
        
        # Yard lines
        for yard in range(10, 110, 10):
            ax.plot([yard, yard], [0, field_width], 'white', linewidth=1, alpha=0.5)
        
        # End zones
        ax.add_patch(patches.Rectangle(
            (0, 0), 10, field_width,
            facecolor='#1a3d1a', alpha=0.3
        ))
        ax.add_patch(patches.Rectangle(
            (110, 0), 10, field_width,
            facecolor='#1a3d1a', alpha=0.3
        ))
        
        # 50 yard line
        ax.plot([60, 60], [0, field_width], 'white', linewidth=2)
        
        ax.set_xlim(0, field_length)
        ax.set_ylim(0, field_width)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def plot_single_play(self, play_idx, save_path=None):
        """
        Plot trajectories for a single play.
        
        Args:
            play_idx: Index of play in dataset
            save_path: Path to save figure (optional)
        """
        # Get data
        data = self.dataset[play_idx]
        
        # Add batch dimension and move to device
        inputs = data['input'].unsqueeze(0).to(self.device)
        input_mask = data['input_mask'].unsqueeze(0).to(self.device)
        targets = data['output'].unsqueeze(0).to(self.device)
        output_mask = data['output_mask'].unsqueeze(0).to(self.device)
        ball_pos = data['ball_position'].unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(inputs, input_mask, ball_pos)
        
        # Adjust shapes - map predictions from input indices to output indices
        batch_size, num_target_players, max_frames, coords = targets.shape
        
        # Get mapping from input player index to output player index
        input_to_output_idx = data.get('input_to_output_idx', {})
        
        # Initialize predictions tensor with zeros
        predictions_mapped = torch.zeros(
            1, num_target_players, max_frames, coords,
            device=predictions.device,
            dtype=predictions.dtype
        )
        
        # Map each prediction from input index to correct output index
        if input_to_output_idx:
            for input_idx, output_idx in input_to_output_idx.items():
                if input_idx < predictions.shape[1] and output_idx < num_target_players:
                    predictions_mapped[0, output_idx, :max_frames, :] = predictions[0, input_idx, :max_frames, :]
        
        predictions = predictions_mapped
        
        # Denormalize
        pred_yards = self.denormalize_coordinates(predictions[0].cpu())
        target_yards = self.denormalize_coordinates(targets[0].cpu())
        mask = output_mask[0].cpu()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        self.draw_field(ax)
        
        # Plot each player's trajectory
        num_players = pred_yards.shape[0]
        colors = plt.cm.rainbow(np.linspace(0, 1, num_players))
        
        for player_idx in range(num_players):
            # Get valid frames
            valid_frames = mask[player_idx].nonzero(as_tuple=True)[0]
            
            if len(valid_frames) > 0:
                # Actual trajectory
                actual = target_yards[player_idx, valid_frames, :]
                ax.plot(actual[:, 0], actual[:, 1], 
                       color=colors[player_idx], linewidth=2, 
                       label=f'Player {player_idx+1} (Actual)', linestyle='-')
                
                # Predicted trajectory
                pred = pred_yards[player_idx, valid_frames, :]
                ax.plot(pred[:, 0], pred[:, 1], 
                       color=colors[player_idx], linewidth=2, 
                       label=f'Player {player_idx+1} (Pred)', linestyle='--', alpha=0.7)
                
                # Start and end markers
                ax.scatter(actual[0, 0], actual[0, 1], 
                          color=colors[player_idx], s=100, marker='o', 
                          edgecolor='white', linewidth=2, zorder=5)
                ax.scatter(actual[-1, 0], actual[-1, 1], 
                          color=colors[player_idx], s=100, marker='s', 
                          edgecolor='white', linewidth=2, zorder=5)
        
        # Ball landing position
        ball_denorm = self.denormalize_coordinates(ball_pos[0].cpu().unsqueeze(0).unsqueeze(0))[0, 0]
        ax.scatter(ball_denorm[0], ball_denorm[1], 
                  color='white', s=200, marker='*', 
                  edgecolor='black', linewidth=2, zorder=10,
                  label='Ball Landing')
        
        ax.set_title(f"Play {data['metadata']['play_id']} - Trajectory Prediction\n"
                    f"Solid = Actual | Dashed = Predicted", 
                    fontsize=14, color='white', pad=20)
        
        # Legend (only show first few players to avoid clutter)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:min(10, len(handles))], labels[:min(10, len(labels))],
                 loc='upper left', framealpha=0.9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                       facecolor='#2d5f2e')
            print(f"✓ Saved plot to: {save_path}")
        
        plt.show()
    
    def plot_multiple_plays(self, num_plays=4, save_path=None):
        """
        Create a grid of multiple play visualizations.
        
        Args:
            num_plays: Number of plays to visualize
            save_path: Path to save figure (optional)
        """
        rows = int(np.ceil(num_plays / 2))
        cols = 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10 * rows))
        axes = axes.flatten() if num_plays > 1 else [axes]
        
        for i in range(num_plays):
            if i >= len(self.dataset):
                break
            
            ax = axes[i]
            
            # Get data
            data = self.dataset[i]
            inputs = data['input'].unsqueeze(0).to(self.device)
            input_mask = data['input_mask'].unsqueeze(0).to(self.device)
            targets = data['output'].unsqueeze(0).to(self.device)
            output_mask = data['output_mask'].unsqueeze(0).to(self.device)
            ball_pos = data['ball_position'].unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                predictions = self.model(inputs, input_mask, ball_pos)
            
            # Adjust shapes - map predictions from input indices to output indices
            batch_size, num_target_players, max_frames, coords = targets.shape
            
            # Get mapping from input player index to output player index
            input_to_output_idx = data.get('input_to_output_idx', {})
            
            # Initialize predictions tensor with zeros
            predictions_mapped = torch.zeros(
                1, num_target_players, max_frames, coords,
                device=predictions.device,
                dtype=predictions.dtype
            )
            
            # Map each prediction from input index to correct output index
            if input_to_output_idx:
                for input_idx, output_idx in input_to_output_idx.items():
                    if input_idx < predictions.shape[1] and output_idx < num_target_players:
                        predictions_mapped[0, output_idx, :max_frames, :] = predictions[0, input_idx, :max_frames, :]
            
            predictions = predictions_mapped
            
            pred_yards = self.denormalize_coordinates(predictions[0].cpu())
            target_yards = self.denormalize_coordinates(targets[0].cpu())
            mask = output_mask[0].cpu()
            
            # Draw field
            self.draw_field(ax)
            
            # Plot trajectories
            num_players = pred_yards.shape[0]
            colors = plt.cm.rainbow(np.linspace(0, 1, num_players))
            
            for player_idx in range(num_players):
                valid_frames = mask[player_idx].nonzero(as_tuple=True)[0]
                
                if len(valid_frames) > 0:
                    actual = target_yards[player_idx, valid_frames, :]
                    pred = pred_yards[player_idx, valid_frames, :]
                    
                    ax.plot(actual[:, 0], actual[:, 1], 
                           color=colors[player_idx], linewidth=2, linestyle='-')
                    ax.plot(pred[:, 0], pred[:, 1], 
                           color=colors[player_idx], linewidth=2, 
                           linestyle='--', alpha=0.7)
                    
                    ax.scatter(actual[0, 0], actual[0, 1], 
                              color=colors[player_idx], s=50, marker='o', 
                              edgecolor='white', linewidth=1, zorder=5)
            
            # Ball landing
            ball_denorm = self.denormalize_coordinates(ball_pos[0].cpu().unsqueeze(0).unsqueeze(0))[0, 0]
            ax.scatter(ball_denorm[0], ball_denorm[1], 
                      color='white', s=150, marker='*', 
                      edgecolor='black', linewidth=2, zorder=10)
            
            ax.set_title(f"Play {data['metadata']['play_id']}", 
                        fontsize=12, color='white')
        
        # Hide unused subplots
        for i in range(num_plays, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle("Trajectory Predictions (Solid = Actual, Dashed = Predicted)", 
                    fontsize=16, color='white', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                       facecolor='#2d5f2e')
            print(f"✓ Saved plot to: {save_path}")
        
        plt.show()
    
    def plot_error_heatmap(self, play_idx, save_path=None):
        """
        Create a heatmap showing prediction error at each frame.
        
        Args:
            play_idx: Index of play in dataset
            save_path: Path to save figure (optional)
        """
        # Get data and predictions
        data = self.dataset[play_idx]
        inputs = data['input'].unsqueeze(0).to(self.device)
        input_mask = data['input_mask'].unsqueeze(0).to(self.device)
        targets = data['output'].unsqueeze(0).to(self.device)
        output_mask = data['output_mask'].unsqueeze(0).to(self.device)
        ball_pos = data['ball_position'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(inputs, input_mask, ball_pos)
        
        # Adjust shapes
        batch_size, num_players, max_frames, coords = targets.shape
        if predictions.shape[1] == 1:
            predictions = predictions.expand(-1, num_players, -1, -1)
        predictions = predictions[:, :num_players, :max_frames, :]
        
        # Denormalize
        pred_yards = self.denormalize_coordinates(predictions[0].cpu())
        target_yards = self.denormalize_coordinates(targets[0].cpu())
        mask = output_mask[0].cpu()
        
        # Calculate errors
        diff = pred_yards - target_yards
        errors = torch.sqrt((diff ** 2).sum(dim=-1))  # [players, frames]
        
        # Mask invalid positions
        errors = errors * mask.float()
        errors = errors.numpy()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.imshow(errors, aspect='auto', cmap='hot', interpolation='nearest')
        
        ax.set_xlabel('Frame', fontsize=12)
        ax.set_ylabel('Player', fontsize=12)
        ax.set_title(f'Prediction Error Heatmap - Play {data["metadata"]["play_id"]}', 
                    fontsize=14)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Error (yards)', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved heatmap to: {save_path}")
        
        plt.show()


def visualize_predictions(model_path=None, play_idx=0, mode='single'):
    """
    Convenience function to visualize predictions.
    
    Args:
        model_path: Path to model checkpoint (default: best model)
        play_idx: Which play to visualize
        mode: 'single', 'multiple', or 'heatmap'
    """
    model_path = model_path or config.BEST_MODEL_PATH
    
    print(f"Loading model from: {model_path}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    model = TrajectoryTransformer()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load dataset
    dataset = FootballTrajectoryDataset(
        config.PROCESSED_INPUT,
        config.PROCESSED_OUTPUT,
        config.METADATA_FILE
    )
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    # Create visualizer
    visualizer = TrajectoryVisualizer(
        model,
        dataset,
        device,
        config.SCALING_PARAMS
    )
    
    # Generate visualization
    save_dir = config.REPORTS_DIR
    
    if mode == 'single':
        save_path = f'{save_dir}/trajectory_play_{play_idx}.png'
        visualizer.plot_single_play(play_idx, save_path)
    elif mode == 'multiple':
        save_path = f'{save_dir}/trajectory_multiple_plays.png'
        visualizer.plot_multiple_plays(num_plays=4, save_path=save_path)
    elif mode == 'heatmap':
        save_path = f'{save_dir}/error_heatmap_play_{play_idx}.png'
        visualizer.plot_error_heatmap(play_idx, save_path)
    else:
        print(f"Unknown mode: {mode}")


if __name__ == "__main__":
    # Visualize first play
    visualize_predictions(play_idx=0, mode='single')
    
    # Visualize multiple plays
    visualize_predictions(mode='multiple')
    
    # Show error heatmap
    visualize_predictions(play_idx=0, mode='heatmap')