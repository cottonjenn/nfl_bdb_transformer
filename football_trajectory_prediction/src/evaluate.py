"""
Evaluation metrics for trajectory prediction.
Computes ADE, FDE, and other common trajectory forecasting metrics.
"""

import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import sys
sys.path.append('..')
import config
from src.dataset import FootballTrajectoryDataset
from src.model import TrajectoryTransformer


class TrajectoryEvaluator:
    """Evaluates trajectory prediction performance."""
    
    def __init__(self, model, dataset, device, scaling_params_path):
        """
        Args:
            model: Trained transformer model
            dataset: FootballTrajectoryDataset instance
            device: 'cuda' or 'cpu'
            scaling_params_path: Path to scaling parameters for denormalization
        """
        self.model = model.to(device)
        self.model.eval()
        self.dataset = dataset
        self.device = device
        
        # Load scaling parameters for denormalization
        with open(scaling_params_path, 'rb') as f:
            self.scaling_params = pickle.load(f)
        
        self.results = []
        
    def denormalize_coordinates(self, coords_normalized):
        """
        Convert normalized coordinates back to yards.
        
        Args:
            coords_normalized: [batch, players, frames, 2] normalized x, y
            
        Returns:
            coords_yards: [batch, players, frames, 2] in yards
        """
        x_mean = self.scaling_params['x_norm']['mean']
        x_std = self.scaling_params['x_norm']['std']
        y_mean = self.scaling_params['y_norm']['mean']
        y_std = self.scaling_params['y_norm']['std']
        
        coords_yards = coords_normalized.clone()
        coords_yards[..., 0] = coords_normalized[..., 0] * x_std + x_mean
        coords_yards[..., 1] = coords_normalized[..., 1] * y_std + y_mean
        
        return coords_yards
    
    def compute_ade(self, predictions, targets, mask):
        """
        Compute Average Displacement Error.
        
        ADE = Average Euclidean distance between predicted and actual positions
        across all time steps.
        
        Args:
            predictions: [batch, players, frames, 2]
            targets: [batch, players, frames, 2]
            mask: [batch, players, frames] (True = valid)
            
        Returns:
            ade: scalar value in yards
        """
        # Denormalize to yards
        pred_yards = self.denormalize_coordinates(predictions)
        target_yards = self.denormalize_coordinates(targets)
        
        # Euclidean distance per frame
        diff = pred_yards - target_yards
        distances = torch.sqrt((diff ** 2).sum(dim=-1))  # [batch, players, frames]
        
        # Apply mask and average
        masked_distances = distances * mask.float()
        total_distance = masked_distances.sum()
        num_valid = mask.sum()
        
        ade = total_distance / num_valid if num_valid > 0 else torch.tensor(0.0)
        return ade.item()
    
    def compute_fde(self, predictions, targets, mask):
        """
        Compute Final Displacement Error.
        
        FDE = Euclidean distance between predicted and actual final positions.
        
        Args:
            predictions: [batch, players, frames, 2]
            targets: [batch, players, frames, 2]
            mask: [batch, players, frames] (True = valid)
            
        Returns:
            fde: scalar value in yards
        """
        # Denormalize to yards
        pred_yards = self.denormalize_coordinates(predictions)
        target_yards = self.denormalize_coordinates(targets)
        
        batch_size, num_players, num_frames, _ = predictions.shape
        
        final_distances = []
        
        for b in range(batch_size):
            for p in range(num_players):
                # Find last valid frame for this player
                valid_frames = mask[b, p].nonzero(as_tuple=True)[0]
                
                if len(valid_frames) > 0:
                    last_frame = valid_frames[-1]
                    
                    # Distance at final frame
                    pred_final = pred_yards[b, p, last_frame]
                    target_final = target_yards[b, p, last_frame]
                    
                    distance = torch.sqrt(((pred_final - target_final) ** 2).sum())
                    final_distances.append(distance.item())
        
        fde = np.mean(final_distances) if final_distances else 0.0
        return fde
    
    def compute_rmse(self, predictions, targets, mask):
        """
        Compute Root Mean Squared Error.
        
        RMSE = sqrt(mean(squared Euclidean distances)) between predicted and actual positions
        across all time steps.
        
        Args:
            predictions: [batch, players, frames, 2]
            targets: [batch, players, frames, 2]
            mask: [batch, players, frames] (True = valid)
            
        Returns:
            rmse: scalar value in yards
        """
        # Denormalize to yards
        pred_yards = self.denormalize_coordinates(predictions)
        target_yards = self.denormalize_coordinates(targets)
        
        # Squared Euclidean distance per frame
        diff = pred_yards - target_yards
        squared_distances = (diff ** 2).sum(dim=-1)  # [batch, players, frames]
        
        # Apply mask and compute mean of squared distances
        masked_squared_distances = squared_distances * mask.float()
        total_squared_distance = masked_squared_distances.sum()
        num_valid = mask.sum()
        
        # Mean squared error, then take square root
        mse = total_squared_distance / num_valid if num_valid > 0 else torch.tensor(0.0)
        rmse = torch.sqrt(mse)
        
        return rmse.item()
    
    def compute_frame_wise_errors(self, predictions, targets, mask):
        """
        Compute error at each frame.
        
        Args:
            predictions: [batch, players, frames, 2]
            targets: [batch, players, frames, 2]
            mask: [batch, players, frames] (True = valid)
            
        Returns:
            frame_errors: dict mapping frame_idx -> average error in yards
        """
        pred_yards = self.denormalize_coordinates(predictions)
        target_yards = self.denormalize_coordinates(targets)
        
        diff = pred_yards - target_yards
        distances = torch.sqrt((diff ** 2).sum(dim=-1))  # [batch, players, frames]
        
        num_frames = distances.shape[2]
        frame_errors = {}
        
        for frame in range(num_frames):
            frame_mask = mask[:, :, frame]
            frame_distances = distances[:, :, frame]
            
            masked_distances = frame_distances * frame_mask.float()
            num_valid = frame_mask.sum()
            
            if num_valid > 0:
                avg_error = (masked_distances.sum() / num_valid).item()
                frame_errors[frame] = avg_error
        
        return frame_errors
    
    def evaluate_dataset(self, num_samples=None):
        """
        Evaluate model on entire dataset or subset.
        
        Args:
            num_samples: Number of plays to evaluate (None = all)
            
        Returns:
            results: dict with evaluation metrics
        """
        print("\n" + "=" * 80)
        print("EVALUATING MODEL")
        print("=" * 80)
        
        num_samples = num_samples or len(self.dataset)
        print(f"Evaluating on {num_samples} plays...")
        
        all_ade = []
        all_fde = []
        all_rmse = []
        all_frame_errors = {}
        
        with torch.no_grad():
            for idx in range(min(num_samples, len(self.dataset))):
                # Get data
                data = self.dataset[idx]
                
                # Add batch dimension
                inputs = data['input'].unsqueeze(0).to(self.device)
                input_mask = data['input_mask'].unsqueeze(0).to(self.device)
                targets = data['output'].unsqueeze(0).to(self.device)
                output_mask = data['output_mask'].unsqueeze(0).to(self.device)
                ball_pos = data['ball_position'].unsqueeze(0).to(self.device)
                
                # Predict
                predictions = self.model(inputs, input_mask, ball_pos)
                
                # Adjust shapes to match
                # Model now outputs [batch, num_input_players, max_frames, 2]
                # We need to map predictions from input player indices to output player indices
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
                
                # Compute metrics
                ade = self.compute_ade(predictions, targets, output_mask)
                fde = self.compute_fde(predictions, targets, output_mask)
                rmse = self.compute_rmse(predictions, targets, output_mask)
                frame_errors = self.compute_frame_wise_errors(predictions, targets, output_mask)
                
                all_ade.append(ade)
                all_fde.append(fde)
                all_rmse.append(rmse)
                
                # Accumulate frame errors
                for frame, error in frame_errors.items():
                    if frame not in all_frame_errors:
                        all_frame_errors[frame] = []
                    all_frame_errors[frame].append(error)
                
                # Store per-play results
                self.results.append({
                    'game_id': data['metadata']['game_id'],
                    'play_id': data['metadata']['play_id'],
                    'ade': ade,
                    'fde': fde,
                    'rmse': rmse,
                    'num_frames': data['metadata']['num_output_frames'],
                    'num_players': data['metadata']['num_output_players']
                })
                
                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx + 1}/{num_samples} plays...")
        
        # Average frame-wise errors
        avg_frame_errors = {
            frame: np.mean(errors) 
            for frame, errors in all_frame_errors.items()
        }
        
        results = {
            'ade': np.mean(all_ade),
            'fde': np.mean(all_fde),
            'rmse': np.mean(all_rmse),
            'ade_std': np.std(all_ade),
            'fde_std': np.std(all_fde),
            'rmse_std': np.std(all_rmse),
            'frame_errors': avg_frame_errors,
            'per_play_results': self.results
        }
        
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Average Displacement Error (ADE): {results['ade']:.3f} ± {results['ade_std']:.3f} yards")
        print(f"Final Displacement Error (FDE):   {results['fde']:.3f} ± {results['fde_std']:.3f} yards")
        print(f"Root Mean Squared Error (RMSE):   {results['rmse']:.3f} ± {results['rmse_std']:.3f} yards")
        print()
        print("Frame-wise errors (yards):")
        for frame in sorted(avg_frame_errors.keys())[:10]:  # Show first 10 frames
            print(f"  Frame {frame+1:2d}: {avg_frame_errors[frame]:.3f}")
        if len(avg_frame_errors) > 10:
            print(f"  ... ({len(avg_frame_errors) - 10} more frames)")
        
        return results
    
    def save_results(self, output_path):
        """Save evaluation results to CSV."""
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")


def evaluate_model(model_path=None, num_samples=None):
    """
    Convenience function to evaluate a trained model.
    
    Args:
        model_path: Path to model checkpoint (default: best model)
        num_samples: Number of plays to evaluate (None = all)
    """
    model_path = model_path or config.BEST_MODEL_PATH
    
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    model = TrajectoryTransformer()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load dataset
    dataset = FootballTrajectoryDataset(
        config.PROCESSED_INPUT,
        config.PROCESSED_OUTPUT,
        config.METADATA_FILE
    )
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    
    # Evaluate
    evaluator = TrajectoryEvaluator(
        model, 
        dataset, 
        device,
        config.SCALING_PARAMS
    )
    
    results = evaluator.evaluate_dataset(num_samples)
    
    # Save results
    output_path = config.REPORTS_DIR + '/evaluation_results.csv'
    evaluator.save_results(output_path)
    
    return results


if __name__ == "__main__":
    evaluate_model()