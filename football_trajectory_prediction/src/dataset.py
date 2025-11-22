"""
PyTorch Dataset for football trajectory sequences.
Handles variable-length sequences with padding, masking, and right-alignment.

Key assumptions for the new model:
- First 2 features in INPUT_FEATURES should be x, y coordinates (normalized)
- The model will use these for residual prediction
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import numpy as np
import sys
sys.path.append('..')
import config

try:
    from torch.utils.data._utils.collate import default_collate
except ImportError:
    from torch.utils.data.dataloader import default_collate


def custom_collate_fn(batch):
    """Custom collate that handles dicts with mixed types."""
    elem = batch[0]
    if isinstance(elem, dict):
        result = {}
        for key in elem.keys():
            if key in ('input_to_output_idx', 'metadata'):
                result[key] = [item[key] for item in batch]
            else:
                result[key] = default_collate([item[key] for item in batch])
        return result
    return default_collate(batch)


class FootballTrajectoryDataset(Dataset):
    """
    Dataset for football player trajectory prediction.
    
    Implements:
    1. Right-alignment: Recent frames aligned at the end
    2. Time features: Explicit time_to_throw encoding
    3. Masking: Ignore padded positions in attention
    
    IMPORTANT: Ensure INPUT_FEATURES has x, y as the first two features
    for the residual prediction to work correctly.
    """
    
    def __init__(self, input_path, output_path, metadata_path):
        print(f"Loading dataset from {input_path}")
        
        self.input_df = pd.read_parquet(input_path)
        self.output_df = pd.read_parquet(output_path)
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.plays = self.input_df[['game_id', 'play_id']].drop_duplicates().values
        
        # Validate that x, y are first two features
        if len(config.INPUT_FEATURES) >= 2:
            print(f"  First two input features: {config.INPUT_FEATURES[0]}, {config.INPUT_FEATURES[1]}")
            print(f"  (These should be x, y for residual prediction)")
        
        print(f"Loaded {len(self.plays)} plays")
        print(f"  Input features: {len(config.INPUT_FEATURES)}")
        print(f"  Output features: {len(config.OUTPUT_FEATURES)}")
        print(f"  Max input frames: {config.MAX_INPUT_FRAMES}")
        print(f"  Max output frames: {config.MAX_OUTPUT_FRAMES}")
        
        # Calculate and print statistics
        self._print_statistics()
        
    def _print_statistics(self):
        """Print dataset statistics for debugging."""
        print("\n  Dataset Statistics:")
        
        # Input sequence lengths
        input_lengths = self.input_df.groupby(['game_id', 'play_id'])['frame_id'].nunique()
        print(f"    Input frames - min: {input_lengths.min()}, max: {input_lengths.max()}, "
              f"mean: {input_lengths.mean():.1f}")
        
        # Output sequence lengths
        output_lengths = self.output_df.groupby(['game_id', 'play_id'])['frame_id'].nunique()
        print(f"    Output frames - min: {output_lengths.min()}, max: {output_lengths.max()}, "
              f"mean: {output_lengths.mean():.1f}")
        
        # Players per play
        input_players = self.input_df.groupby(['game_id', 'play_id'])['nfl_id'].nunique()
        print(f"    Players/play - min: {input_players.min()}, max: {input_players.max()}, "
              f"mean: {input_players.mean():.1f}")
        
        # Coordinate ranges (for checking normalization)
        for col in ['x', 'y']:
            if col in self.input_df.columns:
                print(f"    Input {col} range: [{self.input_df[col].min():.2f}, {self.input_df[col].max():.2f}]")
        
    def __len__(self):
        return len(self.plays)
    
    def __getitem__(self, idx):
        game_id, play_id = self.plays[idx]
        
        input_play = self.input_df[
            (self.input_df['game_id'] == game_id) & 
            (self.input_df['play_id'] == play_id)
        ].copy()
        
        output_play = self.output_df[
            (self.output_df['game_id'] == game_id) & 
            (self.output_df['play_id'] == play_id)
        ].copy()
        
        # Create tensors
        input_tensor, input_mask = self._create_input_tensor(input_play)
        output_tensor, output_mask = self._create_output_tensor(output_play)
        
        # Ball landing position
        ball_x = input_play['ball_land_x'].iloc[0]
        ball_y = input_play['ball_land_y'].iloc[0]
        ball_position = torch.tensor([ball_x, ball_y], dtype=torch.float32)
        
        # Player tracking
        input_players = sorted(input_play['nfl_id'].unique())
        output_players = sorted(output_play['nfl_id'].unique())
        output_player_set = set(output_players)
        
        # Mask for which input players should be predicted
        player_to_predict_mask = torch.zeros(config.MAX_PLAYERS, dtype=torch.bool)
        for p_idx, player_id in enumerate(input_players[:config.MAX_PLAYERS]):
            if player_id in output_player_set:
                player_to_predict_mask[p_idx] = True
        
        # Mapping from input index to output index
        output_player_to_idx = {pid: idx for idx, pid in enumerate(output_players)}
        input_to_output_idx = {}
        for in_idx, pid in enumerate(input_players[:config.MAX_PLAYERS]):
            if pid in output_player_to_idx:
                input_to_output_idx[in_idx] = output_player_to_idx[pid]
        
        metadata = {
            'game_id': game_id,
            'play_id': play_id,
            'num_input_frames': input_play['frame_id'].nunique(),
            'num_output_frames': output_play['frame_id'].nunique(),
            'num_input_players': len(input_players),
            'num_output_players': len(output_players),
            'input_player_ids': input_players[:config.MAX_PLAYERS],
            'output_player_ids': output_players
        }
        
        return {
            'input': input_tensor,
            'input_mask': input_mask,
            'output': output_tensor,
            'output_mask': output_mask,
            'ball_position': ball_position,
            'player_to_predict_mask': player_to_predict_mask,
            'input_to_output_idx': input_to_output_idx,
            'metadata': metadata
        }
    
    def _create_input_tensor(self, play_df):
        """Create RIGHT-ALIGNED input tensor with masking."""
        players = sorted(play_df['nfl_id'].unique())
        
        tensor = torch.zeros(
            config.MAX_PLAYERS,
            config.MAX_INPUT_FRAMES,
            len(config.INPUT_FEATURES),
            dtype=torch.float32
        )
        mask = torch.zeros(
            config.MAX_PLAYERS,
            config.MAX_INPUT_FRAMES,
            dtype=torch.bool
        )
        
        for p_idx, player_id in enumerate(players):
            if p_idx >= config.MAX_PLAYERS:
                break
                
            player_data = play_df[play_df['nfl_id'] == player_id].sort_values('frame_id')
            player_features = player_data[config.INPUT_FEATURES].values
            actual_frames = len(player_features)
            
            # Truncate if needed (keep most recent for right-alignment)
            if actual_frames > config.MAX_INPUT_FRAMES:
                player_features = player_features[-config.MAX_INPUT_FRAMES:]
                actual_frames = config.MAX_INPUT_FRAMES
            
            # Right-align: data at the END
            start_idx = config.MAX_INPUT_FRAMES - actual_frames
            player_tensor = torch.tensor(player_features, dtype=torch.float32)
            
            # Check for NaN/Inf in input data
            if torch.isnan(player_tensor).any() or torch.isinf(player_tensor).any():
                print(f"WARNING: NaN/Inf in input data for player {player_id}, play {play_df['play_id'].iloc[0] if 'play_id' in play_df.columns else 'unknown'}")
                player_tensor = torch.nan_to_num(player_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
            tensor[p_idx, start_idx:, :] = player_tensor
            mask[p_idx, start_idx:] = True
        
        return tensor, mask
    
    def _create_output_tensor(self, output_df):
        """Create LEFT-ALIGNED output tensor with masking."""
        players = sorted(output_df['nfl_id'].unique())
        
        tensor = torch.zeros(
            config.MAX_PLAYERS,
            config.MAX_OUTPUT_FRAMES,
            len(config.OUTPUT_FEATURES),
            dtype=torch.float32
        )
        mask = torch.zeros(
            config.MAX_PLAYERS,
            config.MAX_OUTPUT_FRAMES,
            dtype=torch.bool
        )
        
        for p_idx, player_id in enumerate(players):
            if p_idx >= config.MAX_PLAYERS:
                break
                
            player_data = output_df[output_df['nfl_id'] == player_id].sort_values('frame_id')
            player_coords = player_data[config.OUTPUT_FEATURES].values
            actual_frames = len(player_coords)
            
            if actual_frames > config.MAX_OUTPUT_FRAMES:
                player_coords = player_coords[:config.MAX_OUTPUT_FRAMES]
                actual_frames = config.MAX_OUTPUT_FRAMES
            
            # Left-align output
            player_tensor = torch.tensor(player_coords, dtype=torch.float32)
            
            # Check for NaN/Inf in output data
            if torch.isnan(player_tensor).any() or torch.isinf(player_tensor).any():
                print(f"WARNING: NaN/Inf in output data for player {player_id}, play {output_df['play_id'].iloc[0] if 'play_id' in output_df.columns else 'unknown'}")
                player_tensor = torch.nan_to_num(player_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            
            tensor[p_idx, :actual_frames, :] = player_tensor
            mask[p_idx, :actual_frames] = True
        
        return tensor, mask
    
    def get_play_by_ids(self, game_id, play_id):
        """Get a specific play by game_id and play_id (useful for inference)."""
        idx = None
        for i, (gid, pid) in enumerate(self.plays):
            if gid == game_id and pid == play_id:
                idx = i
                break
        if idx is None:
            raise ValueError(f"Play not found: game_id={game_id}, play_id={play_id}")
        return self[idx]