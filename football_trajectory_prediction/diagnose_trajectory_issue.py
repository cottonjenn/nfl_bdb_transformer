#!/usr/bin/env python3
"""
Diagnostic script to understand trajectory prediction issues.
Checks if the model is confusing players across plays or incorrectly mapping predictions.
"""

import torch
import pandas as pd
import numpy as np
import sys
sys.path.append('.')
import config
from src.dataset import FootballTrajectoryDataset
from src.model import TrajectoryTransformer

print("=" * 80)
print("DIAGNOSING TRAJECTORY PREDICTION ISSUES")
print("=" * 80)

# Load dataset
print("\nLoading dataset...")
dataset = FootballTrajectoryDataset(
    config.PROCESSED_INPUT,
    config.PROCESSED_OUTPUT,
    config.METADATA_FILE
)

# Load model
print("Loading model...")
checkpoint = torch.load(config.BEST_MODEL_PATH, map_location='cpu')
model = TrajectoryTransformer()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

device = torch.device('cpu')
model.to(device)

# Check a few plays
print("\n" + "=" * 80)
print("ANALYZING PLAYER MAPPING AND PREDICTIONS")
print("=" * 80)

for play_idx in range(min(3, len(dataset))):
    print(f"\n{'='*60}")
    print(f"PLAY {play_idx}")
    print(f"{'='*60}")
    
    data = dataset[play_idx]
    metadata = data['metadata']
    
    print(f"\nPlay ID: game_id={metadata['game_id']}, play_id={metadata['play_id']}")
    print(f"Input players: {metadata['num_input_players']}")
    print(f"Output players: {metadata['num_output_players']}")
    print(f"Input frames: {metadata['num_input_frames']}")
    print(f"Output frames: {metadata['num_output_frames']}")
    
    # Get player IDs
    input_player_ids = metadata.get('input_player_ids', [])
    output_player_ids = metadata.get('output_player_ids', [])
    
    print(f"\nInput player IDs: {input_player_ids[:10]}...")
    print(f"Output player IDs: {output_player_ids}")
    
    # Get input_to_output mapping
    input_to_output_idx = data.get('input_to_output_idx', {})
    print(f"\nInput -> Output index mapping:")
    for input_idx, output_idx in list(input_to_output_idx.items())[:5]:
        input_player_id = input_player_ids[input_idx] if input_idx < len(input_player_ids) else "N/A"
        output_player_id = output_player_ids[output_idx] if output_idx < len(output_player_ids) else "N/A"
        print(f"  Input[{input_idx}] (Player {input_player_id}) -> Output[{output_idx}] (Player {output_player_id})")
    
    # Get model prediction
    inputs = data['input'].unsqueeze(0).to(device)
    input_mask = data['input_mask'].unsqueeze(0).to(device)
    targets = data['output'].unsqueeze(0).to(device)
    output_mask = data['output_mask'].unsqueeze(0).to(device)
    ball_pos = data['ball_position'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(inputs, input_mask, ball_pos)
    
    print(f"\nModel output shape: {predictions.shape}")
    print(f"Target shape: {targets.shape}")
    
    # Check if predictions are all zeros or very similar
    pred_std = predictions.std().item()
    print(f"\nPrediction statistics:")
    print(f"  Std dev: {pred_std:.4f}")
    print(f"  Mean: {predictions.mean().item():.4f}")
    print(f"  Min: {predictions.min().item():.4f}")
    print(f"  Max: {predictions.max().item():.4f}")
    
    # Check if predictions are different per player
    player_means = predictions[0].mean(dim=1).mean(dim=0)  # [num_players, 2] -> [2] overall mean
    player_means_per_player = predictions[0].mean(dim=1)  # [num_players, 2]
    print(f"\nPer-player prediction means (should be different):")
    for p_idx in range(min(10, predictions.shape[1])):
        mean_pos = player_means_per_player[p_idx].tolist()
        print(f"  Player[{p_idx}]: mean_pos={mean_pos}")
    
    # Check which input players have valid data
    valid_input_players = input_mask[0].any(dim=1).nonzero(as_tuple=True)[0]
    print(f"\nValid input player indices (have data): {valid_input_players.tolist()}")
    print(f"Total input players: {predictions.shape[1]} (includes padding)")
    
    # Check actual vs predicted positions (after mapping)
    batch_size, num_target_players, max_frames, coords = targets.shape
    predictions_mapped = torch.zeros(
        1, num_target_players, max_frames, coords,
        device=predictions.device,
        dtype=predictions.dtype
    )
    
    if input_to_output_idx:
        for input_idx, output_idx in input_to_output_idx.items():
            if input_idx < predictions.shape[1] and output_idx < num_target_players:
                predictions_mapped[0, output_idx, :max_frames, :] = predictions[0, input_idx, :max_frames, :]
    
    # Check if mapped predictions match targets
    valid_mask = output_mask[0].float()  # [num_players, max_frames]
    errors = ((predictions_mapped[0] - targets[0]) ** 2).sum(dim=-1).sqrt()  # [num_players, max_frames]
    masked_errors = errors * valid_mask
    per_player_errors = masked_errors.sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)
    
    print(f"\nPer-output-player prediction errors (after mapping):")
    for p_idx in range(min(5, num_target_players)):
        if output_mask[0, p_idx].any():
            print(f"  Output Player[{p_idx}] (Player ID {output_player_ids[p_idx]}): "
                  f"avg_error={per_player_errors[p_idx].item():.2f} yards")
    
    # Check if predictions are collapsing (all similar)
    if pred_std < 0.1:
        print(f"\n⚠️  WARNING: Predictions have very low variance ({pred_std:.4f})")
        print(f"   This suggests the model might be predicting the same thing for all players!")
    
    # Check if player predictions are being swapped
    print(f"\nChecking for potential player confusion...")
    if len(input_to_output_idx) > 1:
        # Get starting positions from targets
        first_frames = []
        for p_idx in range(num_target_players):
            valid_frames = output_mask[0, p_idx].nonzero(as_tuple=True)[0]
            if len(valid_frames) > 0:
                first_pos = targets[0, p_idx, valid_frames[0]]
                first_frames.append((p_idx, first_pos))
        
        # Get starting positions from predictions (mapped)
        pred_first_frames = []
        for p_idx in range(num_target_players):
            valid_frames = output_mask[0, p_idx].nonzero(as_tuple=True)[0]
            if len(valid_frames) > 0:
                first_pos = predictions_mapped[0, p_idx, valid_frames[0]]
                pred_first_frames.append((p_idx, first_pos))
        
        # Check if positions are reasonable
        print(f"  Target starting positions (first 3 players):")
        for p_idx, pos in first_frames[:3]:
            print(f"    Player {output_player_ids[p_idx]}: x={pos[0].item():.2f}, y={pos[1].item():.2f}")
        print(f"  Predicted starting positions (first 3 players):")
        for p_idx, pos in pred_first_frames[:3]:
            print(f"    Player {output_player_ids[p_idx]}: x={pos[0].item():.2f}, y={pos[1].item():.2f}")

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
