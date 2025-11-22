"""
Debug script to check player ordering mismatch between input and output.
"""

import torch
import pandas as pd
import pickle
import sys
import os

sys.path.append('.')
import config
from src.dataset import FootballTrajectoryDataset

print("=" * 80)
print("DEBUGGING PLAYER ORDERING")
print("=" * 80)

# Load dataset
dataset = FootballTrajectoryDataset(
    config.PROCESSED_INPUT,
    config.PROCESSED_OUTPUT,
    config.METADATA_FILE
)

# Check first play
play_idx = 0
data = dataset[play_idx]

print(f"\nPlay: game_id={data['metadata']['game_id']}, play_id={data['metadata']['play_id']}")
print(f"Input players: {data['metadata']['num_input_players']}")
print(f"Output players: {data['metadata']['num_output_players']}")

# Load raw data to get actual player IDs
input_play = pd.read_parquet(config.PROCESSED_INPUT)
input_play = input_play[
    (input_play['game_id'] == data['metadata']['game_id']) & 
    (input_play['play_id'] == data['metadata']['play_id'])
]

output_play = pd.read_parquet(config.PROCESSED_OUTPUT)
output_play = output_play[
    (output_play['game_id'] == data['metadata']['game_id']) & 
    (output_play['play_id'] == data['metadata']['play_id'])
]

# Get player IDs
input_players = sorted(input_play['nfl_id'].unique())
output_players = sorted(output_play['nfl_id'].unique())

print(f"\nInput player IDs (sorted): {input_players[:10]}...")  # First 10
print(f"Output player IDs (sorted): {output_players}")

# Check player_to_predict_mask
player_mask = data['player_to_predict_mask']
masked_indices = player_mask.nonzero(as_tuple=True)[0].tolist()

print(f"\nplayer_to_predict_mask indices (True values): {masked_indices}")
print(f"These correspond to input player indices in sorted order:")

for idx in masked_indices[:5]:  # First 5
    if idx < len(input_players):
        print(f"  Input index {idx} -> Player ID {input_players[idx]}")

print(f"\nOutput tensor has {data['output'].shape[0]} players (indices 0 to {data['output'].shape[0]-1})")
print(f"These correspond to output player IDs in sorted order:")

for idx in range(min(5, len(output_players))):
    print(f"  Output index {idx} -> Player ID {output_players[idx]}")

# Check if there's a mismatch
print("\n" + "=" * 80)
print("MAPPING ANALYSIS")
print("=" * 80)

# Get the player IDs that should be predicted (from input indices)
predicted_player_ids_from_input = [input_players[idx] for idx in masked_indices if idx < len(input_players)]

print(f"\nPlayer IDs we're extracting from model predictions (from input indices):")
print(f"  {predicted_player_ids_from_input}")

print(f"\nPlayer IDs in output tensor (in output index order):")
print(f"  {output_players}")

# Check if they match
if set(predicted_player_ids_from_input) == set(output_players):
    print("\n✓ Player IDs match! But need to check ORDER...")
    
    # Check order
    if predicted_player_ids_from_input == output_players:
        print("✓ ORDER also matches!")
    else:
        print("⚠️  MISMATCH: Same players, but DIFFERENT ORDER!")
        print("   This means predictions are being assigned to wrong players!")
        
        print("\nCorrect mapping should be:")
        for input_idx, player_id in enumerate(predicted_player_ids_from_input):
            if player_id in output_players:
                output_idx = output_players.index(player_id)
                print(f"  Input index {input_idx} (Player {player_id}) -> Output index {output_idx}")
else:
    print("\n⚠️  MISMATCH: Different sets of players!")
    print(f"   Missing from output: {set(predicted_player_ids_from_input) - set(output_players)}")
    print(f"   Extra in output: {set(output_players) - set(predicted_player_ids_from_input)}")

