#!/usr/bin/env python3
"""
Diagnostic script to check for player_to_predict mismatch issues.
"""
import pandas as pd
import sys
sys.path.append('..')
import config

# Load processed data
print("Loading processed data...")
input_df = pd.read_parquet(config.PROCESSED_INPUT)
output_df = pd.read_parquet(config.PROCESSED_OUTPUT)

# Check a few plays
print("\n" + "="*80)
print("CHECKING PLAYER ALIGNMENT")
print("="*80)

# Get a sample play
sample_play = input_df[['game_id', 'play_id']].drop_duplicates().iloc[0]
game_id, play_id = sample_play['game_id'], sample_play['play_id']

print(f"\nSample play: game_id={game_id}, play_id={play_id}")

# Get input players
input_players = sorted(input_df[
    (input_df['game_id'] == game_id) & 
    (input_df['play_id'] == play_id)
]['nfl_id'].unique())

# Get output players  
output_players = sorted(output_df[
    (output_df['game_id'] == game_id) & 
    (output_df['play_id'] == play_id)
]['nfl_id'].unique())

print(f"\nInput players ({len(input_players)}): {input_players}")
print(f"Output players ({len(output_players)}): {output_players}")

# Check if output players are subset of input players
if set(output_players).issubset(set(input_players)):
    print("\n✓ Output players are a subset of input players")
else:
    print("\n⚠️  WARNING: Some output players are NOT in input players!")
    print(f"   Missing: {set(output_players) - set(input_players)}")

# Check player indices (how they're ordered in tensors)
print(f"\nPlayer index mapping:")
print("  Input tensor: player_idx -> nfl_id")
for idx, nfl_id in enumerate(input_players[:10]):  # Show first 10
    print(f"    [{idx}] -> {nfl_id}")
    if nfl_id in output_players:
        output_idx = output_players.index(nfl_id)
        print(f"         -> Output tensor: [{output_idx}]")
    else:
        print(f"         -> NOT in output")

# Check if player_to_predict exists in raw data
print("\n" + "="*80)
print("CHECKING RAW DATA FOR player_to_predict")
print("="*80)

# Try to load a raw file
import glob
raw_files = glob.glob(f"{config.RAW_DATA_DIR}/input_*.csv")
if raw_files:
    raw_df = pd.read_csv(raw_files[0], nrows=100)
    if 'player_to_predict' in raw_df.columns:
        print(f"\n✓ Found 'player_to_predict' column in raw data")
        print(f"  Sample values: {raw_df['player_to_predict'].unique()[:10]}")
        
        # Check if output players match player_to_predict
        raw_play = raw_df[
            (raw_df['game_id'] == game_id) & 
            (raw_df['play_id'] == play_id)
        ]
        if len(raw_play) > 0:
            players_to_predict = raw_play['player_to_predict'].unique()
            print(f"\n  For play {play_id}, players_to_predict: {players_to_predict}")
            print(f"  Output players: {output_players}")
            if set(players_to_predict) == set(output_players):
                print("  ✓ Output players match player_to_predict")
            else:
                print("  ⚠️  MISMATCH: Output players don't match player_to_predict!")
    else:
        print("\n⚠️  'player_to_predict' column NOT found in raw data")
else:
    print("\n⚠️  Could not find raw data files")

# Check model output shape vs target shape
print("\n" + "="*80)
print("MODEL OUTPUT vs TARGET SHAPE ANALYSIS")
print("="*80)
print(f"\nModel outputs: [batch, 1, max_frames, 2]")
print(f"  -> Single trajectory prediction (not per-player)")
print(f"\nTarget shape: [batch, {len(output_players)}, max_frames, 2]")
print(f"  -> {len(output_players)} player trajectories")
print(f"\n⚠️  ISSUE: Model predicts 1 trajectory, but target has {len(output_players)} players")
print(f"  -> Code expands prediction to all {len(output_players)} players")
print(f"  -> But which player does the model predict for?")
print(f"  -> And how do we map it to the correct output player?")

# Check evaluation code behavior
print("\n" + "="*80)
print("EVALUATION CODE BEHAVIOR")
print("="*80)
print("\nIn evaluate.py line 205-207:")
print("  if predictions.shape[1] == 1:")
print("      predictions = predictions.expand(-1, num_players, -1, -1)")
print("  -> This copies the SAME prediction to all output players")
print("  -> This is WRONG if we need to predict different players!")