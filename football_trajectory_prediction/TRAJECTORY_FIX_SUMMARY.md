# Trajectory Prediction Fix Summary

## Problem Identified

The model was predicting **nearly identical trajectories for all players** instead of player-specific trajectories. This caused:
1. Jumbled visualizations with all players following similar paths
2. Large errors (ADE: ~7.3 yards) because predictions don't match individual player movements
3. The model learning a "generic" trajectory rather than player-specific patterns

## Root Cause

The model architecture was **losing player identity information** when processing the input:

1. **Encoder flattens all players together**: The model flattens all players and frames into a single sequence `[batch, num_players * num_frames, features]`, which removes information about which frames belong to which player.

2. **No player identity embeddings**: Without player position embeddings, the model couldn't distinguish between different players after flattening.

3. **Fixed-position decoder queries**: The decoder uses fixed-position queries (query[0], query[1], etc.) that don't correspond to actual player IDs, so different plays with different players at different positions would confuse the model.

## Solution Implemented

### Added Player Position Embeddings

Added learnable embeddings for each player position (0 to MAX_PLAYERS-1) that are added to the input embeddings before encoding. This allows the model to:

- Distinguish between different players even after flattening
- Learn player-specific patterns
- Make predictions that vary by player

**Code changes in `src/model.py`**:
- Added `self.player_embedding = nn.Embedding(config.MAX_PLAYERS, d_model)`
- Modified forward pass to add player position indices and embeddings to input embeddings

## Expected Impact

With this fix, the model should:
1. **Predict player-specific trajectories**: Each player should have distinct predicted trajectories
2. **Reduce errors**: ADE/FDE should decrease as predictions better match individual player movements
3. **Fix visualizations**: Trajectories should no longer be jumbled - each player's predicted path should follow their actual path

## Next Steps

1. **Retrain the model**: The model needs to be retrained with the new architecture
   ```bash
   python main.py --train
   ```

2. **Evaluate**: After training, evaluate the model to see if errors decrease
   ```bash
   python main.py --evaluate
   ```

3. **Visualize**: Check visualizations to confirm trajectories are no longer jumbled
   ```bash
   python main.py --visualize
   ```

## Additional Observations

The diagnostic script (`diagnose_trajectory_issue.py`) revealed:
- Model outputs predictions for all 22 players (MAX_PLAYERS), even when only 9 input players are present
- Predictions for different players were almost identical (e.g., all starting positions around x=-0.07, y=-1.67)
- The mapping from input to output indices is correct, but predictions themselves were generic

The player embeddings should fix this by allowing the model to learn distinct patterns for each player position.
