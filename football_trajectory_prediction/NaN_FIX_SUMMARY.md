# NaN Issue Fix Summary

## Problem
The model was producing NaN values in the encoder, which propagated through the decoder and output layers. This caused the model to essentially "guess" (predict the first point for every frame) rather than learn meaningful trajectories.

## Root Causes Identified

1. **All-masked sequences**: When a player has no valid frames (all positions masked), PyTorch's TransformerEncoder can produce NaN values due to numerical instability in the attention mechanism.

2. **Input data issues**: Potential NaN/Inf values in the input data that weren't being caught early.

3. **Numerical instability**: Large initial values or improper weight initialization could cause overflow/underflow.

## Fixes Applied

### 1. Encoder Improvements (`src/model.py`)
- **All-masked sequence handling**: Ensures at least one position is always unmasked per sequence to prevent NaN in attention
- **Input validation**: Checks for NaN/Inf at multiple stages (input, after projection, after positional encoding)
- **Weight initialization**: Proper initialization of input projection with small gain (0.1) to prevent large initial outputs
- **Error handling**: Try-catch block around encoder forward pass with detailed error messages
- **Assertion check**: Verifies no sequences are completely masked before encoding

### 2. Decoder Improvements (`src/model.py`)
- **NaN detection**: Checks for NaN/Inf after decoding and in offset predictions
- **Offset clamping**: Clamps predicted offsets to [-5.0, 5.0] to prevent explosion in cumsum
- **Output projection initialization**: Small weight initialization for output layers

### 3. Loss Function Improvements (`src/train.py`)
- **Input validation**: Checks for NaN/Inf in predictions and targets before loss computation
- **Edge case handling**: Handles cases where no valid frames exist
- **Robust division**: Uses epsilon values to prevent division by zero

### 4. Dataset Improvements (`src/dataset.py`)
- **Data validation**: Checks for NaN/Inf values when loading player data
- **Early detection**: Catches data quality issues at the source

### 5. Training Loop Improvements (`src/train.py`)
- **Comprehensive checks**: Validates inputs, predictions, and loss values
- **Detailed error messages**: Provides context when NaN is detected
- **Reduced verbosity**: Warning messages only print first few times and then periodically

## Testing Recommendations

1. **Run training again** and observe:
   - Whether NaN warnings still appear (they should be much less frequent)
   - If they do appear, the error messages will indicate exactly where
   - The model should now train properly even if some NaN values occur (they'll be replaced with zeros)

2. **Check data quality**:
   - Run the preprocessing validation to ensure no NaN/Inf in source data
   - Verify that all players have at least some valid frames

3. **Monitor training**:
   - Loss should decrease properly (not stay constant)
   - Predictions should vary across frames (not just repeat the first point)
   - Validation loss should track training loss

## If NaN Still Occurs

If NaN warnings still appear frequently:

1. **Check the error messages** - they will indicate the exact location
2. **Reduce learning rate** - try 1e-5 or 5e-5 instead of 1e-4
3. **Reduce model size** - fewer layers or smaller d_model
4. **Check input data ranges** - ensure features are properly normalized
5. **Add gradient clipping** - already present but could reduce max_norm to 0.5

## Key Changes Made

- `src/model.py`: Added comprehensive NaN handling in encoder and decoder
- `src/train.py`: Added input/output validation and improved loss function
- `src/dataset.py`: Added data quality checks

The model should now handle edge cases gracefully and prevent NaN propagation while still allowing training to continue.

