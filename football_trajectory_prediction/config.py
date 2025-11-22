"""
Configuration file for the football trajectory prediction project.
Modify these settings according to your data and requirements.
"""

import os
import glob

# ============================================================================
# PATHS
# ============================================================================

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATA FILE DISCOVERY
# ============================================================================

def discover_data_files(week_filter=None):
    """
    Automatically discover input and output CSV files in raw data directory.
    Matches input_*.csv with output_*.csv based on week identifiers.
    
    Args:
        week_filter: Optional list of week identifiers to include (e.g., ['2023_w01', '2023_w02'])
                    If None, includes all weeks.
    
    Returns:
        list of tuples: [(input_path, output_path, identifier), ...]
    """
    # Find all input files
    input_pattern = os.path.join(RAW_DATA_DIR, 'input_*.csv')
    input_files = sorted(glob.glob(input_pattern))
    
    # Find all output files
    output_pattern = os.path.join(RAW_DATA_DIR, 'output_*.csv')
    output_files = sorted(glob.glob(output_pattern))
    
    if not input_files:
        raise FileNotFoundError(f"No input files found matching: {input_pattern}")
    if not output_files:
        raise FileNotFoundError(f"No output files found matching: {output_pattern}")
    
    # Match input and output files
    file_pairs = []
    
    for input_file in input_files:
        # Extract identifier (e.g., "2023_w01" from "input_2023_w01.csv")
        input_basename = os.path.basename(input_file)
        identifier = input_basename.replace('input_', '').replace('.csv', '')
        
        # Filter by week if specified
        if week_filter is not None and identifier not in week_filter:
            continue
        
        # Find corresponding output file
        output_file = os.path.join(RAW_DATA_DIR, f'output_{identifier}.csv')
        
        if os.path.exists(output_file):
            file_pairs.append((input_file, output_file, identifier))
            print(f"  ✓ Matched: {os.path.basename(input_file)} <-> {os.path.basename(output_file)}")
        else:
            print(f"  ⚠ WARNING: No matching output file for {input_basename}")
    
    if not file_pairs:
        if week_filter:
            raise FileNotFoundError(f"No matching input-output file pairs found for weeks: {week_filter}")
        else:
            raise FileNotFoundError("No matching input-output file pairs found!")
    
    return file_pairs

# Data file pairs (populated when preprocessing runs)
DATA_FILE_PAIRS = []

# Processed data files
PROCESSED_INPUT = os.path.join(PROCESSED_DATA_DIR, 'input_sequences.parquet')
PROCESSED_OUTPUT = os.path.join(PROCESSED_DATA_DIR, 'output_sequences.parquet')
METADATA_FILE = os.path.join(PROCESSED_DATA_DIR, 'metadata.pkl')
SCALING_PARAMS = os.path.join(PROCESSED_DATA_DIR, 'scaling_params.pkl')

# Report files
DATA_REPORT = os.path.join(REPORTS_DIR, 'data_quality_report.html')
TRAINING_LOG = os.path.join(REPORTS_DIR, 'training_log.txt')

# ============================================================================
# DATA PARAMETERS
# ============================================================================

# Maximum sequence lengths
MAX_INPUT_FRAMES = 80   # Accommodate 8-second QB scrambles
MAX_OUTPUT_FRAMES = 25  # 2.5 seconds of prediction after throw
MAX_PLAYERS = 22        # 11 offense + 11 defense

# Plays to remove (known data issues)
PROBLEMATIC_PLAYS = [3167]  # Play with tracking errors

# Features to use for training
INPUT_FEATURES = [
    'x_norm', 'y_norm', 'x_rel_ball', 'y_rel_ball',
    'vx', 'vy', 'ax', 'ay', 's', 'a',
    'dist_to_target', 'angle_to_target',
    'dir_norm', 'o_norm',
    'time_to_throw', 'frame_progress'
]

OUTPUT_FEATURES = ['x_norm', 'y_norm']

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# Transformer architecture
D_MODEL = 128           # Embedding dimension
NHEAD = 8               # Number of attention heads
NUM_ENCODER_LAYERS = 6  # Encoder depth
NUM_DECODER_LAYERS = 6  # Decoder depth
DIM_FEEDFORWARD = 512   # FFN hidden dimension
DROPOUT = 0.1           # Dropout rate

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

BATCH_SIZE = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
TRAIN_SPLIT = 0.8       # 80% train, 20% validation
RANDOM_SEED = 42

# Early stopping
PATIENCE = 10          # Stop if no improvement for N epochs
MIN_DELTA = 0.0001      # Minimum change to qualify as improvement

# Device
DEVICE = 'cuda'         # 'cuda' or 'cpu'

# Checkpoint saving
SAVE_EVERY_N_EPOCHS = 5
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pth')
CHECKPOINT_PATH = os.path.join(MODELS_DIR, 'checkpoint.pth')