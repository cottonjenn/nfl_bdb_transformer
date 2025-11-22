"""
Main entry point for the football trajectory prediction project.
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split

import config
from src.preprocess import FootballDataPreprocessor
from src.dataset import FootballTrajectoryDataset, custom_collate_fn
from src.model import TrajectoryTransformer
from src.train import Trainer
from src.evaluate import evaluate_model
from src.visualize import visualize_predictions


def preprocess(week_filter=None):
    """Run data preprocessing pipeline.
    
    Args:
        week_filter: Optional list of week identifiers to process (e.g., ['2023_w01'])
    """
    preprocessor = FootballDataPreprocessor(week_filter=week_filter)
    preprocessor.run()


def train():
    """Train the model."""
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # Load dataset
    dataset = FootballTrajectoryDataset(
        config.PROCESSED_INPUT,
        config.PROCESSED_OUTPUT,
        config.METADATA_FILE
    )
    
    # Split into train/val
    train_size = int(config.TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    
    torch.manual_seed(config.RANDOM_SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"\nTrain set: {len(train_dataset)} plays")
    print(f"Val set: {len(val_dataset)} plays")
    
    # Create data loaders with custom collate function to handle dictionaries
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        collate_fn=custom_collate_fn
    )
    
    # Create model
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    
    model = TrajectoryTransformer()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Set device
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    if config.DEVICE == 'cuda' and not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
    
    # Create trainer and train
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.train(config.NUM_EPOCHS)


def evaluate():
    """Evaluate the trained model."""
    evaluate_model()


def visualize():
    """Visualize predictions."""
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # Single play
    print("\n1. Visualizing single play...")
    visualize_predictions(play_idx=0, mode='single')
    
    # Multiple plays
    print("\n2. Visualizing multiple plays...")
    visualize_predictions(mode='multiple')
    
    # Error heatmap
    print("\n3. Creating error heatmap...")
    visualize_predictions(play_idx=0, mode='heatmap')
    
    print("\n✓ All visualizations saved to:", config.REPORTS_DIR)


def main():
    parser = argparse.ArgumentParser(
        description='Football Trajectory Prediction with Transformers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --preprocess                    Run data preprocessing only
  python main.py --train                         Train the model only
  python main.py --evaluate                      Evaluate trained model
  python main.py --visualize                     Generate visualizations
  python main.py --all                           Run full pipeline (preprocess + train + evaluate + visualize)
  python main.py --preprocess --weeks 2023_w01    Preprocess only week 1 (for testing)
  python main.py --all --weeks 2023_w01           Run full pipeline on single week (for testing)
        """
    )
    
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Run data preprocessing'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train the model'
    )
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate the trained model'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualizations'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run complete pipeline (preprocess + train + evaluate + visualize)'
    )
    parser.add_argument(
        '--weeks',
        type=str,
        nargs='+',
        help='Filter to specific weeks (e.g., --weeks 2023_w01 2023_w02). Use for testing on single week.'
    )
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Get week filter if specified
    week_filter = args.weeks if args.weeks else None
    if week_filter:
        print(f"\n⚠️  WEEK FILTER ACTIVE: Processing only weeks: {week_filter}")
        print("   This is useful for testing on a single week before full training.\n")
    
    # Run requested operations
    if args.all:
        print("\n" + "=" * 80)
        print("RUNNING COMPLETE PIPELINE")
        print("=" * 80)
        preprocess(week_filter=week_filter)
        train()
        evaluate()
        visualize()
    else:
        if args.preprocess:
            preprocess(week_filter=week_filter)
        if args.train:
            train()
        if args.evaluate:
            evaluate()
        if args.visualize:
            visualize()


if __name__ == "__main__":
    main()