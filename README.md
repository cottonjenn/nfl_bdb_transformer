# NFL Big Data Bowl 2025/2026 – Player Trajectory Prediction

In the NFL Big Data Bowl Prediction track, I built a custom PyTorch transformer model to forecast player positions frame-by-frame after the quarterback releases the ball, using pre-throw Next Gen Stats tracking data, targeted receiver info, and ball landing location.

## Project Overview
This repository contains my full end-to-end solution for the NFL Big Data Bowl Prediction competition (pass-air phase forecasting). Key components include:
- **Data preprocessing**: Direction flipping for field consistency, relative ball features (distance/angle), velocity decomposition, time-to-throw countdowns, and robust scaling/masking.
- **Model**: Encoder-decoder transformer with per-player encoding, cross-player attention for interactions, and a conditioned decoder using last-state summaries, ball landing embedding, and residual deltas for stable cumulative predictions.
- **Training & evaluation**: Masked multi-component loss (position + velocity + final-position), warmup-cosine scheduler, gradient clipping, and yard-based metrics (RMSE, ADE, FDE) with field visualizations.
- **Robustness**: NaN/Inf handling, forced unmasking for empty sequences, and small weight initialization.

The model excels at modeling multi-agent dynamics (e.g., coverage reactions, route stems) and provides insights useful for scouting, film study, or coverage analytics.

## Results
- Validation RMSE: ~0.85435
- Strong on short/medium throws; captures momentum and ball convergence well
- Visualizations show realistic paths (see `visualizations/` folder)

## Repository Structure

This repo is organized for easy navigation and reproducibility:

- **src/** — Core source code
  - `dataset.py` — Custom PyTorch Dataset with right-alignment, masking, and player mapping
  - `model.py` — Encoder-decoder transformer with cross-player attention and conditioned decoder
  - `loss.py` — Multi-component masked loss (position + velocity + final position)
  - `trainer.py` — Training loop with warmup-cosine scheduler, gradient clipping, checkpointing
  - `evaluator.py` — Yard-based metrics (RMSE, ADE, FDE) + denormalization
  - `visualizer.py` — Field plots, multi-play grids, error heatmaps

- **data/raw/** — Raw competition CSVs

- **models/** — Saved model checkpoints

- **reports/** — Evaluation results, trajectory visualizations, CSVs, plots

- **config.py** — Central configuration: hyperparameters, feature lists, paths, constants

- **main.py** — Entry point to run preprocessing, training, evaluation, or inference

- **diagnose_*.py / debug_*.py** — Helper scripts for data quality and debugging (e.g., player mapping mismatches, trajectory issues)

- **calculate_rmse.py** — Standalone RMSE calculator for quick checks

The typical workflow:
1. Run preprocessing → generates processed parquet files
2. Train via `main.py --train`
3. Evaluate/visualize via `main.py --eval` or `main.py --viz`

See the notebooks in the root or `src/` for exploration and inference examples.

## Links
- Kaggle competition page: [https://www.kaggle.com/competitions/nfl-big-data-bowl-2025](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction/overview)

Feel free to fork, star, or open issues/PRs — happy to discuss improvements or extensions (e.g., multi-modal outputs, physics constraints)!

Built with ❤️ using PyTorch, pandas, matplotlib.
