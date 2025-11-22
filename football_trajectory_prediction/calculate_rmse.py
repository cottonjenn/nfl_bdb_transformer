"""
Quick script to calculate summary statistics including RMSE from evaluation results.
If RMSE column doesn't exist, re-run evaluation with updated script.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

results_path = os.path.join('reports', 'evaluation_results.csv')

if not os.path.exists(results_path):
    print(f"Error: {results_path} not found!")
    sys.exit(1)

print("\n" + "=" * 80)
print("EVALUATION SUMMARY STATISTICS")
print("=" * 80)

df = pd.read_csv(results_path)

print(f"\nTotal plays evaluated: {len(df)}")
print(f"\nMetrics:")

if 'rmse' in df.columns:
    print(f"  RMSE: {df['rmse'].mean():.3f} ± {df['rmse'].std():.3f} yards")
    print(f"    Min: {df['rmse'].min():.3f} yards")
    print(f"    Max: {df['rmse'].max():.3f} yards")
    print(f"    Median: {df['rmse'].median():.3f} yards")
else:
    print("  ⚠ RMSE column not found in results.")
    print("    Re-run evaluation with: python3 main.py --evaluate")
    print("    The updated script now includes RMSE computation.")

if 'ade' in df.columns:
    print(f"\n  ADE: {df['ade'].mean():.3f} ± {df['ade'].std():.3f} yards")
    print(f"    Min: {df['ade'].min():.3f} yards")
    print(f"    Max: {df['ade'].max():.3f} yards")
    print(f"    Median: {df['ade'].median():.3f} yards")

if 'fde' in df.columns:
    print(f"\n  FDE: {df['fde'].mean():.3f} ± {df['fde'].std():.3f} yards")
    print(f"    Min: {df['fde'].min():.3f} yards")
    print(f"    Max: {df['fde'].max():.3f} yards")
    print(f"    Median: {df['fde'].median():.3f} yards")

print("\n" + "=" * 80)
print(f"\nFull results saved in: {results_path}")
print("=" * 80)

