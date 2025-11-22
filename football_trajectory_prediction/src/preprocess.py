"""
Data preprocessing pipeline for football tracking data.
Handles cleaning, feature engineering, normalization, and validation.
Supports multiple week files automatically.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
import config


class FootballDataPreprocessor:
    """Preprocesses raw football tracking data for transformer training."""
    
    def __init__(self, week_filter=None):
        """
        Args:
            week_filter: Optional list of week identifiers to process (e.g., ['2023_w01'])
        """
        self.week_filter = week_filter
        self.scaling_params = {}
        self.metadata = {}
        self.input_df = None
        self.output_df = None
    
    def load_data(self):
        """Load raw CSV files from all weeks (or filtered weeks)."""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        if self.week_filter:
            print(f"\n⚠️  WEEK FILTER: Processing only weeks: {self.week_filter}")
        
        # Discover all input/output file pairs
        print("\nDiscovering data files...")
        config.DATA_FILE_PAIRS = config.discover_data_files(week_filter=self.week_filter)
        
        print(f"\nFound {len(config.DATA_FILE_PAIRS)} file pairs")
        
        # Load all files
        input_dfs = []
        output_dfs = []
        
        for input_file, output_file, identifier in config.DATA_FILE_PAIRS:
            print(f"\nLoading {identifier}...")
            
            # Load input
            input_df = pd.read_csv(input_file)
            print(f"  Input:  {len(input_df):,} rows, {input_df['play_id'].nunique()} plays")
            input_dfs.append(input_df)
            
            # Load output
            output_df = pd.read_csv(output_file)
            print(f"  Output: {len(output_df):,} rows, {output_df['play_id'].nunique()} plays")
            output_dfs.append(output_df)
        
        # Combine all weeks
        print("\nCombining all weeks...")
        self.input_df = pd.concat(input_dfs, ignore_index=True)
        self.output_df = pd.concat(output_dfs, ignore_index=True)
        
        print(f"\nTotal combined data:")
        print(f"  Input:  {len(self.input_df):,} rows")
        print(f"  Output: {len(self.output_df):,} rows")
        print(f"  Unique plays: {self.input_df['play_id'].nunique()}")
        
        # Store week info in metadata
        self.metadata['weeks_processed'] = [ident for _, _, ident in config.DATA_FILE_PAIRS]
        
    def explore_data(self):
        """Analyze data distributions and identify issues."""
        print("\n" + "=" * 80)
        print("DATA EXPLORATION")
        print("=" * 80)
        
        # Input frame distribution
        input_frames = self.input_df.groupby(['game_id', 'play_id'])['frame_id'].nunique()
        print("\n--- Input Frames per Play ---")
        print(input_frames.describe())
        print(f"Max: {input_frames.max()} frames")
        print(f"Plays > {config.MAX_INPUT_FRAMES} frames: {(input_frames > config.MAX_INPUT_FRAMES).sum()}")
        
        # Output frame distribution
        output_frames = self.output_df.groupby(['game_id', 'play_id'])['frame_id'].nunique()
        print("\n--- Output Frames per Play ---")
        print(output_frames.describe())
        print(f"Max: {output_frames.max()} frames")
        print(f"Plays > {config.MAX_OUTPUT_FRAMES} frames: {(output_frames > config.MAX_OUTPUT_FRAMES).sum()}")
        
        # Player count distribution
        player_counts = self.input_df.groupby(['game_id', 'play_id'])['nfl_id'].nunique()
        print("\n--- Players per Play ---")
        print(player_counts.describe())
        
        # Check for missing values
        print("\n--- Missing Values ---")
        print("Input data:")
        missing_input = self.input_df.isnull().sum()[self.input_df.isnull().sum() > 0]
        if len(missing_input) > 0:
            print(missing_input)
        else:
            print("None")
            
        print("\nOutput data:")
        missing_output = self.output_df.isnull().sum()[self.output_df.isnull().sum() > 0]
        if len(missing_output) > 0:
            print(missing_output)
        else:
            print("None")
        
        # Store for report
        self.metadata['exploration'] = {
            'input_frames': input_frames.to_dict(),
            'output_frames': output_frames.to_dict(),
            'player_counts': player_counts.to_dict()
        }
        
    def clean_data(self):
        """Remove problematic plays and handle missing values."""
        print("\n" + "=" * 80)
        print("DATA CLEANING")
        print("=" * 80)
        
        initial_plays = self.input_df['play_id'].nunique()
        
        # Remove problematic plays
        if config.PROBLEMATIC_PLAYS:
            print(f"\nRemoving {len(config.PROBLEMATIC_PLAYS)} problematic plays: {config.PROBLEMATIC_PLAYS}")
            self.input_df = self.input_df[~self.input_df['play_id'].isin(config.PROBLEMATIC_PLAYS)]
            self.output_df = self.output_df[~self.output_df['play_id'].isin(config.PROBLEMATIC_PLAYS)]
        
        # Remove plays with excessive output frames
        output_frame_counts = self.output_df.groupby(['game_id', 'play_id'])['frame_id'].nunique()
        long_plays = output_frame_counts[output_frame_counts > config.MAX_OUTPUT_FRAMES * 1.5].index
        if len(long_plays) > 0:
            print(f"\nRemoving {len(long_plays)} plays with excessive output frames (>{config.MAX_OUTPUT_FRAMES * 1.5})")
            self.input_df = self.input_df[~self.input_df.set_index(['game_id', 'play_id']).index.isin(long_plays)]
            self.output_df = self.output_df[~self.output_df.set_index(['game_id', 'play_id']).index.isin(long_plays)]
        
        final_plays = self.input_df['play_id'].nunique()
        print(f"\nPlays remaining: {final_plays} (removed {initial_plays - final_plays})")
        
    def engineer_features(self):
        """Create derived features from raw data."""
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING")
        print("=" * 80)
        
        print("\nProcessing input features...")
        
        # Sort data
        self.input_df = self.input_df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
        
        # Coordinate normalization (flip left-going plays)
        self.input_df['x_norm'] = np.where(
            self.input_df['play_direction'] == 'left',
            120 - self.input_df['x'],
            self.input_df['x']
        )
        self.input_df['y_norm'] = np.where(
            self.input_df['play_direction'] == 'left',
            53.3 - self.input_df['y'],
            self.input_df['y']
        )
        
        # Direction normalization
        self.input_df['dir_norm'] = np.where(
            self.input_df['play_direction'] == 'left',
            (self.input_df['dir'] + 180) % 360,
            self.input_df['dir']
        )
        self.input_df['o_norm'] = np.where(
            self.input_df['play_direction'] == 'left',
            (self.input_df['o'] + 180) % 360,
            self.input_df['o']
        )
        
        # Relative to ball position
        self.input_df['x_rel_ball'] = self.input_df['x_norm'] - self.input_df['ball_land_x']
        self.input_df['y_rel_ball'] = self.input_df['y_norm'] - self.input_df['ball_land_y']
        
        # Velocity components (polar to Cartesian)
        self.input_df['vx'] = self.input_df['s'] * np.cos(np.radians(self.input_df['dir_norm']))
        self.input_df['vy'] = self.input_df['s'] * np.sin(np.radians(self.input_df['dir_norm']))
        
        # Acceleration components
        self.input_df['ax'] = self.input_df['a'] * np.cos(np.radians(self.input_df['dir_norm']))
        self.input_df['ay'] = self.input_df['a'] * np.sin(np.radians(self.input_df['dir_norm']))
        
        # Distance and angle to target
        self.input_df['dist_to_target'] = np.sqrt(
            (self.input_df['x_norm'] - self.input_df['ball_land_x'])**2 +
            (self.input_df['y_norm'] - self.input_df['ball_land_y'])**2
        )
        self.input_df['angle_to_target'] = np.degrees(np.arctan2(
            self.input_df['ball_land_y'] - self.input_df['y_norm'],
            self.input_df['ball_land_x'] - self.input_df['x_norm']
        ))
        
        # TIME FEATURES (critical for variable-length sequences)
        print("Adding time-based features...")
        self.input_df['frames_in_play'] = self.input_df.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].transform('count')
        self.input_df['time_to_throw'] = self.input_df.groupby(['game_id', 'play_id', 'nfl_id']).cumcount(ascending=False) + 1
        self.input_df['frame_progress'] = (self.input_df.groupby(['game_id', 'play_id', 'nfl_id']).cumcount() + 1) / self.input_df['frames_in_play']
        
        print(f"Created {len(config.INPUT_FEATURES)} input features")
        
        # Process output features
        print("\nProcessing output features...")
        self.output_df = self.output_df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
        
        # Get play direction from input
        play_directions = self.input_df[['game_id', 'play_id', 'play_direction']].drop_duplicates()
        self.output_df = self.output_df.merge(play_directions, on=['game_id', 'play_id'], how='left')
        
        # Normalize output coordinates
        self.output_df['x_norm'] = np.where(
            self.output_df['play_direction'] == 'left',
            120 - self.output_df['x'],
            self.output_df['x']
        )
        self.output_df['y_norm'] = np.where(
            self.output_df['play_direction'] == 'left',
            53.3 - self.output_df['y'],
            self.output_df['y']
        )
        
        # Truncate to max output frames
        self.output_df['frame_index'] = self.output_df.groupby(['game_id', 'play_id', 'nfl_id']).cumcount() + 1
        self.output_df = self.output_df[self.output_df['frame_index'] <= config.MAX_OUTPUT_FRAMES]
        
        print(f"Output data truncated to {config.MAX_OUTPUT_FRAMES} frames")
        
    def scale_features(self):
        """Normalize features using z-score scaling."""
        print("\n" + "=" * 80)
        print("FEATURE SCALING")
        print("=" * 80)
        
        # Calculate scaling parameters from input data
        for feature in config.INPUT_FEATURES:
            if feature in self.input_df.columns:
                mean = self.input_df[feature].mean()
                std = self.input_df[feature].std()
                
                if std > 0:
                    self.input_df[feature] = (self.input_df[feature] - mean) / std
                    self.scaling_params[feature] = {'mean': mean, 'std': std}
                    print(f"  {feature:20s} - Mean: {mean:8.3f}, Std: {std:8.3f}")
        
        # Scale output features using same parameters
        for feature in config.OUTPUT_FEATURES:
            if feature in self.scaling_params:
                params = self.scaling_params[feature]
                self.output_df[feature] = (self.output_df[feature] - params['mean']) / params['std']
        
        print(f"\nScaled {len(self.scaling_params)} features")
        
    def validate_data(self):
        """Check for data quality issues."""
        print("\n" + "=" * 80)
        print("DATA VALIDATION")
        print("=" * 80)
        
        issues = []
        
        # Check for NaN/Inf in features
        print("\nChecking for invalid values...")
        for feature in config.INPUT_FEATURES:
            if feature in self.input_df.columns:
                nan_count = self.input_df[feature].isna().sum()
                inf_count = np.isinf(self.input_df[feature]).sum()
                if nan_count > 0 or inf_count > 0:
                    msg = f"  ⚠️  {feature}: {nan_count} NaN, {inf_count} Inf"
                    print(msg)
                    issues.append(msg)
        
        if not issues:
            print("  ✓ No invalid values detected")
        
        # Check for duplicate rows
        print("\nChecking for duplicates...")
        input_dupes = self.input_df.duplicated(subset=['game_id', 'play_id', 'nfl_id', 'frame_id']).sum()
        output_dupes = self.output_df.duplicated(subset=['game_id', 'play_id', 'nfl_id', 'frame_id']).sum()
        
        if input_dupes > 0:
            msg = f"  ⚠️  Found {input_dupes} duplicate input rows"
            print(msg)
            issues.append(msg)
        if output_dupes > 0:
            msg = f"  ⚠️  Found {output_dupes} duplicate output rows"
            print(msg)
            issues.append(msg)
        if input_dupes == 0 and output_dupes == 0:
            print("  ✓ No duplicates found")
        
        # Check for mismatched plays
        print("\nChecking for mismatched plays...")
        input_plays = set(self.input_df[['game_id', 'play_id']].drop_duplicates().itertuples(index=False, name=None))
        output_plays = set(self.output_df[['game_id', 'play_id']].drop_duplicates().itertuples(index=False, name=None))
        
        missing_output = input_plays - output_plays
        missing_input = output_plays - input_plays
        
        if missing_output:
            msg = f"  ⚠️  {len(missing_output)} plays in input but not output"
            print(msg)
            issues.append(msg)
        if missing_input:
            msg = f"  ⚠️  {len(missing_input)} plays in output but not input"
            print(msg)
            issues.append(msg)
        if not missing_output and not missing_input:
            print("  ✓ All plays matched between input and output")
        
        return issues
        
    def create_metadata(self):
        """Generate metadata about processed data."""
        print("\n" + "=" * 80)
        print("CREATING METADATA")
        print("=" * 80)
        
        # Play-level metadata
        play_meta = self.input_df.groupby(['game_id', 'play_id']).agg({
            'frame_id': 'nunique',
            'nfl_id': 'nunique',
            'ball_land_x': 'first',
            'ball_land_y': 'first'
        }).rename(columns={
            'frame_id': 'num_input_frames',
            'nfl_id': 'num_input_players'
        })
        
        output_meta = self.output_df.groupby(['game_id', 'play_id']).agg({
            'frame_id': 'nunique',
            'nfl_id': 'nunique'
        }).rename(columns={
            'frame_id': 'num_output_frames',
            'nfl_id': 'num_output_players'
        })
        
        play_meta = play_meta.join(output_meta)
        
        # Global metadata
        self.metadata['global'] = {
            'num_plays': len(play_meta),
            'max_input_frames': config.MAX_INPUT_FRAMES,
            'max_output_frames': config.MAX_OUTPUT_FRAMES,
            'max_players': config.MAX_PLAYERS,
            'num_input_features': len(config.INPUT_FEATURES),
            'num_output_features': len(config.OUTPUT_FEATURES),
            'input_features': config.INPUT_FEATURES,
            'output_features': config.OUTPUT_FEATURES
        }
        
        self.metadata['plays'] = play_meta
        
        print(f"\nMetadata created for {len(play_meta)} plays")
        
    def save_processed_data(self):
        """Save processed data and metadata."""
        print("\n" + "=" * 80)
        print("SAVING PROCESSED DATA")
        print("=" * 80)
        
        # Save as parquet (efficient for large datasets)
        print(f"\nSaving input sequences to: {config.PROCESSED_INPUT}")
        self.input_df.to_parquet(config.PROCESSED_INPUT, index=False)
        
        print(f"Saving output sequences to: {config.PROCESSED_OUTPUT}")
        self.output_df.to_parquet(config.PROCESSED_OUTPUT, index=False)
        
        # Save metadata
        print(f"Saving metadata to: {config.METADATA_FILE}")
        with open(config.METADATA_FILE, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save scaling parameters
        print(f"Saving scaling parameters to: {config.SCALING_PARAMS}")
        with open(config.SCALING_PARAMS, 'wb') as f:
            pickle.dump(self.scaling_params, f)
        
        print("\n✓ All data saved successfully")
        
    def generate_report(self):
        """Generate comprehensive HTML report of data quality."""
        print("\n" + "=" * 80)
        print("GENERATING DATA QUALITY REPORT")
        print("=" * 80)
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Football Data Quality Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-bottom: 2px solid #95a5a6; padding-bottom: 5px; }}
        h3 {{ color: #7f8c8d; }}
        .section {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-label {{ font-weight: bold; color: #7f8c8d; }}
        .metric-value {{ font-size: 24px; color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th {{ background-color: #3498db; color: white; padding: 12px; text-align: left; }}
        td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .alert {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; }}
        .success {{ background-color: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 15px 0; }}
        .stat-box {{ display: inline-block; background-color: #ecf0f1; padding: 15px; margin: 10px; border-radius: 5px; min-width: 150px; }}
        .week-badge {{ display: inline-block; background-color: #3498db; color: white; padding: 5px 10px; margin: 5px; border-radius: 3px; font-size: 12px; }}
    </style>
</head>
<body>
    <h1>🏈 Football Trajectory Prediction - Data Quality Report</h1>
    <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="section">
        <h2>📁 Data Sources</h2>
        <p><strong>Weeks Processed:</strong></p>
"""
        
        # Add week badges
        for week in self.metadata.get('weeks_processed', []):
            html += f'<span class="week-badge">{week}</span>'
        
        html += f"""
        <p><strong>Total Files:</strong> {len(self.metadata.get('weeks_processed', []))} input/output pairs</p>
    </div>
    
    <div class="section">
        <h2>📊 Dataset Overview</h2>
        <div class="stat-box">
            <div class="metric-label">Total Plays</div>
            <div class="metric-value">{self.metadata['global']['num_plays']}</div>
        </div>
        <div class="stat-box">
            <div class="metric-label">Input Rows</div>
            <div class="metric-value">{len(self.input_df):,}</div>
        </div>
        <div class="stat-box">
            <div class="metric-label">Output Rows</div>
            <div class="metric-value">{len(self.output_df):,}</div>
        </div>
        <div class="stat-box">
            <div class="metric-label">Features</div>
            <div class="metric-value">{len(config.INPUT_FEATURES)}</div>
        </div>
    </div>
    
    <div class="section">
        <h2>📏 Sequence Lengths</h2>
        <h3>Input Frames per Play</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
"""
        
        input_frames = self.metadata['plays']['num_input_frames']
        for stat in ['mean', 'min', 'max', 'std']:
            value = getattr(input_frames, stat)()
            html += f"<tr><td>{stat.upper()}</td><td>{value:.2f}</td></tr>"
        
        html += f"""
        </table>
        
        <h3>Output Frames per Play</h3>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
"""
        
        output_frames = self.metadata['plays']['num_output_frames']
        for stat in ['mean', 'min', 'max', 'std']:
            value = getattr(output_frames, stat)()
            html += f"<tr><td>{stat.upper()}</td><td>{value:.2f}</td></tr>"
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>✨ Feature Engineering</h2>
        <p><strong>Input Features:</strong></p>
        <ul>
"""
        
        for feat in config.INPUT_FEATURES:
            html += f"<li>{feat}</li>"
        
        html += """
        </ul>
    </div>
    
    <div class="section">
        <h2>📐 Feature Scaling Parameters</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>Mean</th>
                <th>Std Dev</th>
            </tr>
"""
        
        for feat, params in self.scaling_params.items():
            html += f"<tr><td>{feat}</td><td>{params['mean']:.4f}</td><td>{params['std']:.4f}</td></tr>"
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>🎯 Top 10 Longest Plays</h2>
        <table>
            <tr>
                <th>Game ID</th>
                <th>Play ID</th>
                <th>Input Frames</th>
                <th>Output Frames</th>
                <th>Players</th>
            </tr>
"""
        
        top_plays = self.metadata['plays'].nlargest(10, 'num_input_frames')
        for (game_id, play_id), row in top_plays.iterrows():
            html += f"""
            <tr>
                <td>{game_id}</td>
                <td>{play_id}</td>
                <td>{row['num_input_frames']}</td>
                <td>{row['num_output_frames']}</td>
                <td>{row['num_input_players']}</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>✅ Data Validation Results</h2>
"""
        
        issues = self.validate_data()
        if issues:
            html += '<div class="alert"><strong>⚠️ Issues Found:</strong><ul>'
            for issue in issues:
                html += f"<li>{issue}</li>"
            html += '</ul></div>'
        else:
            html += '<div class="success"><strong>✓ No data quality issues detected!</strong></div>'
        
        html += """
    </div>
    
    <div class="section">
        <h2>📁 Output Files</h2>
        <ul>
            <li><code>input_sequences.parquet</code> - Processed input sequences</li>
            <li><code>output_sequences.parquet</code> - Processed output sequences</li>
            <li><code>metadata.pkl</code> - Play metadata and statistics</li>
            <li><code>scaling_params.pkl</code> - Feature scaling parameters</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>🚀 Ready for Training</h2>
        <div class="success">
            <p><strong>✓ Data preprocessing complete!</strong></p>
            <p>Your data is now ready for transformer model training.</p>
            <p>Proceed to: <code>python main.py --train</code></p>
        </div>
    </div>
</body>
</html>
"""
        
        with open(config.DATA_REPORT, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"\n✓ Report saved to: {config.DATA_REPORT}")
        print("  Open this file in your browser to view the full report")
        
    def run(self):
        """Execute complete preprocessing pipeline."""
        print("\n")
        print("=" * 80)
        print("FOOTBALL TRAJECTORY PREDICTION - DATA PREPROCESSING")
        print("=" * 80)
        
        self.load_data()
        self.explore_data()
        self.clean_data()
        self.engineer_features()
        self.scale_features()
        self.validate_data()
        self.create_metadata()
        self.save_processed_data()
        self.generate_report()
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE ✓")
        print("=" * 80)
        print(f"\nProcessed data saved to: {config.PROCESSED_DATA_DIR}")
        print(f"Quality report saved to: {config.DATA_REPORT}")
        print("\nNext step: Run training with 'python main.py --train'")
        

if __name__ == "__main__":
    preprocessor = FootballDataPreprocessor()
    preprocessor.run()