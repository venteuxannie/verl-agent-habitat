#!/usr/bin/env python3
"""
Script to analyze GRPO training outputs stored by the training process.
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def load_jsonl_data(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_rollout_data(output_dir):
    """Analyze rollout data from the output directory."""
    rollout_dir = Path(output_dir) / "rollouts"
    
    if not rollout_dir.exists():
        print(f"Rollout directory not found: {rollout_dir}")
        return
    
    jsonl_files = list(rollout_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {rollout_dir}")
        return
    
    print(f"Found {len(jsonl_files)} rollout files:")
    for file in sorted(jsonl_files):
        print(f"  - {file.name}")
    
    # Load all data
    all_data = []
    for file in sorted(jsonl_files):
        data = load_jsonl_data(file)
        all_data.extend(data)
    
    if not all_data:
        print("No data found in rollout files")
        return
    
    df = pd.DataFrame(all_data)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"Training steps: {sorted(df['step'].unique())}")
    
    # Basic statistics
    print("\n=== Basic Statistics ===")
    print(f"Average score: {df['score'].mean():.4f} ± {df['score'].std():.4f}")
    print(f"Score range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")
    
    if 'advantages' in df.columns:
        print(f"Average advantage: {df['advantages'].mean():.4f} ± {df['advantages'].std():.4f}")
        print(f"Advantage range: [{df['advantages'].min():.4f}, {df['advantages'].max():.4f}]")
    
    if 'returns' in df.columns:
        print(f"Average return: {df['returns'].mean():.4f} ± {df['returns'].std():.4f}")
        print(f"Return range: [{df['returns'].min():.4f}, {df['returns'].max():.4f}]")
    
    # Action validity statistics
    if 'is_action_valid' in df.columns:
        valid_ratio = df['is_action_valid'].mean()
        print(f"Valid action ratio: {valid_ratio:.4f}")
    
    # Data source distribution
    if 'data_source' in df.columns:
        print(f"\nData source distribution:")
        source_counts = df['data_source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count} ({count/len(df)*100:.1f}%)")
    
    # Trajectory analysis
    if 'traj_uid' in df.columns:
        unique_trajs = df['traj_uid'].nunique()
        print(f"\nUnique trajectories: {unique_trajs}")
        print(f"Samples per trajectory: {len(df) / unique_trajs:.1f}")
    
    return df

def plot_training_progress(df, output_dir):
    """Plot training progress over time."""
    if 'step' not in df.columns:
        print("No step information available for plotting")
        return
    
    # Group by step and calculate statistics
    step_stats = df.groupby('step').agg({
        'score': ['mean', 'std', 'min', 'max'],
        'advantages': ['mean', 'std'] if 'advantages' in df.columns else [],
        'returns': ['mean', 'std'] if 'returns' in df.columns else []
    }).reset_index()
    
    # Flatten column names
    step_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in step_stats.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Score over time
    axes[0, 0].plot(step_stats['step'], step_stats['score_mean'], 'b-o', label='Mean Score')
    axes[0, 0].fill_between(step_stats['step'], 
                           step_stats['score_mean'] - step_stats['score_std'],
                           step_stats['score_mean'] + step_stats['score_std'],
                           alpha=0.3, color='blue')
    axes[0, 0].set_title('Score Over Training Steps')
    axes[0, 0].set_xlabel('Training Step')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].grid(True)
    
    # Advantages over time
    if 'advantages_mean' in step_stats.columns:
        axes[0, 1].plot(step_stats['step'], step_stats['advantages_mean'], 'r-o', label='Mean Advantage')
        axes[0, 1].fill_between(step_stats['step'],
                               step_stats['advantages_mean'] - step_stats['advantages_std'],
                               step_stats['advantages_mean'] + step_stats['advantages_std'],
                               alpha=0.3, color='red')
        axes[0, 1].set_title('Advantages Over Training Steps')
        axes[0, 1].set_xlabel('Training Step')
        axes[0, 1].set_ylabel('Advantage')
        axes[0, 1].grid(True)
    
    # Score distribution
    axes[1, 0].hist(df['score'], bins=30, alpha=0.7, color='green')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True)
    
    # Advantages distribution
    if 'advantages' in df.columns:
        axes[1, 1].hist(df['advantages'], bins=30, alpha=0.7, color='orange')
        axes[1, 1].set_title('Advantages Distribution')
        axes[1, 1].set_xlabel('Advantage')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / "training_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Training analysis plot saved to: {plot_path}")
    
    plt.show()

def show_sample_outputs(df, num_samples=3):
    """Show sample outputs from the data."""
    print(f"\n=== Sample Outputs (showing {num_samples} samples) ===")
    
    for i, (_, row) in enumerate(df.head(num_samples).iterrows()):
        print(f"\n--- Sample {i+1} ---")
        print(f"Input: {row['input'][:200]}...")
        print(f"Output: {row['output'][:200]}...")
        print(f"Score: {row['score']:.4f}")
        if 'advantages' in row:
            print(f"Advantage: {row['advantages']:.4f}")
        if 'returns' in row:
            print(f"Return: {row['returns']:.4f}")
        if 'is_action_valid' in row:
            print(f"Action Valid: {row['is_action_valid']}")
        if 'data_source' in row:
            print(f"Data Source: {row['data_source']}")

def main():
    parser = argparse.ArgumentParser(description="Analyze GRPO training outputs")
    parser.add_argument("--output-dir", type=str, default="./grpo_habitat_outputs",
                       help="Directory containing GRPO outputs")
    parser.add_argument("--show-samples", type=int, default=3,
                       help="Number of sample outputs to show")
    parser.add_argument("--plot", action="store_true",
                       help="Generate training progress plots")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GRPO Training Output Analysis")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    
    # Analyze rollout data
    df = analyze_rollout_data(args.output_dir)
    
    if df is not None:
        # Show sample outputs
        show_sample_outputs(df, args.show_samples)
        
        # Generate plots if requested
        if args.plot:
            try:
                plot_training_progress(df, args.output_dir)
            except ImportError:
                print("Matplotlib not available, skipping plots")
            except Exception as e:
                print(f"Error generating plots: {e}")
    
    print("\n" + "=" * 60)
    print("Analysis completed!")

if __name__ == "__main__":
    main()
