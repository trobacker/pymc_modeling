#!/usr/bin/env python3
"""
Visualize BART Model Submissions
=============================================================================

This script creates visualizations of the model predictions for sanity checking.

Usage:
    python plot_submission.py --nowcast-date 2024-11-13
    python plot_submission.py --nowcast-date 2024-11-13 --location CA

Author: Auto-generated workflow
Date: 2025-12-31
"""

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


def setup_logging(level="INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_submission(submissions_dir, nowcast_date, model_name):
    """Load submission parquet file"""
    logger = logging.getLogger(__name__)

    submissions_dir = Path(submissions_dir)
    submission_file = submissions_dir / f"{nowcast_date}-{model_name}.parquet"

    if not submission_file.exists():
        raise FileNotFoundError(f"Submission file not found: {submission_file}")

    logger.info(f"Loading submission from: {submission_file}")
    df = pd.read_parquet(submission_file)

    return df


def plot_location_trajectories(df, location, nowcast_date, output_dir):
    """
    Plot variant trajectories for a specific location with uncertainty.

    Shows:
    - Mean predictions (solid lines)
    - 50% credible interval (shaded)
    - 90% credible interval (lighter shaded)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Plotting trajectories for location: {location}")

    # Filter for location
    df_loc = df[df['location'] == location].copy()

    if len(df_loc) == 0:
        logger.warning(f"No data for location {location}")
        return

    # Convert dates
    df_loc['target_date'] = pd.to_datetime(df_loc['target_date'])
    nowcast_dt = pd.to_datetime(nowcast_date)

    # Get clades
    clades = sorted(df_loc['clade'].unique())

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, clade in enumerate(clades):
        if i >= len(axes):
            break

        ax = axes[i]

        df_clade = df_loc[df_loc['clade'] == clade].copy()
        df_clade = df_clade.sort_values('target_date')

        # Get samples and mean
        samples = df_clade[df_clade['output_type'] == 'sample']
        mean = df_clade[df_clade['output_type'] == 'mean']

        if len(mean) > 0:
            # Plot mean
            ax.plot(mean['target_date'], mean['value'],
                   label='Mean', color='black', linewidth=2)

            # Compute quantiles from samples
            if len(samples) > 0:
                quantiles = samples.groupby('target_date')['value'].quantile(
                    [0.05, 0.25, 0.75, 0.95]
                ).unstack()

                dates = quantiles.index

                # 90% CI
                ax.fill_between(dates, quantiles[0.05], quantiles[0.95],
                               alpha=0.2, color='blue', label='90% CI')

                # 50% CI
                ax.fill_between(dates, quantiles[0.25], quantiles[0.75],
                               alpha=0.4, color='blue', label='50% CI')

        # Add nowcast date line
        ax.axvline(nowcast_dt, color='red', linestyle='--', linewidth=1,
                  label='Nowcast date', alpha=0.7)

        ax.set_title(f"{clade}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Proportion")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # Rotate x-axis labels
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"BART Variant Trajectories - {location}\nNowcast Date: {nowcast_date}",
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"trajectories_{location}_{nowcast_date}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot: {output_file}")

    plt.close()


def plot_forecast_comparison(df, locations, nowcast_date, output_dir):
    """
    Plot forecast trajectories across multiple locations for comparison.
    """
    logger = logging.getLogger(__name__)
    logger.info("Plotting multi-location forecast comparison...")

    # Filter for selected locations
    df_filtered = df[df['location'].isin(locations)].copy()
    df_filtered['target_date'] = pd.to_datetime(df_filtered['target_date'])
    nowcast_dt = pd.to_datetime(nowcast_date)

    # Get clades
    clades = sorted(df_filtered['clade'].unique())

    # Create subplots for each clade
    n_clades = len(clades)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, clade in enumerate(clades):
        if i >= len(axes):
            break

        ax = axes[i]

        df_clade = df_filtered[df_filtered['clade'] == clade]

        # Plot mean for each location
        for location in locations:
            df_loc = df_clade[
                (df_clade['location'] == location) &
                (df_clade['output_type'] == 'mean')
            ].sort_values('target_date')

            if len(df_loc) > 0:
                ax.plot(df_loc['target_date'], df_loc['value'],
                       label=location, linewidth=2, alpha=0.8)

        # Add nowcast date line
        ax.axvline(nowcast_dt, color='red', linestyle='--', linewidth=1,
                  label='Nowcast date', alpha=0.5)

        ax.set_title(f"{clade}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Date")
        ax.set_ylabel("Proportion")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Hide extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    locations_str = ', '.join(locations)
    plt.suptitle(f"BART Multi-Location Forecast Comparison\nLocations: {locations_str}\nNowcast Date: {nowcast_date}",
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    output_dir = Path(output_dir)
    output_file = output_dir / f"forecast_comparison_{nowcast_date}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Saved plot: {output_file}")

    plt.close()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Visualize BART model submissions"
    )
    parser.add_argument(
        "--nowcast-date",
        type=str,
        required=True,
        help="Nowcast date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: ../config.yaml)"
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Location to plot (2-letter code). If not specified, plots multiple locations."
    )
    parser.add_argument(
        "--submissions-dir",
        type=str,
        default=None,
        help="Submissions directory (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (overrides config)"
    )

    args = parser.parse_args()

    # Determine config path
    if args.config is None:
        # Default: look for config.yaml in parent directory (if running from scripts/)
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / "config.yaml"
        if not config_path.exists():
            # Fallback: current directory
            config_path = Path("config.yaml")
    else:
        config_path = Path(args.config)

    # Load configuration
    config = load_config(config_path)
    logger = setup_logging(config['logging']['level'])

    submissions_dir = args.submissions_dir or config['paths']['submissions']
    output_dir = args.output_dir or Path(config['paths']['logs']) / "plots"

    logger.info(f"Creating visualizations for nowcast date: {args.nowcast_date}")

    try:
        # Load submission
        model_name = f"{config['model']['team']}-{config['model']['name']}"
        df = load_submission(submissions_dir, args.nowcast_date, model_name)

        logger.info(f"Submission loaded: {len(df)} rows")

        # Plot location trajectories
        if args.location:
            # Single location
            plot_location_trajectories(df, args.location, args.nowcast_date, output_dir)
        else:
            # Multiple locations
            example_locations = ['CA', 'NY', 'TX', 'FL']
            available_locations = [loc for loc in example_locations if loc in df['location'].unique()]

            if available_locations:
                # Plot first location in detail
                plot_location_trajectories(df, available_locations[0],
                                          args.nowcast_date, output_dir)

                # Plot comparison
                if len(available_locations) > 1:
                    plot_forecast_comparison(df, available_locations[:4],
                                            args.nowcast_date, output_dir)

        print("\n" + "=" * 70)
        print("PLOTTING COMPLETE")
        print("=" * 70)
        print(f"Plots saved to: {output_dir}")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Error creating plots: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
