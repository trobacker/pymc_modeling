#!/usr/bin/env python3
"""
Fetch Training Data from Variant Nowcast Hub
=============================================================================

This script fetches the latest time-series data from the variant nowcast hub
for a specified nowcast date. The data is used to train the PyMC model.

Prerequisites:
    Activate the virtual environment first:
    $ source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows

Usage:
    python 01_fetch_data.py --nowcast-date 2024-11-13
    python 01_fetch_data.py --nowcast-date 2024-11-13 --config ../config.yaml

Author: Auto-generated workflow
Date: 2025-11-14
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys
import json

import polars as pl
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


def get_modeled_clades(hub_repo_path, nowcast_date, logger):
    """
    Load the required clades from the hub's modeled-clades JSON file.

    According to hub rules, submissions use the modeled-clades file from 2 days prior
    to the nowcast date. For example, a nowcast on 2025-11-14 uses 2025-11-12.json.

    Returns:
        list: List of clade names to model
    """
    # Nowcast submissions use clades file from 2 days prior
    nowcast_dt = datetime.strptime(nowcast_date, "%Y-%m-%d")
    clades_date = (nowcast_dt - timedelta(days=2)).strftime("%Y-%m-%d")

    clades_path = (Path(hub_repo_path) / "auxiliary-data" / "modeled-clades" /
                   f"{clades_date}.json")

    if not clades_path.exists():
        # Try exact nowcast date if 2-day-prior doesn't exist
        clades_path = (Path(hub_repo_path) / "auxiliary-data" / "modeled-clades" /
                       f"{nowcast_date}.json")

    if not clades_path.exists():
        raise FileNotFoundError(
            f"Modeled clades file not found for {clades_date} or {nowcast_date}. "
            f"Expected at: {clades_path}"
        )

    # Load JSON file
    with open(clades_path, 'r') as f:
        clades_data = json.load(f)

    clades = clades_data.get("clades", [])

    if not clades:
        raise ValueError(f"No clades found in {clades_path}")

    logger.info(f"Loaded {len(clades)} modeled clades from {clades_path.name}")
    logger.info(f"Clades: {clades}")

    return clades


def get_s3_data_date(nowcast_date):
    """
    Determine which S3 data file to use for the given nowcast date.

    S3 data is uploaded every Monday. We use the most recent Monday's data
    that is <= nowcast_date.

    Returns:
        str: Date string in YYYY-MM-DD format for the S3 file
    """
    nowcast_dt = datetime.strptime(nowcast_date, "%Y-%m-%d")

    # Find the most recent Monday on or before the nowcast date
    days_since_monday = nowcast_dt.weekday()  # 0=Monday, 6=Sunday

    if days_since_monday >= 0:
        # Go back to the most recent Monday
        s3_date = nowcast_dt - timedelta(days=days_since_monday)
    else:
        # Should not happen since weekday() returns 0-6
        s3_date = nowcast_dt

    return s3_date.strftime("%Y-%m-%d")


def fetch_training_data_from_s3(nowcast_date, lookback_days, logger):
    """
    Fetch training data from S3 clade counts bucket.

    The S3 bucket contains weekly clade count data uploaded every Monday.
    URL format: https://covid-clade-counts.s3.amazonaws.com/YYYY-MM-DD_covid_clade_counts.parquet

    Returns:
        pl.DataFrame: Clade count data filtered for training period
    """
    # Determine which S3 file to use (most recent Monday)
    s3_date = get_s3_data_date(nowcast_date)

    # Construct S3 URL
    s3_url = f"https://covid-clade-counts.s3.amazonaws.com/{s3_date}_covid_clade_counts.parquet"

    logger.info(f"Fetching data from S3: {s3_url}")

    try:
        # Load data from S3 using polars with skip_signature for public access
        df = pl.scan_parquet(s3_url, storage_options={"skip_signature": "true"}).collect()
        logger.info(f"Loaded {len(df)} rows from S3")

    except Exception as e:
        raise RuntimeError(
            f"Failed to fetch data from S3: {s3_url}\n"
            f"Error: {e}\n"
            f"Make sure the S3 file exists for date {s3_date}"
        )

    # Filter for training period (lookback days before nowcast_date)
    nowcast_dt = datetime.strptime(nowcast_date, "%Y-%m-%d").date()
    start_date = nowcast_dt - timedelta(days=lookback_days)

    # Assume S3 data has a 'date' column (adjust if needed based on actual schema)
    df = df.filter(
        (pl.col("date") >= start_date) &
        (pl.col("date") <= nowcast_dt)
    )

    logger.info(f"Filtered to training period: {start_date} to {nowcast_dt}")
    logger.info(f"Training data shape: {df.shape}")

    return df


def filter_and_aggregate_data(df, min_sequences, modeled_clades):
    """
    Filter data and aggregate by location-date-clade.

    Uses only the hub-specified modeled clades from the auxiliary-data/modeled-clades JSON file.

    Args:
        df: Raw S3 clade count data
        min_sequences: Minimum sequences required for a location-date
        modeled_clades: List of clades to model (from hub's JSON file)

    Returns:
        pl.DataFrame: Filtered and aggregated data
    """
    # Rename 'date' column to 'target_date' for consistency with downstream pipeline
    if "date" in df.columns:
        df = df.rename({"date": "target_date"})

    # Determine which column contains the sequence counts
    # Common names: 'count', 'sequences', 'n', 'observation'
    count_col = None
    for col in ['count', 'sequences', 'n', 'observation', 'seq_count']:
        if col in df.columns:
            count_col = col
            break

    if count_col is None:
        raise ValueError(f"Could not find sequence count column. Available columns: {df.columns}")

    # Filter for modeled clades only
    df = df.filter(pl.col("clade").is_in(modeled_clades))

    # Aggregate by location-date-clade (sum counts)
    df_agg = df.group_by(["location", "target_date", "clade"]).agg(
        pl.col(count_col).sum().alias("sequences")
    )

    # Calculate total sequences per location-date
    totals = df_agg.group_by(["location", "target_date"]).agg(
        pl.col("sequences").sum().alias("total_sequences")
    )

    # Join totals and filter by minimum sequences
    df_agg = df_agg.join(totals, on=["location", "target_date"])
    df_agg = df_agg.filter(pl.col("total_sequences") >= min_sequences)

    # Ensure all modeled clades are present (fill with 0 if missing)
    # Create a complete grid
    locations = df_agg["location"].unique().to_list()
    dates = df_agg["target_date"].unique().to_list()

    grid = []
    for loc in locations:
        for date in dates:
            for clade in modeled_clades:
                grid.append({"location": loc, "target_date": date, "clade": clade})

    grid_df = pl.DataFrame(grid)

    # Left join to ensure all combinations exist
    df_complete = grid_df.join(
        df_agg.select(["location", "target_date", "clade", "sequences"]),
        on=["location", "target_date", "clade"],
        how="left"
    ).with_columns(
        pl.col("sequences").fill_null(0)
    )

    # Re-calculate total sequences after filling
    totals = df_complete.group_by(["location", "target_date"]).agg(
        pl.col("sequences").sum().alias("total_sequences")
    )

    df_complete = df_complete.join(
        totals.select(["location", "target_date", "total_sequences"]),
        on=["location", "target_date"]
    )

    return df_complete


def save_training_data(df, output_dir, nowcast_date):
    """Save training data to disk"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"training_data_{nowcast_date}.parquet"
    df.write_parquet(output_file)

    return output_file


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Fetch training data from variant nowcast hub"
    )
    parser.add_argument(
        "--nowcast-date",
        type=str,
        required=True,
        help="Nowcast date in YYYY-MM-DD format (e.g., 2024-11-13)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for training data (overrides config)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger = setup_logging(config['logging']['level'])

    logger.info(f"Fetching data for nowcast date: {args.nowcast_date}")

    # Get parameters from config
    hub_repo_path = config['hub']['repo_path']
    lookback_days = config['data']['training_lookback_days']
    min_sequences = config['data']['min_sequences']

    output_dir = args.output_dir or config['paths']['data']

    try:
        # Load hub-specified modeled clades
        logger.info("Loading modeled clades from hub...")
        modeled_clades = get_modeled_clades(hub_repo_path, args.nowcast_date, logger)

        # Fetch training data from S3
        logger.info(f"Fetching training data from S3 (lookback: {lookback_days} days)...")
        df = fetch_training_data_from_s3(args.nowcast_date, lookback_days, logger)
        logger.info(f"Raw S3 data shape: {df.shape}")

        # Filter and aggregate
        logger.info("Filtering and aggregating data...")
        df_processed = filter_and_aggregate_data(
            df, min_sequences, modeled_clades
        )
        logger.info(f"Processed data shape: {df_processed.shape}")
        logger.info(f"Unique locations: {df_processed['location'].n_unique()}")
        logger.info(f"Unique dates: {df_processed['target_date'].n_unique()}")
        logger.info(f"Unique clades: {df_processed['clade'].n_unique()}")

        # Save training data
        logger.info("Saving training data...")
        output_file = save_training_data(df_processed, output_dir, args.nowcast_date)
        logger.info(f"Training data saved to: {output_file}")

        # Print summary
        s3_date = get_s3_data_date(args.nowcast_date)
        print("\n" + "=" * 70)
        print("DATA FETCHING COMPLETE")
        print("=" * 70)
        print(f"Nowcast date: {args.nowcast_date}")
        print(f"S3 data date: {s3_date}")
        print(f"Training period: {lookback_days} days lookback")
        print(f"Output file: {output_file}")
        print(f"\nData summary:")
        print(f"  Observations: {len(df_processed)}")
        print(f"  Locations: {df_processed['location'].n_unique()}")
        print(f"  Dates: {df_processed['target_date'].n_unique()}")
        print(f"  Modeled clades: {df_processed['clade'].unique().to_list()}")
        print(f"  Date range: {df_processed['target_date'].min()} to {df_processed['target_date'].max()}")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Error fetching data: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
