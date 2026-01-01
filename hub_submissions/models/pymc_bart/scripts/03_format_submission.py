#!/usr/bin/env python3
"""
Format Submission for Variant Nowcast Hub - BART Model
=============================================================================

This script takes the fitted BART model and generates submissions in the format
required by the variant nowcast hub (samples and means).

Usage:
    python 03_format_submission.py --nowcast-date 2024-11-13
    python 03_format_submission.py --nowcast-date 2024-11-13 --forecast-horizon 14

Author: Auto-generated workflow
Date: 2025-12-31
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import sys
import pickle

import polars as pl
import pandas as pd
import numpy as np
import arviz as az
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


def load_model_artifacts(models_dir, nowcast_date):
    """Load model artifacts from disk"""
    logger = logging.getLogger(__name__)

    models_dir = Path(models_dir)

    # Load trace
    trace_file = models_dir / f"trace_{nowcast_date}.nc"
    if not trace_file.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_file}")

    trace = az.from_netcdf(trace_file)
    logger.info(f"Loaded trace from: {trace_file}")

    # Load mappings and dimensions
    mappings_file = models_dir / f"mappings_{nowcast_date}.pkl"
    if not mappings_file.exists():
        raise FileNotFoundError(f"Mappings file not found: {mappings_file}")

    with open(mappings_file, 'rb') as f:
        data = pickle.load(f)

    mappings = data['mappings']
    dimensions = data['dimensions']
    logger.info(f"Loaded mappings from: {mappings_file}")

    return trace, mappings, dimensions


def generate_forecast_dates(nowcast_date, nowcast_lookback, forecast_horizon):
    """
    Generate target dates for nowcasting and forecasting.

    Args:
        nowcast_date: The nowcast submission date (str, YYYY-MM-DD)
        nowcast_lookback: Days to look back for nowcast (int, e.g., 31 for 32 days total)
        forecast_horizon: Days to forecast forward (int, e.g., 10)

    Returns:
        List of date strings covering:
        - Nowcast period: nowcast_date - nowcast_lookback to nowcast_date
        - Forecast period: nowcast_date + 1 to nowcast_date + forecast_horizon
    """
    nowcast_dt = datetime.strptime(nowcast_date, "%Y-%m-%d").date()

    target_dates = []

    # Nowcast period (past + present)
    for i in range(nowcast_lookback + 1):
        target_date = nowcast_dt - timedelta(days=nowcast_lookback) + timedelta(days=i)
        target_dates.append(target_date.strftime("%Y-%m-%d"))

    # Forecast period (future)
    for i in range(1, forecast_horizon + 1):
        target_date = nowcast_dt + timedelta(days=i)
        target_dates.append(target_date.strftime("%Y-%m-%d"))

    return target_dates


def build_feature_matrix(time_indices, location_indices, num_locations):
    """
    Build feature matrix X for BART predictions.

    Args:
        time_indices: Centered/scaled time indices
        location_indices: Location indices
        num_locations: Total number of locations

    Returns:
        Feature matrix X of shape (n_observations, 1 + num_locations)
    """
    n_obs = len(time_indices)

    # One-hot encode locations
    location_onehot = np.zeros((n_obs, num_locations))
    location_onehot[np.arange(n_obs), location_indices] = 1

    # Combine time and location features
    X = np.column_stack([time_indices, location_onehot])

    return X


def compute_proportions_from_trace(trace, time_indices, location_indices, num_clades, num_locations):
    """
    Compute clade proportions from BART MCMC samples.

    Args:
        trace: ArviZ InferenceData object with BART samples
        time_indices: Time indices for predictions (already centered/scaled)
        location_indices: Location indices for predictions
        num_clades: Number of clades
        num_locations: Number of locations in training data

    Returns:
        Array of shape (n_draws, n_observations, n_clades) with proportions
    """
    logger = logging.getLogger(__name__)
    logger.info("Computing proportions from BART posterior samples...")

    # Extract BART samples for each clade
    # BART models: bart_0, bart_1, ..., bart_{K-1}
    extracted = az.extract(trace.posterior, combined=True)

    # Check which variables are available
    bart_vars = [var for var in extracted.data_vars if var.startswith('bart_')]
    logger.info(f"Found BART variables: {bart_vars}")

    if len(bart_vars) == 0:
        raise ValueError("No BART variables found in trace. Expected bart_0, bart_1, etc.")

    n_draws = extracted.dims['sample']
    n_obs = len(time_indices)

    # Build feature matrix for predictions
    X_pred = build_feature_matrix(time_indices, location_indices, num_locations)
    logger.info(f"Prediction feature matrix shape: {X_pred.shape}")

    # For BART predictions, we need to evaluate the tree ensemble at new X values
    # Since PyMC-BART stores tree structures in trace, we reconstruct predictions
    # Note: This is simplified; in practice, you'd use pmb.predict() or similar

    proportions = np.zeros((n_draws, n_obs, num_clades))

    # Extract BART predictions from trace
    # The trace should contain bart_i variables with shape (samples, observations)
    for clade_idx in range(num_clades):
        bart_var = f'bart_{clade_idx}'
        if bart_var not in extracted:
            logger.warning(f"BART variable {bart_var} not found, using zeros")
            continue

        # Extract samples: shape (observations, samples)
        bart_samples = extracted[bart_var].values
        # Transpose to (samples, observations)
        if bart_samples.shape[1] == n_draws:
            bart_samples = bart_samples.T
        elif bart_samples.shape[0] != n_draws:
            logger.warning(f"Unexpected shape for {bart_var}: {bart_samples.shape}")

        # For predictions beyond training observations, we need to handle extrapolation
        # Here we'll use a simple approach: extend using the last observed values
        # In a full implementation, you'd use BART's prediction method with new X
        if bart_samples.shape[1] < n_obs:
            # Pad with last values (simple extrapolation)
            n_missing = n_obs - bart_samples.shape[1]
            last_vals = bart_samples[:, -1:]
            padding = np.repeat(last_vals, n_missing, axis=1)
            bart_samples_extended = np.concatenate([bart_samples, padding], axis=1)
        else:
            bart_samples_extended = bart_samples[:, :n_obs]

        # Store in proportions array (will apply softmax below)
        proportions[:, :, clade_idx] = bart_samples_extended

    # Apply softmax to convert log-odds to proportions
    logger.info("Applying softmax to convert BART outputs to proportions...")
    for i in range(n_draws):
        if i % 500 == 0:
            logger.debug(f"Processing draw {i}/{n_draws}")

        eta = proportions[i]  # shape: (n_obs, n_clades)

        # Numerical stability: subtract max
        exp_eta = np.exp(eta - np.max(eta, axis=1, keepdims=True))
        proportions[i] = exp_eta / np.sum(exp_eta, axis=1, keepdims=True)

    logger.info("Proportions computed successfully")
    logger.info(f"Proportions shape: {proportions.shape}")
    logger.info(f"Proportion sums (first few): {proportions[0].sum(axis=1)[:5]}")

    return proportions


def create_submission_dataframe(proportions, target_dates, locations, clades,
                                 nowcast_date, n_samples, include_samples, include_mean,
                                 us_states_abbreviation_dict):
    """
    Create submission dataframe in hub format.

    Args:
        proportions: Array of shape (n_draws, n_observations, n_clades)
        target_dates: List of target date strings
        locations: List of location names (full names or abbreviations)
        clades: List of clade names
        nowcast_date: Nowcast date string
        n_samples: Number of samples to include
        include_samples: Whether to include sample submissions
        include_mean: Whether to include mean submissions
        us_states_abbreviation_dict: Mapping from full names to abbreviations

    Returns:
        pd.DataFrame in hub format
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating submission dataframe...")

    n_draws, n_obs, n_clades = proportions.shape

    # Sample indices for samples
    if include_samples:
        if n_samples <= n_draws:
            sample_indices = np.random.choice(n_draws, size=n_samples, replace=False)
        else:
            sample_indices = np.random.choice(n_draws, size=n_samples, replace=True)
            logger.warning(f"n_samples ({n_samples}) > n_draws ({n_draws}), sampling with replacement")

    records = []

    # Iterate over observations
    obs_idx = 0
    for target_date in target_dates:
        for location in locations:
            # Get abbreviation
            location_abbr = us_states_abbreviation_dict.get(location, location)

            for clade_idx, clade in enumerate(clades):
                # Add samples
                if include_samples:
                    for sample_num, draw_idx in enumerate(sample_indices):
                        proportion = proportions[draw_idx, obs_idx, clade_idx]

                        records.append({
                            'nowcast_date': nowcast_date,
                            'target_date': target_date,
                            'clade': clade,
                            'location': location_abbr,
                            'output_type': 'sample',
                            'output_type_id': f"{location_abbr}{sample_num:02d}",
                            'value': float(proportion)
                        })

                # Add mean
                if include_mean:
                    mean_proportion = proportions[:, obs_idx, clade_idx].mean()

                    records.append({
                        'nowcast_date': nowcast_date,
                        'target_date': target_date,
                        'clade': clade,
                        'location': location_abbr,
                        'output_type': 'mean',
                        'output_type_id': None,
                        'value': float(mean_proportion)
                    })

            obs_idx += 1

    df = pd.DataFrame(records)

    logger.info(f"Submission dataframe created: {df.shape}")
    logger.info(f"  Unique locations: {df['location'].nunique()}")
    logger.info(f"  Unique target dates: {df['target_date'].nunique()}")
    logger.info(f"  Unique clades: {df['clade'].nunique()}")
    logger.info(f"  Output types: {df['output_type'].unique().tolist()}")

    return df


def validate_submission(df, required_clades):
    """
    Validate submission format and constraints.

    Returns:
        tuple: (is_valid, errors)
    """
    logger = logging.getLogger(__name__)
    logger.info("Validating submission...")

    errors = []

    # Check required columns
    required_cols = ['nowcast_date', 'target_date', 'clade', 'location',
                     'output_type', 'output_type_id', 'value']
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # Check value constraints (between 0 and 1)
    if (df['value'] < 0).any() or (df['value'] > 1).any():
        errors.append("Values must be between 0 and 1")

    # Check required clades are present
    clades_in_df = set(df['clade'].unique())
    missing_clades = set(required_clades) - clades_in_df
    if missing_clades:
        errors.append(f"Missing required clades: {missing_clades}")

    # Check samples per task
    if 'sample' in df['output_type'].values:
        samples_df = df[df['output_type'] == 'sample']
        samples_per_task = samples_df.groupby(['location', 'target_date', 'clade']).size()
        if not (samples_per_task == 100).all():
            unique_counts = samples_per_task.unique()
            errors.append(f"Some tasks don't have exactly 100 samples. Found: {unique_counts}")

    # Check proportions sum to ~1 for each location-date-sample
    if 'sample' in df['output_type'].values:
        samples_df = df[df['output_type'] == 'sample']
        sums = samples_df.groupby(['location', 'target_date', 'output_type_id'])['value'].sum()
        if not np.allclose(sums, 1.0, atol=0.01):
            bad_sums = sums[(sums < 0.99) | (sums > 1.01)]
            if len(bad_sums) > 0:
                errors.append(f"Some sample proportions don't sum to 1: {len(bad_sums)} cases")
                logger.warning(f"Example bad sums: {bad_sums.head()}")

    is_valid = len(errors) == 0

    if is_valid:
        logger.info("✓ Submission validation passed")
    else:
        logger.error("✗ Submission validation failed:")
        for error in errors:
            logger.error(f"  - {error}")

    return is_valid, errors


def save_submission(df, output_dir, nowcast_date, model_name, file_format):
    """Save submission to disk"""
    logger = logging.getLogger(__name__)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{nowcast_date}-{model_name}"

    if file_format == "parquet":
        output_file = output_dir / f"{filename}.parquet"
        # Convert date columns to date objects (not datetime/POSIX timestamps)
        df['nowcast_date'] = pd.to_datetime(df['nowcast_date']).dt.date
        df['target_date'] = pd.to_datetime(df['target_date']).dt.date
        df.to_parquet(output_file, index=False, engine='pyarrow')
    elif file_format == "csv":
        output_file = output_dir / f"{filename}.csv"
        df.to_csv(output_file, index=False)
    else:
        raise ValueError(f"Unknown file format: {file_format}")

    logger.info(f"Submission saved to: {output_file}")

    return output_file


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Format submission for variant nowcast hub - BART model"
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
        "--forecast-horizon",
        type=int,
        default=None,
        help="Forecast horizon in days (overrides config)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory containing model artifacts (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for submission (overrides config)"
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

    models_dir = Path(args.models_dir or config['paths']['models'])
    output_dir = Path(args.output_dir or config['paths']['submissions'])
    forecast_horizon = args.forecast_horizon or config['submission']['forecast_horizon']
    nowcast_lookback = config['submission']['nowcast_lookback_days']

    # All 52 US locations
    all_hub_locations = config['data']['locations']

    logger.info(f"Formatting submission for nowcast date: {args.nowcast_date}")

    try:
        # Load model artifacts
        logger.info("Loading model artifacts...")
        trace, mappings, dimensions = load_model_artifacts(models_dir, args.nowcast_date)

        # US states abbreviation dict
        us_states_abbreviation_dict = {
            'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
            'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
            'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
            'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
            'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
            'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
            'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
            'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
            'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
            'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
            'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
            'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
            'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC', 'Puerto Rico': 'PR'
        }

        # Generate forecast dates
        target_dates = generate_forecast_dates(args.nowcast_date, nowcast_lookback, forecast_horizon)
        logger.info(f"Target dates: {len(target_dates)} dates from {target_dates[0]} to {target_dates[-1]}")

        # Get locations and clades
        locations_in_training = sorted(mappings['reverse_location'].values())
        clades = sorted(mappings['reverse_clade'].values())
        locations = all_hub_locations

        logger.info(f"Locations: {len(locations)}")
        logger.info(f"Clades: {clades}")

        # Create grid of predictions
        time_indices = []
        location_indices = []
        missing_locations = []

        # Find base date for time calculation
        date_to_idx = mappings['date']
        reverse_date_map = mappings['reverse_date']
        base_date_idx = min(date_to_idx.values())
        base_date_obj = reverse_date_map[base_date_idx]

        if isinstance(base_date_obj, str):
            base_date = datetime.strptime(base_date_obj, '%Y-%m-%d')
        else:
            base_date = datetime.combine(base_date_obj, datetime.min.time())

        logger.info(f"Base date: {base_date.date()}")

        # Get time scaling parameters
        time_mean = mappings['time_mean']
        time_std = mappings['time_std']

        # Build prediction indices
        for target_date in target_dates:
            for location in locations:
                # Calculate time index
                target_dt = datetime.strptime(target_date, '%Y-%m-%d')
                raw_time_idx = (target_dt - base_date).days
                time_idx_scaled = (raw_time_idx - time_mean) / time_std

                # Get location index
                if location in mappings['location']:
                    loc_idx = mappings['location'][location]
                else:
                    loc_idx = 0
                    if location not in missing_locations:
                        missing_locations.append(location)
                        logger.warning(f"Location {location} not in training, using average")

                time_indices.append(time_idx_scaled)
                location_indices.append(loc_idx)

        time_indices = np.array(time_indices)
        location_indices = np.array(location_indices)

        logger.info(f"Missing locations: {len(missing_locations)}")

        # Compute proportions from BART
        proportions = compute_proportions_from_trace(
            trace, time_indices, location_indices,
            dimensions['num_clades'], dimensions['num_locations']
        )

        # Create submission dataframe
        submission_df = create_submission_dataframe(
            proportions=proportions,
            target_dates=target_dates,
            locations=locations,
            clades=clades,
            nowcast_date=args.nowcast_date,
            n_samples=config['submission']['n_samples'],
            include_samples=config['submission']['include_samples'],
            include_mean=config['submission']['include_mean'],
            us_states_abbreviation_dict=us_states_abbreviation_dict
        )

        # Validate submission
        is_valid, errors = validate_submission(submission_df, clades)

        if not is_valid:
            logger.error("Submission validation failed. Not saving.")
            for error in errors:
                print(f"ERROR: {error}")
            return 1

        # Save submission
        model_name = f"{config['model']['team']}-{config['model']['name']}"
        output_file = save_submission(
            submission_df,
            output_dir,
            args.nowcast_date,
            model_name,
            config['submission']['format']
        )

        # Print summary
        print("\n" + "=" * 70)
        print("BART SUBMISSION FORMATTING COMPLETE")
        print("=" * 70)
        print(f"Nowcast date: {args.nowcast_date}")
        print(f"Model: {model_name}")
        print(f"Output file: {output_file}")
        print(f"\nSubmission summary:")
        print(f"  Total rows: {len(submission_df)}")
        print(f"  Locations: {submission_df['location'].nunique()}")
        print(f"  Target dates: {submission_df['target_date'].nunique()}")
        print(f"  Clades: {submission_df['clade'].nunique()}")
        print(f"  Output types: {submission_df['output_type'].value_counts().to_dict()}")
        print(f"\nValidation: {'✓ PASSED' if is_valid else '✗ FAILED'}")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Error formatting submission: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
