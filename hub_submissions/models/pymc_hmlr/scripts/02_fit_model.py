#!/usr/bin/env python3
"""
Fit PyMC Hierarchical Multinomial Model
=============================================================================

This script fits the hierarchical multinomial logistic regression model
to the training data using PyMC.

Prerequisites:
    Activate the virtual environment first:
    $ source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows

Usage:
    python 02_fit_model.py --nowcast-date 2024-11-13
    python 02_fit_model.py --nowcast-date 2024-11-13 --mode prod

Author: Auto-generated workflow
Date: 2025-11-14
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys
import warnings

import polars as pl
import numpy as np
import pymc as pm
import pytensor as pt
import arviz as az
import yaml

warnings.filterwarnings('ignore')


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


def replace_string_with_int(df: pl.DataFrame, column: str) -> pl.DataFrame:
    """
    Convert string values in a column to sequential integer indices.

    Returns both the DataFrame and the mapping dict for later decoding.
    """
    unique_strings = df[column].unique().to_list()
    string_to_int = {s: i for i, s in enumerate(unique_strings)}
    df_encoded = df.with_columns(pl.col(column).replace(string_to_int).alias(column))
    return df_encoded, string_to_int


def prepare_data_for_modeling(df):
    """
    Prepare data for PyMC modeling.

    Returns:
        tuple: (time, locations, Y, total_counts, mappings, dimensions)
    """
    logger = logging.getLogger(__name__)
    logger.info("Preparing data for modeling...")

    # Integer encode categorical variables
    data, clade_mapping = replace_string_with_int(df, 'clade')
    data, location_mapping = replace_string_with_int(data, 'location')
    data, date_mapping = replace_string_with_int(data, 'target_date')

    # Cast to integers
    data = data.with_columns([
        pl.col("clade").cast(pl.Int64),
        pl.col("location").cast(pl.Int64),
        pl.col("target_date").cast(pl.Int64)
    ])

    # Pivot to wide format for multinomial
    pivot_df = data.pivot(
        values="sequences",
        on="clade",
        index=["location", "target_date"],
        aggregate_function="sum"
    )

    pivot_df = pivot_df.fill_null(0)

    # Create count vectors
    clade_cols = pivot_df.columns[2:]
    pivot_df = pivot_df.with_columns([
        pl.concat_list(clade_cols).alias("clade_counts")
    ])

    pivot_df = pivot_df.with_columns([
        pl.sum_horizontal(clade_cols).alias("total_counts")
    ])

    # Extract arrays for PyMC
    time_raw = pivot_df.get_column('target_date').to_numpy()
    locations = pivot_df.get_column('location').to_numpy()
    Y = np.vstack(pivot_df.get_column('clade_counts').to_numpy())
    total_counts = pivot_df.get_column('total_counts').to_numpy()

    # Center and scale time for better numerical stability and interpretation
    # This makes the intercept (alpha) interpretable at the midpoint of the data
    time_mean = time_raw.mean()
    time_std = time_raw.std()
    time = (time_raw - time_mean) / time_std

    logger.info(f"Time variable: mean={time_mean:.2f}, std={time_std:.2f}")
    logger.info(f"Time range (centered): {time.min():.2f} to {time.max():.2f}")

    # Get dimensions
    num_locations = data['location'].n_unique()
    num_clades = data['clade'].n_unique()
    num_observations = len(pivot_df)

    # Create reverse mappings
    reverse_location_mapping = {v: k for k, v in location_mapping.items()}
    reverse_clade_mapping = {v: k for k, v in clade_mapping.items()}
    reverse_date_mapping = {v: k for k, v in date_mapping.items()}

    mappings = {
        'location': location_mapping,
        'clade': clade_mapping,
        'date': date_mapping,
        'reverse_location': reverse_location_mapping,
        'reverse_clade': reverse_clade_mapping,
        'reverse_date': reverse_date_mapping,
        'time_mean': float(time_mean),
        'time_std': float(time_std)
    }

    dimensions = {
        'num_locations': num_locations,
        'num_clades': num_clades,
        'num_observations': num_observations
    }

    logger.info(f"Data prepared:")
    logger.info(f"  Observations: {num_observations}")
    logger.info(f"  Locations: {num_locations}")
    logger.info(f"  Clades: {num_clades}")
    logger.info(f"  Y shape: {Y.shape}")

    return time, locations, Y, total_counts, mappings, dimensions


def build_and_fit_model(time, locations, Y, total_counts, dimensions, config):
    """
    Build and fit the hierarchical multinomial model.

    Returns:
        tuple: (model, trace, posterior_predictive)
    """
    logger = logging.getLogger(__name__)

    num_locations = dimensions['num_locations']
    num_clades = dimensions['num_clades']

    # Get modeling parameters
    mode = config['modeling']['mode']
    if mode == "prod":
        n_draws = config['modeling']['n_draws_prod']
        n_warmup = config['modeling']['n_warmup_prod']
    else:
        n_draws = config['modeling']['n_draws_test']
        n_warmup = config['modeling']['n_warmup_test']

    cores = config['modeling']['cores']
    target_accept = config['modeling']['target_accept']
    max_treedepth = config['modeling']['max_treedepth']
    alpha_sd = config['modeling']['alpha_prior_sd']
    beta_sd = config['modeling']['beta_prior_sd']

    logger.info(f"Building model (mode: {mode})...")
    logger.info(f"  Draws: {n_draws}, Warmup: {n_warmup}")
    logger.info(f"  Cores: {cores}, Target accept: {target_accept}")

    with pm.Model() as variant_model:
        # Priors for intercepts and slopes (one per location-clade combination)
        # These represent log-odds in the multinomial logistic regression
        alpha = pm.Normal('alpha', mu=0, sigma=alpha_sd,
                          shape=(num_locations, num_clades),
                          initval=np.random.randn(num_locations, num_clades) * 0.1)

        beta = pm.Normal('beta', mu=0, sigma=beta_sd,
                         shape=(num_locations, num_clades),
                         initval=np.random.randn(num_locations, num_clades) * 0.01)

        # Linear predictor in logit space: η = α + β * time
        # This creates linear trends in log-odds space
        eta = alpha[locations] + beta[locations] * time[:, None]

        # Convert from log-odds to probabilities using softmax
        # This is the standard inverse link for multinomial logistic regression
        # Softmax ensures: (1) all probabilities are positive, (2) they sum to 1
        # The linear structure in eta translates to linear trends in logit space
        p = pm.math.softmax(eta, axis=1)

        # Likelihood: multinomial distribution
        Y_obs = pm.Multinomial('Y_obs', n=total_counts, p=p, observed=Y)

        # Sample
        logger.info("Starting MCMC sampling...")
        trace = pm.sample(
            n_draws,
            tune=n_warmup,
            cores=cores,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            return_inferencedata=True
        )

        # Posterior predictive
        logger.info("Generating posterior predictive samples...")
        posterior_predictive = pm.sample_posterior_predictive(
            trace,
            var_names=['Y_obs']
        )

    logger.info("Model fitting complete!")

    # Check diagnostics
    n_divergences = trace.sample_stats['diverging'].sum().item()
    logger.info(f"Divergences: {n_divergences}")
    if n_divergences > 0:
        logger.warning(f"Model had {n_divergences} divergences. Consider adjusting parameters.")

    return variant_model, trace, posterior_predictive


def save_model_artifacts(trace, posterior_predictive, mappings, dimensions,
                          output_dir, nowcast_date):
    """Save model artifacts to disk"""
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trace
    trace_file = output_dir / f"trace_{nowcast_date}.nc"
    trace.to_netcdf(trace_file)
    logger.info(f"Trace saved to: {trace_file}")

    # Save posterior predictive
    pp_file = output_dir / f"posterior_predictive_{nowcast_date}.nc"
    posterior_predictive.to_netcdf(pp_file)
    logger.info(f"Posterior predictive saved to: {pp_file}")

    # Save mappings and dimensions
    import pickle
    mappings_file = output_dir / f"mappings_{nowcast_date}.pkl"
    with open(mappings_file, 'wb') as f:
        pickle.dump({'mappings': mappings, 'dimensions': dimensions}, f)
    logger.info(f"Mappings saved to: {mappings_file}")

    return trace_file, pp_file, mappings_file


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Fit PyMC hierarchical multinomial model"
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
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['test', 'prod'],
        default=None,
        help="Modeling mode: test or prod (overrides config)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing training data (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for model artifacts (overrides config)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger = setup_logging(config['logging']['level'])

    # Override config with command line args
    if args.mode:
        config['modeling']['mode'] = args.mode

    data_dir = Path(args.data_dir or config['paths']['data'])
    output_dir = Path(args.output_dir or config['paths']['models'])

    logger.info(f"Fitting model for nowcast date: {args.nowcast_date}")
    logger.info(f"Mode: {config['modeling']['mode']}")

    try:
        # Load training data
        training_file = data_dir / f"training_data_{args.nowcast_date}.parquet"
        if not training_file.exists():
            raise FileNotFoundError(
                f"Training data not found: {training_file}\n"
                f"Run 01_fetch_data.py first!"
            )

        logger.info(f"Loading training data from: {training_file}")
        df = pl.read_parquet(training_file)

        # Prepare data
        time, locations, Y, total_counts, mappings, dimensions = \
            prepare_data_for_modeling(df)

        # Fit model
        model, trace, posterior_predictive = build_and_fit_model(
            time, locations, Y, total_counts, dimensions, config
        )

        # Save artifacts
        trace_file, pp_file, mappings_file = save_model_artifacts(
            trace, posterior_predictive, mappings, dimensions,
            output_dir, args.nowcast_date
        )

        # Print summary
        print("\n" + "=" * 70)
        print("MODEL FITTING COMPLETE")
        print("=" * 70)
        print(f"Nowcast date: {args.nowcast_date}")
        print(f"Mode: {config['modeling']['mode']}")
        print(f"\nModel artifacts saved:")
        print(f"  Trace: {trace_file}")
        print(f"  Posterior predictive: {pp_file}")
        print(f"  Mappings: {mappings_file}")
        print(f"\nModel dimensions:")
        print(f"  Observations: {dimensions['num_observations']}")
        print(f"  Locations: {dimensions['num_locations']}")
        print(f"  Clades: {dimensions['num_clades']}")
        print(f"  Parameters: {dimensions['num_locations'] * dimensions['num_clades'] * 2} (alpha + beta for each location-clade)")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Error fitting model: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
