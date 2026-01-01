#!/usr/bin/env python3
"""
Fit PyMC BART Model for Variant Nowcasting
=============================================================================

This script fits a Bayesian Additive Regression Trees (BART) model
to the training data using PyMC. BART provides flexible non-parametric
modeling of variant proportions over time.

Key differences from hierarchical multinomial logistic regression:
- BART learns non-linear temporal patterns automatically via tree ensemble
- No explicit hierarchical structure; BART handles partial pooling through trees
- More flexible for capturing complex variant dynamics (e.g., sharp transitions)

Prerequisites:
    Activate the virtual environment first:
    $ source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows

Usage:
    python 02_fit_model.py --nowcast-date 2024-11-13
    python 02_fit_model.py --nowcast-date 2024-11-13 --mode prod

Author: Auto-generated workflow
Date: 2025-12-31
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
import pymc_bart as pmb
import pytensor.tensor as pt
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
    Prepare data for PyMC BART modeling.

    For BART, we need:
    - Features: time, location, potentially interaction terms
    - Target: clade proportions (via multinomial counts)

    Returns:
        tuple: (X, time, locations, Y, total_counts, mappings, dimensions)
    """
    logger = logging.getLogger(__name__)
    logger.info("Preparing data for BART modeling...")

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

    # Center and scale time for better numerical stability
    time_mean = time_raw.mean()
    time_std = time_raw.std()
    time = (time_raw - time_mean) / time_std

    logger.info(f"Time variable: mean={time_mean:.2f}, std={time_std:.2f}")
    logger.info(f"Time range (centered): {time.min():.2f} to {time.max():.2f}")

    # Create feature matrix X for BART
    # BART uses (n_observations, n_features) input
    # Features: time, location (one-hot encoded)
    num_locations = data['location'].n_unique()

    # One-hot encode locations
    location_onehot = np.zeros((len(locations), num_locations))
    location_onehot[np.arange(len(locations)), locations] = 1

    # Combine time and location features
    # Shape: (n_observations, 1 + n_locations)
    X = np.column_stack([time, location_onehot])

    logger.info(f"Feature matrix X shape: {X.shape}")

    # Get dimensions
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
        'num_observations': num_observations,
        'num_features': X.shape[1]
    }

    logger.info(f"Data prepared:")
    logger.info(f"  Observations: {num_observations}")
    logger.info(f"  Locations: {num_locations}")
    logger.info(f"  Clades: {num_clades}")
    logger.info(f"  Features: {X.shape[1]}")
    logger.info(f"  Y shape: {Y.shape}")

    return X, time, locations, Y, total_counts, mappings, dimensions


def build_and_fit_model(X, time, locations, Y, total_counts, dimensions, config):
    """
    Build and fit the BART model for variant nowcasting.

    Model structure:
    - For each clade k (in K-1 parameterization), fit BART to predict log-odds
    - BART learns f_k(time, location) automatically via ensemble of regression trees
    - Apply softmax to get proportions: p = softmax([f_1, f_2, ..., f_{K-1}, 0])
    - Likelihood: Y ~ Multinomial(n, p)

    Returns:
        tuple: (model, trace, posterior_predictive)
    """
    logger = logging.getLogger(__name__)

    num_locations = dimensions['num_locations']
    num_clades = dimensions['num_clades']
    num_observations = dimensions['num_observations']

    # Get modeling parameters
    mode = config['modeling']['mode']
    if mode == "prod":
        n_draws = config['modeling']['n_draws_prod']
        n_warmup = config['modeling']['n_warmup_prod']
    else:
        n_draws = config['modeling']['n_draws_test']
        n_warmup = config['modeling']['n_warmup_test']

    chains = config['modeling'].get('chains', 4)
    cores = config['modeling']['cores']
    target_accept = config['modeling']['target_accept']

    # BART-specific parameters
    n_trees = config['modeling'].get('n_trees', 50)  # Number of trees in ensemble
    n_particles = config['modeling'].get('n_particles', 10)  # PGBART particles

    logger.info(f"Building BART model (mode: {mode})...")
    logger.info(f"  Draws: {n_draws}, Warmup: {n_warmup}")
    logger.info(f"  Chains: {chains}, Cores: {cores}")
    logger.info(f"  BART trees: {n_trees}, Particles: {n_particles}")

    with pm.Model() as variant_model:
        # BART models for each clade (K-1 parameterization)
        # Each BART learns the log-odds for one clade vs reference
        # eta[i, k] = f_k(time[i], location[i])

        logger.info(f"Creating {num_clades} BART models (one per clade)...")

        # Store BART models for each clade
        eta_list = []

        for clade_idx in range(num_clades):
            logger.info(f"  Initializing BART for clade {clade_idx + 1}/{num_clades}")

            # BART regression tree ensemble
            # pmb.BART expects: X (features), Y (target), m (number of trees)
            # For multinomial, we'll use the observed proportions as pseudo-targets during initialization
            # The actual likelihood will be multinomial below

            # Initialize with zero (neutral log-odds)
            # BART will learn the function from data during sampling
            bart = pmb.BART(
                f"bart_{clade_idx}",
                X=X,
                Y=np.zeros(num_observations),  # Will be updated during sampling
                m=n_trees,
                shape=num_observations
            )

            eta_list.append(bart)

        # Stack BART outputs into matrix: (n_observations, n_clades)
        eta = pt.stack(eta_list, axis=1)

        logger.info("BART models created, applying softmax link...")

        # Convert from log-odds to proportions using softmax
        # This ensures valid probability simplex (positive, sum to 1)
        p_mean = pm.math.softmax(eta, axis=1)

        # Add Dirichlet layer for overdispersion (optional but recommended)
        # This allows observation-level variability beyond multinomial
        use_dirichlet = config['modeling'].get('use_dirichlet', True)

        if use_dirichlet:
            # Location-specific concentration parameters
            concentration_init = config['modeling'].get('concentration_init', 50)
            logger.info(f"Using Dirichlet-Multinomial with concentration ~{concentration_init}")

            # Hierarchical concentration (varies by location)
            mu_concentration = pm.Normal('mu_concentration',
                                         mu=np.log(concentration_init),
                                         sigma=2)
            sigma_concentration = pm.HalfNormal('sigma_concentration', sigma=2)

            log_concentration = pm.Normal('log_concentration',
                                          mu=mu_concentration,
                                          sigma=sigma_concentration,
                                          shape=num_locations)
            concentration = pm.math.exp(log_concentration)

            # Apply concentration by location
            concentration_expanded = concentration[locations]
            alpha_dirichlet = concentration_expanded[:, None] * p_mean

            # Dirichlet-distributed proportions
            theta = pm.Dirichlet('theta', a=alpha_dirichlet,
                                shape=(num_observations, num_clades))

            # Multinomial likelihood with Dirichlet proportions
            Y_obs = pm.Multinomial('Y_obs', n=total_counts, p=theta, observed=Y)
        else:
            logger.info("Using plain Multinomial (no Dirichlet overdispersion)")
            # Direct multinomial likelihood
            Y_obs = pm.Multinomial('Y_obs', n=total_counts, p=p_mean, observed=Y)

        # Sample using NUTS (BART parameters are sampled via PGBART internally)
        logger.info("Starting MCMC sampling with NUTS + PGBART...")
        logger.info("  (BART uses Particle Gibbs for tree structure updates)")

        trace = pm.sample(
            n_draws,
            tune=n_warmup,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            return_inferencedata=True,
            # BART-specific: use PGBART sampler for bart_* variables
            nuts_sampler="numpyro",  # NumPyro is recommended for BART
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
        description="Fit PyMC BART model for variant nowcasting"
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

    # Override config with command line args
    if args.mode:
        config['modeling']['mode'] = args.mode

    data_dir = Path(args.data_dir or config['paths']['data'])
    output_dir = Path(args.output_dir or config['paths']['models'])

    logger.info(f"Fitting BART model for nowcast date: {args.nowcast_date}")
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
        X, time, locations, Y, total_counts, mappings, dimensions = \
            prepare_data_for_modeling(df)

        # Fit model
        model, trace, posterior_predictive = build_and_fit_model(
            X, time, locations, Y, total_counts, dimensions, config
        )

        # Save artifacts
        trace_file, pp_file, mappings_file = save_model_artifacts(
            trace, posterior_predictive, mappings, dimensions,
            output_dir, args.nowcast_date
        )

        # Print summary
        print("\n" + "=" * 70)
        print("BART MODEL FITTING COMPLETE")
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
        print(f"  Features: {dimensions['num_features']}")
        print(f"  BART trees per clade: {config['modeling'].get('n_trees', 50)}")
        print("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"Error fitting model: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
