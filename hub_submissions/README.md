# Variant Nowcast Hub Submissions

This directory contains model implementations for COVID-19 variant nowcast submissions to the [Variant Nowcast Hub](https://github.com/reichlab/variant-nowcast-hub).

## Directory Structure

```
hub_submissions/
├── README.md                    # This file
├── common/                      # Shared utilities across all models
│   ├── __init__.py
│   ├── config_utils.py          # Configuration and logging
│   ├── hub_utils.py             # Hub-specific utilities (clades, S3 data)
│   └── validation_utils.py      # Submission validation
├── models/                      # Model implementations
│   ├── pymc_hmlr/              # Hierarchical Multinomial Logistic Regression
│   │   ├── config.yaml          # Model-specific configuration
│   │   ├── scripts/             # Model workflow scripts
│   │   │   ├── 01_fetch_data.py
│   │   │   ├── 02_fit_model.py
│   │   │   ├── 03_format_submission.py
│   │   │   ├── run_workflow.py
│   │   │   ├── plot_submission.py
│   │   │   └── generate_hub_plots.R
│   │   ├── data/                # Training data
│   │   ├── fitted/              # Fitted model artifacts (traces, posteriors)
│   │   ├── submissions/         # Generated submission files
│   │   └── logs/                # Workflow logs
│   ├── pymc_gp/                # Future: Gaussian Process model
│   └── experimental/           # Experimental approaches
└── hub_output/                 # Mirrors VNH structure for final submissions
    └── YourTeam-PyMC-HMLR/     # Team-Model format
        ├── 2025-11-14-YourTeam-PyMC-HMLR.parquet
        └── ...
```

## Design Philosophy

### Model Isolation
Each model variant has its own self-contained directory under `models/`. This enables:
- Parallel development of different modeling approaches
- Easy comparison between model variants
- Independent configuration and tuning per model
- Clear separation of concerns

### Shared Utilities
The `common/` directory provides utilities used across all models:
- **config_utils.py**: Configuration loading and logging setup
- **hub_utils.py**: Hub-specific functions (modeled clades, S3 data, date utilities)
- **validation_utils.py**: Submission validation against hub schema

### VNH Structure Compliance
The `hub_output/` directory follows the Variant Nowcast Hub's expected structure:
- One directory per model: `{TeamName}-{ModelName}/`
- Dated submission files: `YYYY-MM-DD-{TeamName}-{ModelName}.parquet`
- Easy to copy to VNH repository for pull requests

## Workflows

### Running a Model

Each model has a `run_workflow.py` script that orchestrates the complete pipeline:

```bash
cd models/pymc_hmlr/scripts

# Quick test run (1000 MCMC draws)
python run_workflow.py --nowcast-date 2025-11-14 --mode test

# Production run (10000 MCMC draws)
python run_workflow.py --nowcast-date 2025-11-14 --mode prod

# Skip data fetching (use cached data)
python run_workflow.py --nowcast-date 2025-11-14 --mode test --skip-fetch
```

### Individual Steps

You can also run individual steps:

```bash
# 1. Fetch training data
python 01_fetch_data.py --nowcast-date 2025-11-14

# 2. Fit PyMC model
python 02_fit_model.py --nowcast-date 2025-11-14 --mode test

# 3. Format submission
python 03_format_submission.py --nowcast-date 2025-11-14
```

### Generating Plots

**Python plots** (using matplotlib):
```bash
python plot_submission.py --submission ../submissions/2025-11-14-YourTeam-PyMC-HMLR.parquet
```

**Hub official plots** (using R):
```bash
Rscript generate_hub_plots.R
```

## Model Descriptions

### PyMC-HMLR (Hierarchical Multinomial Logistic Regression)

**Mathematical Form:**
```
η_{i,v,l} = α_{v,l} + β_{v,l} * t_i
p_i = softmax(η_i)
Y_i ~ Multinomial(n_i, p_i)
```

Where:
- `v` = variant/clade index
- `l` = location index
- `t` = time index (standardized)
- `α_{v,l}` = location-variant specific intercepts
- `β_{v,l}` = location-variant specific time trends (linear in logit space)

**Key Features:**
- Location-specific clade trajectories
- Linear trends in logit space (exponential in probability space)
- Bayesian uncertainty quantification via MCMC
- Hierarchical structure borrows strength across locations

## Requirements

- Python 3.9+
- PyMC 5.20+
- ArviZ 0.20+
- Polars (data manipulation)
- NumPy, Pandas
- R 4.0+ (for hub plots)

