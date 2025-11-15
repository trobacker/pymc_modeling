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
│   ├── pymc_hmlr_dirichlet/                     # Base Dirichlet-Multinomial
│   ├── pymc_hmlr_dirichlet_hierarchical/        # With hierarchical priors
│   ├── pymc_hmlr_dirichlet_loc_concentration/   # With location-specific concentration
│   └── [Model Directory Structure]
│       ├── config.yaml          # Model-specific configuration
│       ├── scripts/             # Model workflow scripts
│       │   ├── 01_fetch_data.py
│       │   ├── 02_fit_model.py
│       │   ├── 03_format_submission.py
│       │   ├── run_workflow.py
│       │   ├── plot_submission.py
│       │   └── generate_hub_plots.R
│       ├── data/                # Training data (not tracked)
│       ├── fitted/              # Fitted model artifacts (not tracked)
│       ├── submissions/         # Generated submission files (not tracked)
│       └── logs/                # Workflow logs (not tracked)
└── hub_output/                  # Mirrors VNH structure for final submissions
    └── YourTeam-[ModelName]/    # Team-Model format
        └── YYYY-MM-DD-YourTeam-[ModelName].parquet
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

## Model Variants

### Model Evolution

```
Base Dirichlet-Multinomial
    ↓
    + Hierarchical hyperpriors
    ↓
Hierarchical Dirichlet-Multinomial
    ↓
    + Location-specific concentration
    ↓
Location-Specific Concentration Model
```

### 1. PyMC-HMLR-Dirichlet (Base Dirichlet-Multinomial)

**Location:** `models/pymc_hmlr_dirichlet/`

**Mathematical Form:**
```
η_{i,v,l} = α_{v,l} + β_{v,l} * t_i
p_mean_i = softmax(η_i)
concentration ~ exp(Normal(log(50), 1))
θ_i ~ Dirichlet(concentration * p_mean_i)
Y_i ~ Multinomial(n_i, θ_i)
```

**Key Features:**
- Location-variant specific intercepts and slopes
- Linear trends in logit space
- Global concentration parameter for overdispersion
- Dirichlet-Multinomial observation process

**Parameters:** 217 (216 location-clade + 1 concentration)

**When to Use:**
- Standard nowcasting with overdispersion
- When computational efficiency is important
- Baseline for model comparison

**Test Results (2025-11-12):**
- Runtime: ~79 minutes (4 chains, 3000 draws, 500 warmup)
- Divergences: 54
- Convergence: Good (rhat < 1.01)

### 2. PyMC-HMLR-Dirichlet-Hierarchical

**Location:** `models/pymc_hmlr_dirichlet_hierarchical/`

**Mathematical Form:**
```
# Hierarchical hyperpriors for intercepts
mu_alpha ~ Normal(0, 3)
sigma_alpha ~ HalfNormal(3)
alpha[l,c] ~ Normal(mu_alpha, sigma_alpha)

# Hierarchical hyperpriors for slopes
mu_beta ~ Normal(0, 1)
sigma_beta ~ HalfNormal(1)
beta[l,c] ~ Normal(mu_beta, sigma_beta)

# Linear predictor and observation model
eta = alpha[locations] + beta[locations] * time
p_mean = softmax(eta)
concentration ~ exp(Normal(log(50), 1))
theta ~ Dirichlet(concentration * p_mean)
Y ~ Multinomial(n, theta)
```

**Key Features:**
- Partial pooling across locations and clades via hyperpriors
- Improved parameter estimation for locations with sparse data
- Global mean trajectories learned across all locations
- Same Dirichlet-Multinomial observation as base model

**Parameters:** 221 (216 location-clade + 4 hyperpriors + 1 concentration)

**When to Use:**
- When you want to borrow strength across locations
- For locations with limited historical data
- When you expect similar trends across locations

**Test Results (2025-11-12):**
- Runtime: ~60 minutes (1 chain, 3000 draws, 500 warmup)
- Divergences: 0 (excellent!)
- Convergence: Excellent

### 3. PyMC-HMLR-Dirichlet-Loc-Concentration

**Location:** `models/pymc_hmlr_dirichlet_loc_concentration/`

**Mathematical Form:**
```
# Hierarchical priors (same as above)
mu_alpha ~ Normal(0, 3)
sigma_alpha ~ HalfNormal(3)
alpha[l,c] ~ Normal(mu_alpha, sigma_alpha)
mu_beta ~ Normal(0, 1)
sigma_beta ~ HalfNormal(1)
beta[l,c] ~ Normal(mu_beta, sigma_beta)

# Location-specific concentration (NEW)
mu_concentration ~ Normal(log(50), 1)
sigma_concentration ~ HalfNormal(1)
log_concentration[l] ~ Normal(mu_concentration, sigma_concentration)
concentration[l] = exp(log_concentration[l])

# Observation model with location-specific concentration
eta = alpha[locations] + beta[locations] * time
p_mean = softmax(eta)
concentration_expanded = concentration[locations]
theta ~ Dirichlet(concentration_expanded * p_mean)
Y ~ Multinomial(n, theta)
```

**Key Features:**
- All features from hierarchical model
- **Location-specific overdispersion parameters**
- Adaptive uncertainty calibration per location
- Allows wider intervals for sparse-data locations (e.g., FL)
- Allows narrower intervals for dense-data locations (e.g., CA)

**Parameters:** 239 (216 location-clade + 4 hyperpriors + 18 location concentrations + 2 concentration hyperpriors)

**When to Use:**
- When uncertainty calibration varies by location
- To address heterogeneous data quality across locations
- When some locations show over/under-dispersed intervals

**Test Results (2025-11-12):**
- Runtime: ~60 minutes (1 chain, 3000 draws, 500 warmup)
- Divergences: 0 (excellent!)
- Convergence: Excellent

## Workflows

### Running a Model

Each model has a `run_workflow.py` script that orchestrates the complete pipeline:

```bash
cd models/pymc_hmlr_dirichlet_loc_concentration/scripts

# Quick test run (3000 MCMC draws)
python run_workflow.py --nowcast-date 2025-11-12 --mode test

# Production run (15000 MCMC draws)
python run_workflow.py --nowcast-date 2025-11-12 --mode prod

# Skip data fetching (use cached data)
python run_workflow.py --nowcast-date 2025-11-12 --mode test --skip-fetch
```

**Important:** `--nowcast-date` should be the Wednesday submission deadline date, NOT the current date.

### Individual Steps

You can also run individual steps:

```bash
# 1. Fetch training data
python 01_fetch_data.py --nowcast-date 2025-11-12

# 2. Fit PyMC model
python 02_fit_model.py --nowcast-date 2025-11-12 --mode test

# 3. Format submission
python 03_format_submission.py --nowcast-date 2025-11-12
```

### Generating Plots

**Python plots** (using matplotlib):
```bash
python plot_submission.py --submission ../submissions/2025-11-12-YourTeam-PyMC-HMLR-Dirichlet-Loc-Concentration.parquet
```

**Hub official plots** (using R):
```bash
# Update model_output_file in generate_hub_plots.R
Rscript generate_hub_plots.R
```

## Configuration

All models use a `config.yaml` file with the following structure:

### Common Settings

```yaml
data:
  training_lookback_days: 150      # Days of historical data
  min_sequences: 5                 # Minimum sequences per location-date

modeling:
  n_draws_test: 3000               # MCMC draws for testing
  n_warmup_test: 500               # Warmup draws for testing
  n_draws_prod: 15000              # MCMC draws for production
  n_warmup_prod: 3000              # Warmup draws for production
  chains: 1                        # Number of MCMC chains (1 for hierarchical)
  cores: 4                         # CPU cores for parallelization
  target_accept: 0.90              # Target acceptance rate
  max_treedepth: 10                # Maximum tree depth for NUTS

submission:
  n_samples: 100                   # Samples to submit (required by hub)
  nowcast_lookback_days: 31        # Nowcast window (32 days total)
  forecast_horizon: 10             # Days to forecast into future
```

### Model-Specific Parameters

**Base Dirichlet-Multinomial:**
- Uses 4 chains (independent sampling)
- Global concentration parameter

**Hierarchical Models:**
- Uses 1 chain (reduces computational cost)
- Hierarchical hyperpriors on alpha and beta

**Location-Specific Concentration:**
- Same as hierarchical
- Additional location-specific concentration parameters

## Model Comparison

| Feature | Base | Hierarchical | Loc-Concentration |
|---------|------|--------------|-------------------|
| **Partial Pooling** | ❌ | ✅ | ✅ |
| **Heterogeneous Uncertainty** | ❌ | ❌ | ✅ |
| **Parameters** | 217 | 221 | 239 |
| **Chains (Test)** | 4 | 1 | 1 |
| **Runtime (Test)** | ~79 min | ~60 min | ~60 min |
| **Divergences** | 54 | 0 | 0 |
| **Best For** | Baseline | Sparse data | Varying data quality |

## Output Files

### Model Artifacts (Not Tracked)

Each model generates the following artifacts in its directory:

```
models/[model_name]/
├── data/
│   └── training_data_YYYY-MM-DD.parquet
├── fitted/
│   ├── trace_YYYY-MM-DD.nc              # MCMC trace
│   ├── posterior_predictive_YYYY-MM-DD.nc
│   └── mappings_YYYY-MM-DD.pkl          # Index mappings
└── submissions/
    └── YYYY-MM-DD-YourTeam-[ModelName].parquet
```

### Submission Format

Submissions follow the Variant Nowcast Hub schema:

- **reference_date**: Date data was pulled (most recent Monday ≤ nowcast_date)
- **target**: `wk flu hosp inc` (not actually flu, but follows hub format)
- **horizon**: Days relative to reference_date (-31 to +10)
- **target_end_date**: Specific date being forecasted
- **location**: US state/territory abbreviation
- **clade**: Variant clade name
- **output_type**: `sample` or `mean`
- **output_type_id**: Sample index (0-99) for samples, NA for mean
- **value**: Predicted proportion [0, 1]

## Requirements

- Python 3.9+
- PyMC 5.20+
- ArviZ 0.20+
- Polars (data manipulation)
- NumPy, Pandas
- R 4.0+ (for hub plots)
  - arrow, dplyr packages

## Troubleshooting

### Divergences
- Base model: 54 divergences (acceptable, monitor rhat)
- Hierarchical models: 0 divergences (excellent convergence)
- If you see divergences, increase `target_accept` to 0.95

### Runtime
- Test mode: ~60 minutes (3000 draws, 1 chain)
- Production mode: ~5 hours (15000 draws, 1 chain)
- Use multiple cores to speed up internal computations

### Memory
- Typical memory usage: ~500 MB during sampling
- Posterior predictive generation may spike to ~2 GB
- Reduce `n_draws` if you encounter memory issues

### Validation Errors
- Check that all required clades are present
- Ensure output values are in [0, 1]
- Verify submission has exactly 100 samples per location-date-clade
- Check that target_end_date ranges from nowcast-31 to nowcast+10

## References

- [Variant Nowcast Hub](https://github.com/reichlab/variant-nowcast-hub)
- [PyMC Documentation](https://www.pymc.io/)
- [Dirichlet-Multinomial Distribution](https://en.wikipedia.org/wiki/Dirichlet-multinomial_distribution)
- [Hierarchical Bayesian Models](https://en.wikipedia.org/wiki/Bayesian_hierarchical_modeling)
