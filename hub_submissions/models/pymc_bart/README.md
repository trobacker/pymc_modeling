# PyMC BART Variant Nowcast Model

Bayesian Additive Regression Trees (BART) implementation for COVID-19 variant proportion nowcasting and forecasting.

## Overview

This model uses **BART** - a flexible non-parametric Bayesian approach - to model variant proportions over time and across locations. Unlike hierarchical multinomial logistic regression (HMLR), BART automatically learns complex non-linear temporal patterns through an ensemble of regression trees.

### Key Features

- **Non-parametric flexibility**: BART learns arbitrary time trends without specifying functional form
- **Automatic feature interaction**: Trees naturally capture location-time interactions
- **Built-in regularization**: Sum-of-trees prior provides adaptive shrinkage
- **Uncertainty quantification**: Full Bayesian posterior for predictions
- **Dirichlet-Multinomial**: Optional overdispersion layer for calibrated uncertainty

### Model Architecture

```
For each clade k (K-1 parameterization):
  η_k ~ BART(time, location)  [ensemble of regression trees]

Proportions:
  p = softmax([η_1, η_2, ..., η_{K-1}, 0])

Observation model (optional Dirichlet layer):
  θ ~ Dirichlet(concentration * p)
  Y ~ Multinomial(n, θ)
```

**BART advantages over HMLR:**
- Captures sharp variant transitions (e.g., emergence, displacement)
- No need to specify hierarchical structure explicitly
- Handles heterogeneous temporal patterns across locations
- More robust to outliers via tree splits

**BART considerations:**
- More computationally expensive (tree sampling via PGBART)
- Less interpretable than linear models (black box)
- Requires more careful tuning of tree parameters

## Directory Structure

```
pymc_bart/
├── config.yaml              # Model configuration
├── README.md               # This file
├── scripts/
│   ├── 01_fetch_data.py    # Fetch training data from S3
│   ├── 02_fit_model.py     # Fit BART model
│   ├── 03_format_submission.py  # Format hub submission
│   ├── plot_submission.py  # Visualization (optional)
│   └── run_workflow.py     # Full workflow orchestration
├── data/                   # Training data (gitignored)
├── fitted/                 # Model artifacts (gitignored)
├── logs/                   # Logs (gitignored)
└── submissions/            # Formatted submissions (gitignored)
```

## Installation

To run this model locally, follow these steps from this directory:

```bash
# Create virtual environment
python -m venv .venv

# Activate environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
python -m pip install -r requirements.txt

# Verify installation
python -c "import pymc; import pymc_bart; print('✓ Environment ready!')"
```

The first model run will take longer as BART fits tree ensembles (~10-30 minutes in test mode).

## Usage

### Quick Start

Run the complete workflow for a specific nowcast date:

```bash
cd scripts
python run_workflow.py --nowcast-date 2024-11-13 --mode test
```

This will:
1. Fetch training data from S3 (150 days lookback)
2. Fit BART model (test mode: 1000 draws, 500 warmup)
3. Generate submission file with 100 samples + mean

### Step-by-Step

**1. Fetch training data:**
```bash
python 01_fetch_data.py --nowcast-date 2024-11-13
```

**2. Fit BART model:**
```bash
# Test mode (fast, fewer samples)
python 02_fit_model.py --nowcast-date 2024-11-13 --mode test

# Production mode (slower, more samples)
python 02_fit_model.py --nowcast-date 2024-11-13 --mode prod
```

**3. Format submission:**
```bash
python 03_format_submission.py --nowcast-date 2024-11-13
```

**4. (Optional) Visualize results:**
```bash
python plot_submission.py --nowcast-date 2024-11-13
```

### Advanced Options

**Custom configuration:**
```bash
python run_workflow.py --nowcast-date 2024-11-13 --config custom_config.yaml
```

**Skip steps (use existing artifacts):**
```bash
# Skip data fetching
python run_workflow.py --nowcast-date 2024-11-13 --skip-data

# Skip both data and model (only format submission)
python run_workflow.py --nowcast-date 2024-11-13 --skip-data --skip-model
```

**Custom forecast horizon:**
```bash
python 03_format_submission.py --nowcast-date 2024-11-13 --forecast-horizon 14
```

## Configuration

Edit `config.yaml` to customize model behavior:

### Key Parameters

**BART-specific:**
- `n_trees`: Number of trees in ensemble (default: 50)
  - More trees = more flexible, but slower
  - Typical range: 20-200
- `n_particles`: PGBART particles (default: 10)
  - More particles = better mixing, but slower
  - Typical range: 5-20

**Sampling:**
- `n_draws_test`: 1000 (faster for testing)
- `n_draws_prod`: 3000 (more robust for production)
- `chains`: 2 (BART is expensive per chain)
- `cores`: 4

**Overdispersion:**
- `use_dirichlet`: true (recommended for calibrated uncertainty)
- `concentration_init`: 50 (higher = tighter around mean)

**Data:**
- `training_lookback_days`: 150 (5 months of training data)
- `min_sequences`: 5 (filter low-count observations)

## Model Interpretation

### BART Components

Each clade has its own BART model:
- **Trees**: Each tree captures a specific pattern (e.g., "California after day 100")
- **Ensemble**: Sum of trees provides flexible function approximation
- **Regularization**: Prior on tree depth and number of terminal nodes prevents overfitting

### Feature Engineering

BART receives:
- **Time**: Centered/scaled date indices
- **Location**: One-hot encoded (50 states + DC + PR)

The trees automatically learn interactions like:
- "Clade X grows rapidly in location Y after time T"
- "Clade Z plateaus everywhere after time T'"

### Prediction Strategy

For new times/locations:
- **In-sample**: Direct BART evaluation
- **Extrapolation**: Trees extend patterns linearly beyond training range
- **Missing locations**: Average across observed locations

## Output Format

Submissions follow variant-nowcast-hub specifications:

**Columns:**
- `nowcast_date`: Submission date
- `target_date`: Date being predicted
- `clade`: Variant clade name
- `location`: 2-letter state code
- `output_type`: "sample" or "mean"
- `output_type_id`: Sample identifier (for samples)
- `value`: Predicted proportion [0, 1]

**Structure:**
- 100 samples per task (location × date × clade)
- Mean predictions for each task
- Proportions sum to 1 within each sample

## Diagnostics

### During Fitting

Monitor for:
- **Divergences**: Should be 0 (BART rarely diverges)
- **Tree depth**: Check if hitting max depth (increase if needed)
- **Acceptance rate**: BART uses PGBART, not NUTS (no target_accept)

### Trace Inspection

```python
import arviz as az

# Load trace
trace = az.from_netcdf("fitted/trace_2024-11-13.nc")

# Summary statistics
print(az.summary(trace))

# Trace plots
az.plot_trace(trace, var_names=['mu_concentration', 'sigma_concentration'])

# Posterior predictive check
az.plot_ppc(trace, group='posterior_predictive')
```

### Submission Validation

The workflow automatically validates:
- All required clades present
- Values in [0, 1]
- Proportions sum to 1
- Exactly 100 samples per task

## Comparison with HMLR

| Aspect | BART | HMLR |
|--------|------|------|
| **Flexibility** | High (non-parametric) | Medium (linear in logit) |
| **Interpretability** | Low (black box) | High (coefficients) |
| **Speed** | Slower (tree sampling) | Faster (gradient-based) |
| **Transitions** | Captures sharp changes | Smooth trends |
| **Extrapolation** | Linear tree extension | Linear logit trends |
| **Tuning** | Trees, particles | Priors, concentration |

**When to use BART:**
- Expect non-linear variant dynamics
- Care more about prediction than interpretation
- Have sufficient compute resources
- Want automatic feature engineering

**When to use HMLR:**
- Need interpretable coefficients
- Want faster iteration
- Prefer smooth extrapolations
- Have strong prior beliefs about trends

## Troubleshooting

**Memory issues:**
- Reduce `n_trees` (e.g., 50 → 20)
- Reduce `chains` (e.g., 4 → 2)
- Filter data more aggressively (`min_sequences`)

**Slow sampling:**
- Use test mode initially
- Reduce `n_particles` (10 → 5)
- Use fewer chains with more cores per chain

**Poor predictions:**
- Increase `n_trees` for more flexibility
- Extend `training_lookback_days` for more history
- Tune `concentration_init` for better calibration

**Divergences (rare for BART):**
- Check data preprocessing (outliers?)
- Ensure proportions sum to 1
- Try reducing `use_dirichlet` to false

## References

- **BART**: Chipman, George, McCulloch (2010). "BART: Bayesian additive regression trees"
- **PyMC-BART**: https://github.com/pymc-devs/pymc-bart
- **PGBART**: Lakshminarayanan & Roy (2015). "Particle Gibbs for Bayesian Additive Regression Trees"
- **Variant Nowcast Hub**: https://github.com/reichlab/variant-nowcast-hub

## License

Same as parent repository.

## Contact

For questions about this BART implementation, consult the PyMC forum or open an issue in the repository.
