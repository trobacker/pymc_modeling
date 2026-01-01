# PyMC BART Variant Nowcast Model

Bayesian Additive Regression Trees (BART) for COVID-19 variant nowcasting with flexible non-parametric modeling.

## Overview

BART automatically learns complex non-linear temporal patterns through tree ensembles, capturing sharp variant transitions better than linear models. Each clade gets its own BART ensemble with location-time feature interactions.

**Key Features:**
- Non-parametric: No functional form assumptions
- Automatic interactions: Trees capture location-time patterns
- Dirichlet-Multinomial: Optional overdispersion for uncertainty calibration
- Full Bayesian: NUTS + PGBART hybrid sampling

**When to use BART vs HMLR:**
- ✅ BART: Sharp transitions, complex dynamics, non-linear patterns
- ✅ HMLR: Smooth trends, interpretable coefficients, faster iteration

## Quick Start

```bash
# 1. Setup (once)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run workflow
cd scripts
python run_workflow.py --nowcast-date 2024-11-13 --mode test
```

This fetches data from S3, fits BART (10-30 min), and generates hub submission.

## Installation

### First-Time Setup

```bash
# From pymc_bart directory
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Verify
python -c "import pymc; import pymc_bart; print('✓ Ready!')"
```

### Each Session

```bash
source .venv/bin/activate  # Always activate before running
cd scripts
python run_workflow.py --nowcast-date 2024-11-13 --mode test
```

**Troubleshooting:** If you see `ModuleNotFoundError: No module named 'pymc_bart'`, you forgot to activate the `.venv` or need to run `pip install -r requirements.txt`.

## Usage

### Complete Workflow

```bash
cd scripts
python run_workflow.py --nowcast-date 2024-11-13 --mode test
```

**Modes:**
- `test`: 1000 draws, 500 warmup (~10-30 min)
- `prod`: 3000 draws, 1000 warmup (~1-3 hours)

### Individual Steps

```bash
# 1. Fetch data (150 days lookback)
python 01_fetch_data.py --nowcast-date 2024-11-13

# 2. Fit model
python 02_fit_model.py --nowcast-date 2024-11-13 --mode test

# 3. Format submission
python 03_format_submission.py --nowcast-date 2024-11-13

# 4. Plot (optional)
python plot_submission.py --nowcast-date 2024-11-13 --location CA
```

## Configuration

Edit `config.yaml`:

```yaml
modeling:
  mode: "test"        # or "prod"
  n_trees: 50         # Trees per clade
  n_draws_test: 1000  # MCMC samples
  chains: 2

data:
  training_lookback_days: 150
  min_sequences: 5
```

**Key parameters:**
- `n_trees`: More = flexible but slower (20-200)
- `n_draws`: More = better inference but slower
- `training_lookback_days`: Historical context (60-180)

## Model Architecture

```
For each clade k:
  η_k ~ BART(time, location)  [50-tree ensemble]

Proportions:
  p = softmax([η_1, ..., η_{K-1}, 0])

Observation:
  θ ~ Dirichlet(concentration_location * p)
  Y ~ Multinomial(n, θ)
```

**Features:**
- Time: Centered/scaled date indices
- Location: One-hot encoded (52 US locations)
- Concentration: Location-specific (hierarchical prior)

## Output

**Submission format:**
- `submissions/YYYY-MM-DD-YourTeam-PyMC-BART.parquet`
- 100 samples + mean per task
- 32-day nowcast + 10-day forecast

**Model artifacts:**
- `data/training_data_*.parquet` - Processed training data
- `fitted/trace_*.nc` - MCMC samples
- `logs/` - Execution logs

## Documentation

- **README.md** (this file): Quick start and usage
- **PYMC_MODELING_GUIDE.md**: Technical deep dive (PyMC details, priors, sampling)
- **requirements.txt**: Dependencies

## Performance

**Test mode:**
- Time: 10-30 minutes
- Memory: 2-4 GB
- Good for: Iteration, testing

**Production mode:**
- Time: 1-3 hours
- Memory: 4-8 GB
- Good for: Final submissions

**Too slow?** Reduce `n_trees` (50→20) or `training_lookback_days` (150→60) in `config.yaml`.

## Comparison: BART vs HMLR

| Aspect | BART | HMLR |
|--------|------|------|
| Flexibility | High (non-parametric) | Medium (linear) |
| Speed | Slower (trees) | Faster (gradient) |
| Sharp transitions | Excellent | Limited |
| Interpretability | Low (black box) | High (coefficients) |
| Use case | Complex dynamics | Smooth trends |

## Dependencies

```
pymc>=5.20.1           # Bayesian modeling
pymc-bart>=0.6.0       # BART implementation
numpyro>=0.13.0        # Fast sampling backend
polars>=0.20.0         # Data handling
arviz>=0.20.0          # Diagnostics
```

See `requirements.txt` for complete list.

## Troubleshooting

**"No module named 'pymc_bart'"**
- Activate environment: `source .venv/bin/activate`
- Install: `pip install -r requirements.txt`

**Slow sampling**
- Reduce `n_trees` in config.yaml (50→20)
- Use test mode instead of prod
- Reduce `training_lookback_days` (150→60)

**Out of memory**
- Reduce `n_trees` (50→10)
- Reduce `chains` (2→1)
- Filter more data: increase `min_sequences` (5→20)

**Which environment am I using?**
```bash
which python  # Should show .../pymc_bart/.venv/bin/python
echo $VIRTUAL_ENV  # Should show .../pymc_bart/.venv
```

## Model-Specific Environments

This model uses its own `.venv` to avoid conflicts:
- `pymc_bart` needs `pymc-bart` (other models don't)
- Different models can use different package versions
- Clean isolation per model

## References

- **BART**: Chipman, George, McCulloch (2010). "BART: Bayesian additive regression trees"
- **PyMC-BART**: https://github.com/pymc-devs/pymc-bart
- **Variant Nowcast Hub**: https://github.com/reichlab/variant-nowcast-hub
