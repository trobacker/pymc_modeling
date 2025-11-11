# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a PyMC modeling repository focused on Bayesian statistical modeling using PyMC. The repository contains implementations of various probabilistic models, with a particular emphasis on COVID-19 variant modeling using hierarchical multinomial regression.

## Project Structure

```
pymc_modeling/
├── basic_examples/          # Introductory PyMC examples
│   ├── basic_examples.ipynb    # Linear regression with PyMC
│   └── count_models.ipynb      # Count-based models
├── variant_models/          # COVID-19 variant modeling
│   ├── pymc_covid_variant_models.ipynb  # Main multinomial variant model
│   ├── s3_hubdata.ipynb                 # S3 data access for COVID variant hub
│   ├── counts_2025-02-19.tsv.gz        # Variant count data
│   └── metadata.csv                     # Location and clade metadata
```

## Environment Setup

This project uses a conda environment called `asper_pymc`. Key dependencies:
- PyMC 5.20.1+ (probabilistic programming framework)
- ArviZ 0.20.0+ (Bayesian visualization and diagnostics)
- NumPy (numerical computing)
- Polars (fast data manipulation)
- Matplotlib/Seaborn (visualization)
- PyTensor (backend for PyMC)

To run notebooks, ensure you're in the `asper_pymc` environment as indicated in notebook headers.

## Model Architecture

### Basic Linear Models (`basic_examples/`)

Standard Bayesian linear regression:
- Normal priors on regression coefficients
- HalfNormal prior on error variance
- NUTS sampler with 4 chains
- Posterior predictive checking with ArviZ

### Hierarchical Multinomial Variant Model (`variant_models/pymc_covid_variant_models.ipynb`)

A sophisticated spatiotemporal model for COVID-19 variant proportions:

**Mathematical Form:**
```
η_i = α_{vl} + β_{vl} * t
p_i = softmax(η_i)
Y_i ~ Multinomial(n_i, p_i)
```

Where:
- `v` = variant/clade index
- `l` = location index
- `t` = time index
- `α_{vl}` = location-variant specific intercepts
- `β_{vl}` = location-variant specific time trends

**Key Implementation Details:**
- Uses K-1 parameterization for K variants (reference category added post-softmax)
- Softmax link ensures probabilities sum to 1
- Normal priors on α and β: N(0, 3)
- Shape: (num_locations, num_clades-1) for both α and β
- Data preprocessing creates pivot tables with multinomial count vectors

**Model Configuration:**
- Typically uses 1000 draws with 200 tuning steps for quick tests
- For production: increase to 10000 draws, 2000 tuning steps
- Monitor for divergences and tree depth warnings (increase `target_accept` if needed)
- Check rhat < 1.01 and effective sample size > 100

## Data Preprocessing

Variant modeling requires specific data format:
1. Raw data: location, clade, date, sequences (counts)
2. Convert categorical variables to integers using `replace_string_with_int()`
3. Pivot to create multinomial count vectors: shape (n_observations, n_clades)
4. Extract total counts per observation for Multinomial likelihood

Example structure:
```python
# Each row needs:
time: [date_index]
locations: [location_index]
Y: [[count_clade_0, count_clade_1, ..., count_clade_K]]
total_counts: [sum_of_counts_per_observation]
```

## Data Sources

### COVID Variant Hub (S3 Access)
Access via `s3_hubdata.ipynb`:
- Model outputs: `s3://covid-variant-nowcast-hub/model-output/*/*.parquet`
- Oracle data: `s3://covid-variant-nowcast-hub/target-data/oracle-output/*/*.parquet`
- Use Polars `scan_parquet()` with `storage_options={"skip_signature": "true"}`
- Supports streaming collection for large datasets

### Local Data
- `counts_2025-02-19.tsv.gz`: Aggregated variant counts by location/date/clade
- `metadata.csv`: Processed metadata with location and clade information

## Common Patterns

### PyMC Model Structure
```python
with pm.Model() as model:
    # Define priors
    param = pm.Normal('param', mu=0, sigma=1)

    # Define likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=data)

    # Sample
    trace = pm.sample(draws, tune=warmup, cores=4)

    # Posterior predictive
    post_pred = pm.sample_posterior_predictive(trace)
```

### Posterior Analysis Workflow
1. Check trace diagnostics: `pm.summary(trace)` - verify rhat and ESS
2. Visual diagnostics: `az.plot_trace(trace)`
3. Posterior predictive checks: `az.plot_ppc(post_pred, kind='cumulative')`
4. Extract samples: `az.extract(idata)` for stacked chain/draw dimensions

### Working with InferenceData
PyMC returns ArviZ InferenceData objects with groups:
- `posterior`: MCMC samples of parameters
- `posterior_predictive`: Samples from posterior predictive distribution
- `observed_data`: Original observed data
- `sample_stats`: Sampler diagnostics

Access via: `idata.posterior['param_name']` or `idata.posterior_predictive['obs_name']`

## Debugging Notes

- **Divergences**: Increase `target_accept` (e.g., 0.95) or reparameterize
- **Max tree depth**: Increase `max_treedepth` or simplify model
- **High rhat (>1.01)**: Run longer chains or check for multimodality
- **Low ESS (<100)**: Increase number of draws or improve sampling efficiency
- **Numerical issues in softmax**: Apply double softmax as in variant model to avoid probability > 1.0 errors

## US States Reference

The variant models include mappings for all 50 US states:
- Full names list: `us_states_full_names`
- Abbreviation dictionary: `us_states_abbreviation_dict`

Used for filtering and aggregating variant data by state.
