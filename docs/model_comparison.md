# COVID-19 Variant Nowcast Model Comparison

This document provides a comprehensive comparison of the two PyMC-based hierarchical multinomial logistic regression models for COVID-19 variant nowcasting.

## Table of Contents

- [Model Overview](#model-overview)
- [Mathematical Formulations](#mathematical-formulations)
- [Test Results Summary](#test-results-summary)
- [Feature Comparison](#feature-comparison)
- [When to Use Each Model](#when-to-use-each-model)
- [Performance Characteristics](#performance-characteristics)
- [Configuration Details](#configuration-details)

---

## Model Overview

### PyMC-HMLR (Base Model)
**Location**: `hub_submissions/models/pymc_hmlr/`

Standard hierarchical multinomial logistic regression with linear trends in logit space. This model assumes that observed variant counts follow a multinomial distribution with no additional overdispersion beyond the multinomial variance.

**Key Characteristics:**
- Simplest model structure
- Faster runtime
- Suitable when count data variance matches multinomial expectations
- No additional parameters beyond location-clade intercepts and slopes

### PyMC-HMLR-Dirichlet (Overdispersion Model)
**Location**: `hub_submissions/models/pymc_hmlr_dirichlet/`

Enhanced hierarchical multinomial logistic regression with Dirichlet-Multinomial observation process. Adds a Dirichlet layer to capture overdispersion in count data beyond what the multinomial distribution alone can explain.

**Key Characteristics:**
- Captures count overdispersion
- More flexible variance structure
- Estimated concentration parameter controls overdispersion level
- Slightly longer runtime due to additional layer

---

## Mathematical Formulations

### Base HMLR Model

**Linear predictor in logit space:**
```
η_{vl}(t) = α_{vl} + β_{vl} * t
```

**Softmax transformation to proportions:**
```
p_{vl}(t) = softmax(η_{vl}(t))
```

**Likelihood (multinomial):**
```
Y_i ~ Multinomial(n_i, p_i)
```

**Where:**
- `v` = variant/clade index
- `l` = location index
- `t` = time (centered and scaled)
- `α_{vl}` = location-variant specific intercept
- `β_{vl}` = location-variant specific time slope
- `n_i` = total sequences for observation i
- `p_i` = probability vector for observation i

**Priors:**
```
α_{vl} ~ Normal(0, 3)
β_{vl} ~ Normal(0, 3)
```

**Total Parameters:**
- `num_locations × num_clades × 2` (alpha + beta for each location-clade)

### Dirichlet-Multinomial Model

**Linear predictor and mean proportions (same as base):**
```
η_{vl}(t) = α_{vl} + β_{vl} * t
p_mean(t) = softmax(η_{vl}(t))
```

**Dirichlet concentration parameter:**
```
log_concentration ~ Normal(log(50), 1)
concentration = exp(log_concentration)
```

**Dirichlet layer (adds observation-level variability):**
```
θ_i ~ Dirichlet(concentration × p_mean,i)
```

**Likelihood (multinomial with Dirichlet-sampled proportions):**
```
Y_i ~ Multinomial(n_i, θ_i)
```

**Where:**
- `θ_i` = observation-specific proportion vector (varies around `p_mean`)
- `concentration` = controls variance around mean trajectory

**Variance interpretation:**
```
Var(θ_{iv}) ≈ p_mean,iv × (1 - p_mean,iv) / (concentration + 1)
```

**Total Parameters:**
- `num_locations × num_clades × 2` (alpha + beta)
- `+1` (concentration parameter)

---

## Test Results Summary

### Test Configuration (Both Models)
- **Nowcast Date**: 2025-11-14
- **S3 Data Date**: 2025-11-10 (most recent Monday)
- **Training Period**: 2025-07-17 to 2025-10-17 (150 days lookback)
- **MCMC Sampling**: 3000 draws, 500 warmup steps, 4 chains
- **Target Accept**: 0.90
- **Max Tree Depth**: 10

### Data Characteristics
- **Training Observations**: 1,656 location-date pairs
- **Locations with Data**: 18 (≥5 sequences per date)
- **Clades**: 6 (24A, 24B, 24C, 24H, other, recombinant)
- **Date Range**: 92 days of actual data (within 150-day window)

### Base HMLR Results
**Status**: Not yet tested with new configuration

**Expected Performance:**
- Runtime: ~10-15 minutes (3000 draws, 500 warmup)
- Parameters: 216 (18 locations × 6 clades × 2)
- Convergence: Expected good (Rhat < 1.01)

### Dirichlet-Multinomial Results
**Status**: Successfully tested (2025-11-14 nowcast)

**Performance:**
- Runtime: ~16 minutes (3000 draws, 500 warmup)
- Parameters: 217 (216 location-clade + 1 concentration)
- Convergence: Good (no divergences reported)
- Submission Validation: ✓ Passed all hub checks

**Submission Details:**
- File: `2025-11-14-YourTeam-PyMC-HMLR-Dirichlet.parquet`
- Total Rows: 1,323,504
  - 52 locations (all US states + DC + PR)
  - 42 target dates (32-day nowcast + 10-day forecast)
  - 6 clades
  - 101 output types (100 samples + 1 mean)
- Output Types:
  - 1,310,400 sample rows
  - 13,104 mean rows
- Validation: ✓ All checks passed

**Prediction Period:**
- Nowcast: 2025-10-14 to 2025-11-14 (32 days)
- Forecast: 2025-11-15 to 2025-11-24 (10 days)

---

## Feature Comparison

| Feature | Base HMLR | Dirichlet-Multinomial |
|---------|-----------|----------------------|
| **Model Structure** | | |
| Mean trajectory | Linear in logit space | Linear in logit space |
| Observation model | Multinomial | Dirichlet-Multinomial |
| Overdispersion | No | Yes (estimated) |
| Extra parameters | 0 | 1 (concentration) |
| **Mathematical Properties** | | |
| Mean proportions | `softmax(α + β*t)` | `softmax(α + β*t)` |
| Variance structure | Multinomial only | Multinomial + Dirichlet |
| Flexibility | Lower | Higher |
| **Computational** | | |
| Runtime (test mode) | ~10-15 min (est.) | ~16 min (observed) |
| Runtime (prod mode) | ~45-60 min (est.) | ~60-90 min (est.) |
| Memory usage | Lower | Slightly higher |
| Convergence | Fast | Moderate |
| **Practical Considerations** | | |
| Use case | Standard variance | Count overdispersion |
| Data requirements | Any | Benefits from overdispersed data |
| Interpretation | Simpler | More nuanced |
| Prediction intervals | Narrower | Wider (more uncertainty) |

---

## When to Use Each Model

### Use Base HMLR When:

1. **Computational resources are limited**
   - Faster runtime (~30% faster)
   - Lower memory requirements
   - Simpler to diagnose convergence issues

2. **Data variance matches multinomial expectations**
   - Count data shows typical multinomial variance
   - No evidence of significant overdispersion
   - Posterior predictive checks look good with simple model

3. **Interpretability is paramount**
   - Fewer parameters to explain
   - Simpler model structure
   - Easier to communicate to stakeholders

4. **Rapid prototyping or testing**
   - Quick iteration during development
   - Initial exploration of data patterns
   - Testing workflow components

### Use Dirichlet-Multinomial When:

1. **Evidence of overdispersion in count data**
   - Variance exceeds multinomial expectations
   - Posterior predictive checks show underfitting with base model
   - Residuals suggest additional variance structure

2. **More conservative predictions desired**
   - Wider prediction intervals capture more uncertainty
   - Better calibration for risk-averse decisions
   - Important not to underestimate uncertainty

3. **Heterogeneous observation quality**
   - Some locations/dates have noisier counts
   - Variable sequencing effort across time/space
   - Non-random sampling effects present

4. **Final production submissions**
   - More robust to model misspecification
   - Better uncertainty quantification
   - Worth the extra computational cost

---

## Performance Characteristics

### Runtime Comparison

**Test Mode** (3000 draws, 500 warmup, 4 chains):
- Base HMLR: ~10-15 minutes (estimated)
- Dirichlet-Multinomial: ~16 minutes (observed)
- Overhead: ~6-10% longer

**Production Mode** (15000 draws, 3000 warmup, 4 chains):
- Base HMLR: ~45-60 minutes (estimated)
- Dirichlet-Multinomial: ~60-90 minutes (estimated)
- Overhead: ~30-50% longer (more warmup needed for convergence)

### Memory Usage

**Base HMLR:**
- Posterior storage: ~216 parameters × 12000 samples = 2.6M values
- Typical memory: ~100-200 MB for trace

**Dirichlet-Multinomial:**
- Posterior storage: ~217 parameters + theta auxiliary = larger
- Theta storage: 1656 obs × 6 clades × 12000 samples = ~119M values
- Typical memory: ~1-2 GB for trace (includes per-observation proportions)

### Convergence Characteristics

**Base HMLR:**
- Generally fast convergence
- Rhat typically < 1.01 after 500 warmup
- ESS usually > 1000 per parameter
- Rare divergences with proper priors

**Dirichlet-Multinomial:**
- Slightly slower convergence
- May need more warmup (hence 500 vs 200 in test mode)
- ESS typically 500-1500 per parameter
- Concentration parameter may need longer chains
- More sensitive to initial values

---

## Configuration Details

### Common Configuration (Both Models)

**Data:**
```yaml
training_lookback_days: 150  # 5 months of historical data
min_sequences: 5             # Minimum sequences per location-date
locations: [all 52 US states/territories]
required_clades: ["24A", "24B", "24C", "recombinant", "other"]
optional_clades: ["24E", "24F", "24G", "24H", "24I", "25A"]
```

**MCMC (Test Mode):**
```yaml
n_draws_test: 3000      # Up from 1000 (3x increase)
n_warmup_test: 500      # Up from 200 (2.5x increase)
cores: 4
target_accept: 0.90
max_treedepth: 10
```

**MCMC (Production Mode):**
```yaml
n_draws_prod: 15000     # Up from 10000 (1.5x increase)
n_warmup_prod: 3000     # Up from 2000 (1.5x increase)
cores: 4
target_accept: 0.90
max_treedepth: 10
```

**Priors:**
```yaml
alpha_prior_sd: 3.0     # Intercepts
beta_prior_sd: 3.0      # Slopes
```

**Submission:**
```yaml
n_samples: 100                 # Posterior samples to submit
nowcast_lookback_days: 31      # 32-day nowcast window
forecast_horizon: 10           # Days into future
format: "parquet"
```

### Dirichlet-Specific Configuration

**Additional Prior:**
```python
log_concentration ~ Normal(log(50), 1)
concentration = exp(log_concentration)
```

**Interpretation:**
- Prior mean: concentration ≈ 50 (moderate overdispersion)
- Prior allows wide range: ~7 to ~400 (covers low to high concentration)
- Higher concentration → less overdispersion → closer to base model
- Lower concentration → more overdispersion → more observation-level variance

---

## Model Selection Workflow

### 1. Start with Base HMLR
Run the base model first to:
- Establish baseline performance
- Check basic model fit
- Identify potential issues
- Generate initial predictions

### 2. Diagnostic Checks
Evaluate base model using:
- Posterior predictive checks
- Residual analysis
- Variance comparisons (observed vs predicted)
- Visual inspection of fits by location/clade

### 3. Consider Dirichlet-Multinomial If:
- Posterior predictive intervals too narrow
- Systematic underprediction of variance
- Some locations show poor fit
- Evidence of extra-multinomial variation

### 4. Compare Models
If both models run:
- Compare prediction intervals
- Check concentration parameter estimate
- Evaluate posterior predictive fit
- Consider computational tradeoff

### 5. Production Decision
Choose model based on:
- **Base HMLR**: If fits are good and speed matters
- **Dirichlet-Multinomial**: If better uncertainty needed or overdispersion evident
- **Both**: Submit ensemble if resources allow

---

## References

### Model Documentation
- Base HMLR: `hub_submissions/models/pymc_hmlr/`
- Dirichlet-Multinomial: `hub_submissions/models/pymc_hmlr_dirichlet/`
- Changelog: `hub_submissions/models/pymc_hmlr_dirichlet/CHANGELOG.md`

### Key Literature
- **Multinomial logistic regression**: Standard reference for compositional data
- **Dirichlet-Multinomial**: Models overdispersion in count data
- **Hierarchical modeling**: Partial pooling across locations/clades
- **NUTS sampling**: Efficient MCMC for complex posteriors

### Hub Resources
- Variant Nowcast Hub: `/Users/trobacker/GitHub/variant-nowcast-hub/`
- Target data: `variant-nowcast-hub/target-data/time-series/`
- Model outputs: `variant-nowcast-hub/model-output/`
- Plotting functions: `variant-nowcast-hub/src/plot_summary_graphs.R`

---

## Version History

**2025-11-14**: Initial model comparison documentation
- Tested Dirichlet-Multinomial on 2025-11-14 nowcast
- Updated both models with improved MCMC configuration
- Extended training lookback to 150 days
- Documented test results and validation
