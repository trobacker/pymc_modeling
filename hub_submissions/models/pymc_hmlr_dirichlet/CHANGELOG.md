# Dirichlet-Multinomial Model Changelog

## 2025-11-14: Initial Implementation + Parameter Updates

### Model Architecture

**New Dirichlet-Multinomial Observation Process:**
- Added Dirichlet layer between mean proportions and multinomial likelihood
- Captures overdispersion in count data beyond multinomial variance
- Estimated concentration parameter controls variance around mean trajectories

**Mathematical Structure:**
```
η = α + β * t                               # Linear predictor (as before)
p_mean = softmax(η)                         # Mean proportions (as before)
concentration ~ exp(Normal(log(50), 1))     # NEW: Estimated concentration
theta[i] ~ Dirichlet(concentration * p_mean[i])  # NEW: Per-observation proportions
Y ~ Multinomial(n, theta)                   # Observation with overdispersion
```

**Key Parameters:**
- `log_concentration`: Log-scale concentration parameter (prior: Normal(log(50), 1))
- `concentration`: Exp-transformed concentration (positive constraint)
- `theta`: Observation-level proportion vectors (shape: n_obs × n_clades)

**Interpretation:**
- Higher concentration → less variance → closer to mean trajectory p_mean
- Lower concentration → more variance → more overdispersion
- Variance: Var(theta) ≈ p_mean * (1 - p_mean) / (concentration + 1)

### Configuration Updates (Applied to Both Models)

**Training Data:**
- `training_lookback_days`: 120 → **150 days**
  - Captures 5 months of historical data instead of 4 months
  - Provides more context for learning seasonal patterns and longer-term trends
  - Improves estimation of location-specific parameters with more data points

**MCMC Sampling (Test Mode):**
- `n_draws_test`: 1000 → **3000 draws**
- `n_warmup_test`: 200 → **500 warmup steps**
- **Rationale**: 
  - 3x more posterior samples for better convergence diagnostics
  - Extended warmup improves adaptation, especially for Dirichlet layer
  - Better exploration of complex posterior with overdispersion parameters

**MCMC Sampling (Production Mode):**
- `n_draws_prod`: 10000 → **15000 draws**
- `n_warmup_prod`: 2000 → **3000 warmup steps**
- **Rationale**:
  - 50% more samples for higher-quality uncertainty quantification
  - Critical for Dirichlet-Multinomial with added parameters
  - Ensures robust estimates for submission

### Expected Impact

**Training Data Extension (150 days):**
- More stable parameter estimates (especially location effects)
- Better capture of long-term trends
- Reduced sensitivity to recent anomalies
- ~25% increase in training observations (depending on data availability)

**Increased MCMC Samples:**
- Improved convergence (lower Rhat, higher ESS)
- Better posterior coverage
- More reliable credible intervals
- Reduced Monte Carlo error in predictions

**Test Mode Runtime:**
- Previous: ~5-10 minutes
- Expected: ~15-30 minutes (3x samples + Dirichlet layer)

**Production Mode Runtime:**
- Previous: ~30-60 minutes  
- Expected: ~45-90 minutes (1.5x samples + Dirichlet layer)

### Model Comparison

| Feature | HMLR (pymc_hmlr) | Dirichlet-Multinomial (pymc_hmlr_dirichlet) |
|---------|------------------|---------------------------------------------|
| Mean trajectory | Linear in logit space | Linear in logit space |
| Observation variance | Multinomial only | Multinomial + Dirichlet |
| Overdispersion | No | Yes (estimated) |
| Extra parameters | 0 | 1 (concentration) |
| Use case | Standard variance | Count overdispersion |

### Files Modified

**Dirichlet-Multinomial Model:**
- `models/pymc_hmlr_dirichlet/config.yaml` - Updated parameters
- `models/pymc_hmlr_dirichlet/scripts/02_fit_model.py` - Added Dirichlet layer
- `models/pymc_hmlr_dirichlet/CHANGELOG.md` - This file

**Base HMLR Model:**
- `models/pymc_hmlr/config.yaml` - Updated parameters (training & MCMC)

### Usage

```bash
cd models/pymc_hmlr_dirichlet/scripts

# Test mode (3000 draws, 150 days lookback)
python run_workflow.py --nowcast-date 2025-11-14 --mode test

# Production mode (15000 draws, 150 days lookback)
python run_workflow.py --nowcast-date 2025-11-14 --mode prod
```

### Next Steps

1. Run both models with updated parameters
2. Compare convergence diagnostics (Rhat, ESS)
3. Evaluate if Dirichlet layer improves fit (posterior predictive checks)
4. Compare estimated concentration to understand overdispersion level
5. Assess prediction intervals (wider with Dirichlet due to extra variance)
