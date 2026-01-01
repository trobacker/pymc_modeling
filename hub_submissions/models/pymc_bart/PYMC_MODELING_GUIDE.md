# PyMC Modeling Guide - BART Variant Nowcast

A deep dive into the PyMC modeling aspects of the BART variant nowcast implementation.

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Mathematical Formulation](#mathematical-formulation)
3. [PyMC Implementation Details](#pymc-implementation-details)
4. [BART Specifics](#bart-specifics)
5. [Prior Choices](#prior-choices)
6. [Sampling Strategy](#sampling-strategy)
7. [Posterior Analysis](#posterior-analysis)
8. [Common PyMC Patterns](#common-pymc-patterns)
9. [Troubleshooting](#troubleshooting)

---

## Model Architecture

### High-Level Structure

The model uses **Bayesian Additive Regression Trees (BART)** within a PyMC probabilistic programming framework to predict variant proportions over time and location.

```
Input: (time, location) → Features: X
        ↓
BART Ensembles (one per clade)
        ↓
Log-odds: η_k = f_k(X)
        ↓
Softmax Link: p = softmax(η)
        ↓
Dirichlet Layer: θ ~ Dirichlet(concentration * p)
        ↓
Likelihood: Y ~ Multinomial(n, θ)
```

### Component Breakdown

**Layer 1: Feature Engineering (02_fit_model.py:61-152)**
```python
# Time: centered and scaled
time = (time_raw - time_mean) / time_std

# Location: one-hot encoded
location_onehot = np.zeros((n_obs, num_locations))
location_onehot[np.arange(n_obs), locations] = 1

# Feature matrix
X = np.column_stack([time, location_onehot])
# Shape: (n_observations, 1 + num_locations)
```

**Layer 2: BART Ensembles (02_fit_model.py:219-247)**
```python
# One BART model per clade
eta_list = []
for clade_idx in range(num_clades):
    bart = pmb.BART(
        f"bart_{clade_idx}",
        X=X,                    # Feature matrix
        Y=np.zeros(n_obs),      # Initialized to zero
        m=n_trees,              # Number of trees (default: 50)
        shape=n_observations
    )
    eta_list.append(bart)

# Stack into matrix: (n_obs, n_clades)
eta = pt.stack(eta_list, axis=1)
```

**Layer 3: Softmax Link (02_fit_model.py:249-254)**
```python
# Convert log-odds to probabilities
p_mean = pm.math.softmax(eta, axis=1)
# Ensures: p_i >= 0 and sum(p_i) = 1
```

**Layer 4: Dirichlet Overdispersion (02_fit_model.py:256-282)**
```python
# Hierarchical concentration (location-specific)
mu_concentration = pm.Normal('mu_concentration',
                              mu=np.log(50), sigma=2)
sigma_concentration = pm.HalfNormal('sigma_concentration', sigma=2)

log_concentration = pm.Normal('log_concentration',
                              mu=mu_concentration,
                              sigma=sigma_concentration,
                              shape=num_locations)
concentration = pm.math.exp(log_concentration)

# Apply to observations
concentration_expanded = concentration[locations]
alpha_dirichlet = concentration_expanded[:, None] * p_mean

# Dirichlet-distributed proportions
theta = pm.Dirichlet('theta',
                     a=alpha_dirichlet,
                     shape=(n_observations, num_clades))
```

**Layer 5: Likelihood (02_fit_model.py:284-290)**
```python
# Multinomial observation model
Y_obs = pm.Multinomial('Y_obs',
                       n=total_counts,  # Total sequences per observation
                       p=theta,         # Dirichlet-sampled proportions
                       observed=Y)      # Observed clade counts
```

---

## Mathematical Formulation

### Full Hierarchical Model

```
Feature Engineering:
  X[i] = [time[i], location_onehot[i, :]]

BART Layer (for each clade k):
  η[i, k] = Σ_{j=1}^{m} T_{k,j}(X[i])
  where T_{k,j} is tree j in ensemble k

Softmax Link:
  p[i, k] = exp(η[i, k]) / Σ_k exp(η[i, k])

Hierarchical Concentration:
  μ_φ ~ Normal(log(50), 2)
  σ_φ ~ HalfNormal(2)
  log(φ_l) ~ Normal(μ_φ, σ_φ)  [for location l]
  φ_l = exp(log(φ_l))

Dirichlet Layer:
  α[i, :] = φ[location[i]] * p[i, :]
  θ[i, :] ~ Dirichlet(α[i, :])

Likelihood:
  Y[i, :] ~ Multinomial(n[i], θ[i, :])
```

### Variance Structure

**BART variance:**
- Trees provide function flexibility
- Sum-of-trees prior induces regularization
- Each tree contributes small piece of prediction

**Dirichlet variance:**
```
Var(θ[i, k]) ≈ p[i, k] * (1 - p[i, k]) / (φ[location[i]] + 1)
```
- Higher φ → tighter around p (less overdispersion)
- Lower φ → wider uncertainty (more overdispersion)
- Location-specific φ allows heterogeneous calibration

**Total uncertainty:**
1. **Epistemic** (parameter): From BART trees (captured by MCMC)
2. **Aleatoric** (data): From Dirichlet + Multinomial layers

---

## PyMC Implementation Details

### Key PyMC Objects

**1. Model Context Manager**
```python
with pm.Model() as variant_model:
    # All PyMC distributions defined here
    # Automatic graph construction
```

**2. BART Distribution (via pymc-bart)**
```python
bart = pmb.BART(
    name="bart_0",           # Variable name
    X=X,                     # Features (n_obs, n_features)
    Y=np.zeros(n_obs),       # Pseudo-targets (updated during sampling)
    m=50,                    # Number of trees
    shape=n_observations     # Output shape
)
```
- Returns PyTensor tensor of shape `(n_observations,)`
- Each BART variable has `m` trees in ensemble
- Trees are sampled using Particle Gibbs BART (PGBART)

**3. PyTensor Stack Operation**
```python
eta = pt.stack([bart_0, bart_1, ..., bart_K], axis=1)
# Shape: (n_observations, num_clades)
```
- Combines individual BART outputs into matrix
- Required for vectorized softmax

**4. Softmax Link**
```python
p_mean = pm.math.softmax(eta, axis=1)
```
- PyMC's softmax is numerically stable
- Automatically applies to correct axis
- Ensures valid probability simplex

**5. Hierarchical Normal**
```python
mu = pm.Normal('mu', mu=0, sigma=10)       # Hyperprior mean
sigma = pm.HalfNormal('sigma', sigma=10)   # Hyperprior std

param = pm.Normal('param', mu=mu, sigma=sigma, shape=n)
```
- Standard hierarchical pattern
- Partial pooling across groups
- Here applied to concentration parameters

**6. Dirichlet Distribution**
```python
theta = pm.Dirichlet('theta',
                     a=alpha_dirichlet,  # Shape: (n_obs, n_clades)
                     shape=(n_observations, num_clades))
```
- `a` parameter is concentration
- Each row of theta is independent Dirichlet
- Rows sum to 1

**7. Multinomial Likelihood**
```python
Y_obs = pm.Multinomial('Y_obs',
                       n=total_counts,  # Shape: (n_obs,)
                       p=theta,         # Shape: (n_obs, n_clades)
                       observed=Y)      # Shape: (n_obs, n_clades)
```
- `n` can be array (different totals per observation)
- `p` provides probabilities
- `observed` fixes Y_obs to data

### Variable Shapes Reference

| Variable | Shape | Description |
|----------|-------|-------------|
| `X` | `(n_obs, n_features)` | Feature matrix |
| `bart_k` | `(n_obs,)` | BART output for clade k |
| `eta` | `(n_obs, n_clades)` | Stacked log-odds |
| `p_mean` | `(n_obs, n_clades)` | Mean proportions |
| `mu_concentration` | `()` | Scalar hyperprior |
| `sigma_concentration` | `()` | Scalar hyperprior |
| `log_concentration` | `(num_locations,)` | Per-location log-concentration |
| `concentration` | `(num_locations,)` | Per-location concentration |
| `concentration_expanded` | `(n_obs,)` | Concentration for each observation |
| `alpha_dirichlet` | `(n_obs, n_clades)` | Dirichlet parameters |
| `theta` | `(n_obs, n_clades)` | Sampled proportions |
| `Y_obs` | `(n_obs, n_clades)` | Observed counts |

---

## BART Specifics

### What is BART?

**Bayesian Additive Regression Trees** is a non-parametric method that represents a function as:

```
f(X) = Σ_{j=1}^{m} g(X; T_j, M_j)
```

Where:
- `T_j` is tree structure (splits and terminal nodes)
- `M_j` is terminal node values (leaf predictions)
- `m` is number of trees (typically 50-200)

**Key properties:**
1. **Additive**: Sum of small trees, not one large tree
2. **Regularization**: Prior on tree depth/splits prevents overfitting
3. **Bayesian**: Full posterior over tree structures and parameters

### PGBART Sampler

PyMC-BART uses **Particle Gibbs BART (PGBART)** for sampling:

```
For each MCMC iteration:
  1. Sample tree structures T_j using Particle Gibbs
  2. Sample leaf values M_j conditional on T_j
  3. Sample other model parameters using NUTS
```

**Configuration:**
```python
bart = pmb.BART(
    ...,
    m=50,              # Number of trees
    # Internal PGBART settings (auto-configured):
    # - n_particles: Number of particles (default: adaptive)
    # - alpha, beta: Tree depth prior (default: 0.95, 2)
)
```

### BART Prior Structure

**Tree depth prior:**
```
P(depth = d) ∝ α * (1 + d)^{-β}
```
- Default: α=0.95, β=2
- Favors shallow trees
- Prevents overfitting

**Leaf value prior:**
```
M_j,l ~ Normal(0, σ²_μ)
```
- σ²_μ chosen so sum of trees has reasonable scale
- Automatic calibration based on Y scale

**Splitting rule prior:**
- Uniform over available features
- Uniform over available split points
- Trees naturally discover important features

### Feature Importance

BART implicitly learns feature importance through splitting frequency:

```python
# After fitting, analyze which features appear in splits
# Time vs Location features
# Not directly accessible in PyMC-BART (would need custom analysis)
```

In our model:
- **Time feature**: Captures temporal trends
- **Location features**: Capture spatial heterogeneity
- **Interactions**: Trees automatically learn time×location effects

---

## Prior Choices

### 1. BART Priors (Implicit)

```python
bart = pmb.BART(name, X, Y, m=50)
```

**Defaults (sensible for most cases):**
- Tree depth prior: α=0.95, β=2 (shallow trees)
- Leaf values: σ²_μ auto-calibrated
- 50 trees: Good balance of flexibility and efficiency

**Rationale:**
- Shallow trees prevent overfitting
- Sum of many small trees provides smoothness
- 50 trees sufficient for variant nowcast complexity

### 2. Concentration Hierarchical Priors

```python
mu_concentration = pm.Normal('mu_concentration',
                              mu=np.log(50), sigma=2)
```

**Choice: μ ~ log(50)**
- 50 is moderate concentration
- Variance ≈ p(1-p)/51 ≈ 0.005 when p=0.5
- Provides reasonable uncertainty calibration

**Choice: σ = 2 (on log scale)**
- Allows concentration range: exp(log(50)±4) = [9, 270]
- Flexible enough for location heterogeneity
- Not too wide (avoids extreme values)

```python
sigma_concentration = pm.HalfNormal('sigma_concentration', sigma=2)
```

**Choice: σ = 2**
- Weakly informative
- Allows σ_φ ∈ (0, ~6) with high probability
- Sufficient for modeling between-location variation

### 3. Prior Sensitivity

**BART is relatively robust to priors:**
- Tree structure is data-driven
- Leaf values adapt to scale
- Main tuning: number of trees `m`

**Concentration priors matter more:**
- Controls uncertainty calibration
- Too high → overconfident predictions
- Too low → overly wide intervals
- Hierarchical structure learns from data

### 4. Effective Sample Size

With hierarchical priors, effective information:
```
ESS(concentration[l]) depends on:
  - Data for location l: n_l observations
  - Prior: σ_φ controls pooling strength
  - Hyperprior: μ_φ pools across locations
```

Locations with more data → less shrinkage toward μ_φ
Locations with less data → more shrinkage (partial pooling)

---

## Sampling Strategy

### NUTS + PGBART Hybrid

PyMC uses **No-U-Turn Sampler (NUTS)** for continuous parameters and **PGBART** for tree structures:

```python
trace = pm.sample(
    n_draws,
    tune=n_warmup,
    chains=chains,
    cores=cores,
    target_accept=target_accept,
    return_inferencedata=True,
    nuts_sampler="numpyro",  # Recommended for BART
)
```

**What's sampled with NUTS:**
- `mu_concentration`, `sigma_concentration`
- `log_concentration[l]` for each location
- `theta[i]` for each observation (Dirichlet parameters)

**What's sampled with PGBART:**
- Tree structures `T_j` for each BART
- Leaf values `M_j` for each BART

### NumPyro Backend

```python
nuts_sampler="numpyro"
```

**Why NumPyro?**
1. **Speed**: JIT-compiled sampling (faster than PyMC default)
2. **Memory**: More efficient gradient computation
3. **BART compatibility**: Better integration with PGBART
4. **Stability**: Robust for complex hierarchical models

**Installation:**
```bash
pip install numpyro jax jaxlib
```

### Sampling Parameters

**For test mode:**
```python
n_draws = 1000
n_warmup = 500
chains = 2
target_accept = 0.90
```

**Rationale:**
- 1000 draws: Sufficient for convergence check
- 500 warmup: BART trees adapt structure
- 2 chains: Balance between speed and convergence check
- 0.90 target_accept: Lower than typical (BART rarely diverges)

**For production mode:**
```python
n_draws = 3000
n_warmup = 1000
chains = 2
target_accept = 0.90
```

**Rationale:**
- 3000 draws: More robust posterior estimates
- 1000 warmup: Thorough tree structure exploration
- Still 2 chains: BART is expensive, more draws > more chains

### Warmup Phase Details

**What happens during warmup:**
1. **Adaptation (first 75%)**: Tree structures change rapidly
2. **Stabilization (last 25%)**: Trees settle, focus on parameters
3. **Step size tuning**: NUTS adapts step size
4. **Mass matrix**: NUTS estimates parameter correlations

**For BART specifically:**
- Trees need time to find good splits
- Early iterations: Trees are random
- Later iterations: Trees capture patterns
- Warmup should be at least 50% of total (here 500/1500 = 33%, acceptable but conservative)

### Convergence Diagnostics

**1. R-hat (Gelman-Rubin)**
```python
summary = az.summary(trace)
print(summary['r_hat'])
```
- **Target**: < 1.01 (ideally < 1.05)
- Measures between-chain vs within-chain variance
- > 1.01 → chains haven't converged

**2. Effective Sample Size (ESS)**
```python
print(summary['ess_bulk'], summary['ess_tail'])
```
- **Target**: > 100 per chain (ideally > 400)
- ESS < draws due to autocorrelation
- BART has higher autocorrelation than typical NUTS

**3. Divergences**
```python
n_divergences = trace.sample_stats['diverging'].sum().item()
```
- **Target**: 0
- BART rarely diverges (trees are sampled separately)
- If divergences occur: check data quality or increase target_accept

**4. Tree Depth**
```python
max_depth = trace.sample_stats['tree_depth'].max().item()
max_tree_depth_setting = 10  # from config
```
- **Warning if**: Frequently hitting max_tree_depth
- Indicates gradient evaluation issues
- Less relevant for BART (trees sampled differently)

### Monitoring Progress

```python
# Built-in progress bar
# Shows:
# - Current iteration
# - Divergences
# - Mean acceptance probability
# - Step size

# Example output:
# Sampling 2 chains, 3000 draws per chain:
# 100%|████████| 8000/8000 [45:32<00:00, 2.93it/s]
```

**Typical iteration times:**
- HMLR: ~0.5-1s per iteration
- BART: ~2-5s per iteration (slower due to trees)

---

## Posterior Analysis

### Loading Results

```python
import arviz as az

# Load trace
trace = az.from_netcdf("fitted/trace_2024-11-13.nc")

# Load posterior predictive
post_pred = az.from_netcdf("fitted/posterior_predictive_2024-11-13.nc")
```

### Summary Statistics

```python
# All parameters
summary = az.summary(trace)
print(summary)

# Specific variables
summary_concentration = az.summary(
    trace,
    var_names=['mu_concentration', 'sigma_concentration', 'log_concentration']
)
print(summary_concentration)
```

**Key columns:**
- `mean`: Posterior mean
- `sd`: Posterior standard deviation
- `hdi_3%`, `hdi_97%`: 94% credible interval
- `r_hat`: Convergence diagnostic
- `ess_bulk`, `ess_tail`: Effective sample size

### Trace Plots

```python
import matplotlib.pyplot as plt

# Concentration parameters
az.plot_trace(
    trace,
    var_names=['mu_concentration', 'sigma_concentration'],
    compact=True
)
plt.tight_layout()
plt.savefig("trace_concentration.png")
```

**What to look for:**
- **Left panels**: Posterior distributions (should be smooth)
- **Right panels**: MCMC traces (should be "hairy caterpillars")
- **Between chains**: Should overlap (good mixing)

### Posterior Predictive Checks

```python
# Compare observed data to posterior predictions
az.plot_ppc(
    post_pred,
    group='posterior_predictive',
    num_pp_samples=100
)
plt.savefig("ppc.png")
```

**Interpretation:**
- Blue histogram: Observed data
- Orange lines: Posterior predictive samples
- Good fit: Blue histogram within orange cloud

### Extracting Predictions

```python
# Extract BART predictions
extracted = az.extract(trace.posterior, combined=True)

# BART outputs (log-odds)
bart_0 = extracted['bart_0'].values  # Shape: (n_obs, n_samples)

# Concentration parameters
concentrations = extracted['log_concentration'].values
concentrations = np.exp(concentrations)  # Transform to natural scale
# Shape: (num_locations, n_samples)

# Compute proportions manually
def compute_proportions(trace, time_idx, location_idx):
    """Compute proportions for specific time/location"""
    extracted = az.extract(trace.posterior, combined=True)

    # Get BART predictions for each clade
    etas = []
    for k in range(num_clades):
        bart_k = extracted[f'bart_{k}'].values  # (n_obs, n_samples)
        etas.append(bart_k[time_idx, :])

    eta_matrix = np.stack(etas, axis=0)  # (n_clades, n_samples)

    # Apply softmax
    exp_eta = np.exp(eta_matrix - np.max(eta_matrix, axis=0))
    props = exp_eta / exp_eta.sum(axis=0)

    return props  # Shape: (n_clades, n_samples)
```

### Credible Intervals

```python
# 90% credible interval for proportions
def get_credible_interval(proportions, alpha=0.1):
    """
    proportions: (n_samples,) array
    Returns: (lower, upper) tuple
    """
    lower = np.quantile(proportions, alpha/2)
    upper = np.quantile(proportions, 1-alpha/2)
    return lower, upper

# Example
prop_ca_clade0_day100 = proportions[location=CA, clade=0, time=100, :]
lower, upper = get_credible_interval(prop_ca_clade0_day100)
print(f"90% CI: [{lower:.3f}, {upper:.3f}]")
```

### Comparing Chains

```python
# Check if chains converged to same distribution
az.plot_forest(
    trace,
    var_names=['mu_concentration', 'sigma_concentration'],
    combined=False  # Show each chain separately
)
plt.savefig("forest_by_chain.png")
```

**Good convergence**: Chain-specific intervals overlap substantially

---

## Common PyMC Patterns

### Pattern 1: Hierarchical Structure

```python
# Level 1: Hyperpriors (population-level)
mu = pm.Normal('mu', mu=0, sigma=10)
sigma = pm.HalfNormal('sigma', sigma=5)

# Level 2: Group-level parameters
param = pm.Normal('param', mu=mu, sigma=sigma, shape=n_groups)

# Level 3: Observation-level (implicit through likelihood)
```

**Used in our model for concentration parameters**

### Pattern 2: Broadcasting

```python
# Parameter has shape (n_groups,)
param = pm.Normal('param', mu=0, sigma=1, shape=n_groups)

# Observation indices have shape (n_obs,)
group_indices = [0, 0, 1, 1, 2, ...]  # Which group each obs belongs to

# Expand to observation level
param_expanded = param[group_indices]  # Shape: (n_obs,)
```

**Used in our model for location-specific concentration**

### Pattern 3: Softmax for Compositions

```python
# Log-odds (unconstrained)
eta = some_model(X)  # Shape: (n_obs, n_categories)

# Convert to probabilities (constrained to simplex)
p = pm.math.softmax(eta, axis=1)  # Shape: (n_obs, n_categories)
# Guarantees: p >= 0 and sum(p) = 1
```

**Used in our model for clade proportions**

### Pattern 4: Dirichlet-Multinomial

```python
# Mean proportions from model
p = some_model(X)  # Shape: (n_obs, n_categories)

# Add Dirichlet overdispersion
concentration = pm.Gamma('concentration', alpha=2, beta=0.1)
alpha = concentration * p
theta = pm.Dirichlet('theta', a=alpha, shape=(n_obs, n_categories))

# Multinomial likelihood
Y = pm.Multinomial('Y', n=n_total, p=theta, observed=data)
```

**Used in our model for overdispersed multinomial**

### Pattern 5: Non-centered Parameterization

**Centered (can have sampling issues):**
```python
param = pm.Normal('param', mu=mu, sigma=sigma, shape=n)
```

**Non-centered (better geometry):**
```python
param_raw = pm.Normal('param_raw', mu=0, sigma=1, shape=n)
param = pm.Deterministic('param', mu + sigma * param_raw)
```

**Not used for BART** (BART handles this internally)
**Could be used for concentration** (currently centered)

### Pattern 6: Deterministic Transformations

```python
# Log-scale parameter (sampled)
log_param = pm.Normal('log_param', mu=0, sigma=1)

# Natural-scale parameter (deterministic)
param = pm.Deterministic('param', pm.math.exp(log_param))
```

**Used in our model:**
```python
log_concentration = pm.Normal('log_concentration', ...)
concentration = pm.math.exp(log_concentration)  # Not pm.Deterministic, but could be
```

---

## Troubleshooting

### Issue 1: Slow Sampling

**Symptom:** Iterations/second < 0.5

**Solutions:**
```python
# 1. Reduce number of trees
n_trees = 20  # Default: 50

# 2. Use fewer chains, more draws per chain
chains = 1
n_draws = 3000

# 3. Reduce data size
training_lookback_days = 90  # Default: 150
min_sequences = 10  # Default: 5

# 4. Disable Dirichlet layer (faster, less accurate)
use_dirichlet = false
```

### Issue 2: Poor Convergence (R-hat > 1.05)

**Symptom:** Chains haven't mixed

**Solutions:**
```python
# 1. Increase warmup
n_warmup = 2000  # Default: 1000

# 2. Increase draws
n_draws = 5000  # Default: 3000

# 3. Check for data issues
# - Outliers in sequence counts?
# - Missing data patterns?
# - Extreme imbalance in clades?

# 4. Non-center concentration parameters
log_concentration_raw = pm.Normal('log_concentration_raw', mu=0, sigma=1, shape=n_locations)
log_concentration = pm.Deterministic('log_concentration',
                                     mu_concentration + sigma_concentration * log_concentration_raw)
```

### Issue 3: Low ESS (<100)

**Symptom:** High autocorrelation in chains

**Solutions:**
```python
# 1. Run longer chains
n_draws = 10000  # More draws → higher ESS

# 2. Thin samples (last resort)
trace_thinned = trace.sel(draw=slice(None, None, 10))  # Keep every 10th

# 3. Check target_accept
target_accept = 0.95  # Higher → smaller steps → less autocorrelation
```

### Issue 4: Divergences

**Symptom:** `diverging > 0`

**Rare for BART, but if it happens:**
```python
# 1. Increase target_accept
target_accept = 0.95  # Default: 0.90

# 2. Check data quality
# - Are there zero counts?
# - Extreme outliers?
# - Missing location-date combinations?

# 3. Adjust priors
# - Wider priors on concentration
sigma_concentration = pm.HalfNormal('sigma_concentration', sigma=5)  # Default: 2
```

### Issue 5: Memory Issues

**Symptom:** Out of memory error

**Solutions:**
```python
# 1. Reduce observations
min_sequences = 20  # Filter more aggressively
training_lookback_days = 60  # Shorter training period

# 2. Reduce trees
n_trees = 20  # Default: 50

# 3. Reduce chains
chains = 1  # Default: 2

# 4. Don't save posterior predictive during sampling
# (sample it separately after fitting)
```

### Issue 6: BART Not Learning

**Symptom:** Flat predictions, no temporal patterns

**Solutions:**
```python
# 1. Increase trees
n_trees = 100  # Default: 50

# 2. Check feature scaling
# - Time should be centered/scaled
# - Verify time range is reasonable

# 3. Increase warmup
n_warmup = 2000  # Give trees time to adapt

# 4. Check data
# - Sufficient temporal variation?
# - Clear patterns to learn?
```

### Issue 7: Overfitting

**Symptom:** Perfect fit to training, poor on forecast

**Solutions:**
```python
# 1. Reduce trees
n_trees = 30  # Default: 50

# 2. Stronger regularization (BART priors)
# (Would require custom BART implementation)

# 3. Increase concentration
# - Tighter around mean → less overfitting to noise
concentration_init = 100  # Default: 50

# 4. Cross-validation
# - Hold out recent dates
# - Evaluate forecast accuracy
```

---

## Summary

This BART model combines several PyMC patterns:

1. **Non-parametric regression** (BART) for flexible function approximation
2. **Hierarchical priors** for location-specific parameters
3. **Compositional data** via Dirichlet-Multinomial
4. **Hybrid sampling** (NUTS + PGBART) for mixed parameter types

**Key advantages:**
- Automatically learns complex temporal patterns
- Handles interactions without specification
- Full Bayesian uncertainty quantification

**Key considerations:**
- More computationally expensive than parametric models
- Requires careful warmup for tree adaptation
- Less interpretable than hierarchical linear models

**For questions:**
- PyMC discourse: https://discourse.pymc.io/
- PyMC-BART docs: https://github.com/pymc-devs/pymc-bart
- ArviZ docs: https://arviz-devs.github.io/arviz/

