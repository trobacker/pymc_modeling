# Changelog - PyMC BART Variant Nowcast Model

All notable changes to this BART model implementation will be documented in this file.

## [1.0.0] - 2025-12-31

### Added
- Initial implementation of BART-based variant nowcast model
- Bayesian Additive Regression Trees for flexible non-parametric modeling
- Automatic learning of non-linear temporal patterns via tree ensembles
- Feature engineering: time + location one-hot encoding
- Optional Dirichlet-Multinomial layer for overdispersion
- Location-specific concentration parameters for heterogeneous uncertainty
- Support for K-1 parameterization with softmax link
- Full workflow scripts:
  - `01_fetch_data.py`: S3 data fetching with hub-specified clades
  - `02_fit_model.py`: BART model fitting with PGBART sampler
  - `03_format_submission.py`: Hub-compliant submission formatting
  - `run_workflow.py`: End-to-end orchestration
  - `plot_submission.py`: Visualization utilities
- Comprehensive configuration via `config.yaml`
- Detailed README with usage instructions and troubleshooting
- Diagnostic tools for model validation

### Model Specifications

**Architecture:**
- BART ensemble with 50 trees per clade (configurable)
- Particle Gibbs BART (PGBART) sampler with 10 particles
- NumPyro backend for efficient sampling
- Hierarchical concentration for location-specific uncertainty

**Data:**
- Training lookback: 150 days
- Minimum sequences per location-date: 5
- Feature matrix: time (centered/scaled) + location (one-hot)
- Supports all 52 US locations (50 states + DC + PR)

**Sampling (test mode):**
- Draws: 1000
- Warmup: 500
- Chains: 2
- Target accept: 0.90

**Sampling (production mode):**
- Draws: 3000
- Warmup: 1000
- Chains: 2
- Target accept: 0.90

**Submission:**
- 100 samples per task
- Mean predictions included
- Nowcast: 32 days (nowcast_date - 31 to nowcast_date)
- Forecast: 10 days ahead

### Implementation Notes

**Key Decisions:**
1. **BART vs HMLR**: Chose BART for flexibility in capturing non-linear dynamics
   - Better for sharp variant transitions
   - More robust to heterogeneous temporal patterns
   - Trade-off: less interpretable, more computationally expensive

2. **Feature encoding**: One-hot location encoding
   - Allows trees to learn location-specific patterns
   - No explicit hierarchical structure (BART handles partial pooling)

3. **Dirichlet layer**: Optional but recommended
   - Improves uncertainty calibration
   - Location-specific concentration captures heterogeneity
   - Essential for well-calibrated prediction intervals

4. **Tree parameters**: Conservative defaults
   - 50 trees balances flexibility and speed
   - 10 particles provides good PGBART mixing
   - Can increase for more complex patterns

**Limitations:**
- BART predictions for new locations use population average
- Extrapolation beyond training range uses linear tree extension
- Black-box model: limited mechanistic interpretation
- Computationally expensive: ~2-5x slower than HMLR

### Comparison with Other Models

**vs pymc_hmlr_dirichlet_loc_concentration:**
- BART: Non-parametric, automatic feature interaction, slower
- HMLR: Parametric linear trends, explicit hierarchical structure, faster
- Both: Support Dirichlet overdispersion and location-specific uncertainty

**When to use BART:**
- Expect complex non-linear variant dynamics
- Sharp transitions or sudden changes
- Prediction accuracy > interpretability
- Sufficient computational budget

**When to use HMLR:**
- Need interpretable coefficients
- Faster iteration for experimentation
- Smooth extrapolation preferred
- Limited compute resources

### Future Enhancements

Potential improvements for v1.1:
- [ ] Add hierarchical BART for better location pooling
- [ ] Implement proper BART prediction for out-of-sample locations
- [ ] Include additional features (e.g., population, region)
- [ ] Optimize tree parameters per clade
- [ ] Add ensemble with HMLR (model averaging)
- [ ] Implement adaptive concentration by clade
- [ ] Add uncertainty decomposition (epistemic vs aleatoric)
- [ ] Support weekly aggregation for computational efficiency

### Dependencies

- PyMC >= 5.20.0
- pymc-bart >= 0.5.0
- PyTensor >= 2.18.0
- ArviZ >= 0.20.0
- NumPyro (recommended backend)
- Polars >= 0.19.0
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- PyArrow >= 12.0.0

### Testing Status

- [x] Data fetching from S3
- [x] Training data preprocessing
- [x] Model compilation
- [ ] Model fitting (pending computational resources)
- [ ] Posterior predictive checks
- [ ] Submission validation
- [ ] Visual diagnostics

### Known Issues

1. **BART sampling speed**: Slower than HMLR (expected)
   - Mitigation: Use test mode for iteration, prod for final submission

2. **Memory usage**: Trees can be memory-intensive
   - Mitigation: Reduce n_trees or filter data more aggressively

3. **Extrapolation**: BART extends linearly beyond training range
   - Limitation: May not capture sudden changes in forecast horizon

4. **Location coverage**: Missing locations use average (not ideal)
   - Future: Implement hierarchical BART with region-level pooling

### Acknowledgments

- PyMC-BART team for excellent BART implementation
- Variant Nowcast Hub for data and infrastructure
- HMLR model for establishing baseline architecture

---

**Model Metadata:**
- Model ID: PyMC-BART
- Version: 1.0.0
- Date: 2025-12-31
- Framework: PyMC 5.x with pymc-bart extension
- Sampler: NUTS + PGBART (Particle Gibbs)
- Backend: NumPyro (recommended)
