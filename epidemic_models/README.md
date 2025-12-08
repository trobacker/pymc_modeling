# Epidemic Models with PyMC

This directory contains Bayesian epidemic modeling examples using PyMC.

## Contents

### Notebooks

- **`sir_epidemic_model_tutorial.ipynb`**: Comprehensive tutorial on fitting a classic SIR (Susceptible-Infected-Recovered) epidemic model to data using PyMC. This notebook demonstrates:
  - SIR model structure and ODEs
  - Simulating epidemic data with observation noise
  - Bayesian parameter inference with PyMC
  - Posterior predictive checks and model validation
  - Uncertainty quantification in epidemic trajectories
  - Technical deep dives into: ODE solvers, Negative Binomial observation models, HMC-NUTS sampling

- **`real_covid_data_analysis.ipynb`**: Analysis of real COVID-19 surveillance data using Bayesian semi-parametric models. This notebook shows:
  - Fetching real epidemic data from CMU Delphi Epidata API
  - Exploratory data analysis and visualization of COVID-19 waves
  - Building flexible growth rate models for real-world data
  - Log-linear trends with periodic components
  - Handling overdispersion in real count data
  - Model validation and residual analysis
  - Practical considerations for epidemic forecasting

## Environment

### SIR Tutorial (`sir_epidemic_model_tutorial.ipynb`)

Uses the existing `.venv` environment from the main repository. Required packages:
- PyMC 5.20.1+
- ArviZ 0.20.0+
- NumPy
- SciPy (for ODE integration)
- Matplotlib/Seaborn
- Pandas

Optional packages for enhanced visualization:
- `graphviz` (Python package + system binary) - for model structure visualization
  - Install: `pip install graphviz` + system package (e.g., `brew install graphviz`)
  - Note: The tutorial works fine without this - it's only for visualizing the model DAG

### Real Data Analysis (`real_covid_data_analysis.ipynb`)

Uses a dedicated virtual environment (`epidata_env`) with additional packages:
- All packages from above, plus:
- `epidatpy` - CMU Delphi Epidata API client (installed from GitHub)
- `requests`, `epiweeks` - API dependencies

**Setup:**
```bash
cd epidemic_models
source epidata_env/bin/activate
jupyter notebook real_covid_data_analysis.ipynb
```

**Optional API Key:**
- The Epidata API works without a key but has rate limits
- Register for a free key at: https://api.delphi.cmu.edu/epidata/admin/registration_form
- Set environment variable: `export DELPHI_EPIDATA_KEY=your_key_here`

## Getting Started

### For SIR Tutorial:

1. Ensure you're in the repository root and have activated the main environment

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

3. Navigate to `epidemic_models/` and open `sir_epidemic_model_tutorial.ipynb`

4. Select the "pymc-modeling" kernel (or your environment kernel) and run cells sequentially

**Note**: The notebook will display a message about graphviz being optional if it's not installed. This is expected and the tutorial will work perfectly without it.

### For Real COVID-19 Data Analysis:

1. Navigate to the epidemic_models directory:
   ```bash
   cd epidemic_models
   ```

2. Activate the dedicated environment:
   ```bash
   source epidata_env/bin/activate
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook real_covid_data_analysis.ipynb
   ```

4. Run cells sequentially - the notebook will fetch real data from the Delphi API

## Model Overview: SIR Epidemic Model

The SIR model divides a population into three compartments:
- **S(t)**: Susceptible individuals
- **I(t)**: Infected individuals
- **R(t)**: Recovered individuals

Governed by differential equations:
```
dS/dt = -β * S * I / N
dI/dt = β * S * I / N - γ * I
dR/dt = γ * I
```

Where:
- β = transmission rate
- γ = recovery rate
- N = total population
- R₀ = β/γ = basic reproduction number

## Key Learning Objectives

1. **Data-Model Connection**: Understanding how noisy observations connect to underlying deterministic dynamics
2. **Bayesian Inference**: Using prior knowledge and data to infer epidemic parameters
3. **Uncertainty Quantification**: Propagating parameter uncertainty through predictions
4. **Model Validation**: Posterior predictive checks and diagnostic assessments
5. **PyMC Workflow**: Complete pipeline from model specification to inference

## Future Extensions

Potential additions to this directory:
- SEIR model (with exposed compartment)
- Age-structured models
- Spatial epidemic models
- Time-varying parameters
- Real-world data applications
- Model comparison examples
