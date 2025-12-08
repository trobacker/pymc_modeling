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

## Environment

These notebooks use the existing `.venv` environment from the main repository. Required packages:
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

## Getting Started

1. Ensure you're in the repository root and have activated the environment

2. Launch Jupyter:
   ```bash
   jupyter notebook
   ```

3. Navigate to `epidemic_models/` and open `sir_epidemic_model_tutorial.ipynb`

4. Select the "pymc-modeling" kernel (or your environment kernel) and run cells sequentially

**Note**: The notebook will display a message about graphviz being optional if it's not installed. This is expected and the tutorial will work perfectly without it.

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
