# Basic PyMC Examples

This directory contains introductory Bayesian modeling examples using PyMC.

## Virtual Environment

A hidden `.venv` directory contains the Python virtual environment with all dependencies.

### Activating the Environment (Shell)

```bash
# From this directory
source .venv/bin/activate

# Or from the repo root
source basic_examples/.venv/bin/activate
```

To deactivate:
```bash
deactivate
```

### Using with Jupyter

A Jupyter kernel named **Python (basic_examples_pymc)** is already registered and points to this environment. Select it when opening notebooks in Jupyter or VS Code.

To verify the kernel is available:
```bash
jupyter kernelspec list
```

### Recreating the Environment

If you need to rebuild the environment from scratch:

```bash
cd basic_examples

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name basic_examples_pymc --display-name "Python (basic_examples_pymc)"
```

## Contents

- `basic_examples.ipynb` - Linear regression with PyMC
- `count_models.ipynb` - Count-based Bayesian models
- `data_viz_basics.ipynb` - Data visualization basics
