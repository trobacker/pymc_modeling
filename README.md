# PyMC Modeling

A repository for constructing and testing Bayesian statistical models using PyMC in various contexts, with a focus on hierarchical models and COVID-19 variant analysis.

## Project Structure

```
pymc_modeling/
├── basic_examples/              # Introductory PyMC examples
│   ├── basic_examples.ipynb     # Linear regression with PyMC
│   └── count_models.ipynb       # Count-based models
├── variant_models/              # COVID-19 variant modeling
│   ├── pymc_covid_variant_models.ipynb  # Hierarchical multinomial model
│   ├── s3_hubdata.ipynb                 # S3 data access examples
│   ├── counts_2025-02-19.tsv.gz        # Variant count data
│   └── metadata.csv                     # Location and clade metadata
├── data/                        # Data files (gitignored)
├── results/                     # Output files (gitignored)
├── scripts/                     # Python scripts
├── pyproject.toml              # Project dependencies
├── CLAUDE.md                   # AI assistant guidance
└── README.md                   # This file
```

## Quick Start

### 1. Install uv (Package Manager)

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Via pip (alternative):**
```bash
pip install uv
```

### 2. Set Up Environment

Clone the repository and set up the environment:

```bash
# Clone the repository
cd pymc_modeling

# Create virtual environment and install dependencies
uv sync

# This creates a .venv directory and installs all dependencies from pyproject.toml
```

### 3. Activate Environment

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\activate
```

### 4. Run Notebooks

**Option A: Using Jupyter Lab (Recommended)**
```bash
jupyter lab
```

**Option B: Using VS Code**
1. Open the project in VS Code
2. Install the Python and Jupyter extensions
3. Open a notebook file (.ipynb)
4. Click "Select Kernel" in the top right
5. Choose the Python interpreter from `.venv/bin/python`

**Option C: Using Jupyter Notebook**
```bash
jupyter notebook
```

## Installing Additional Packages

To add new dependencies to the project:

```bash
# Add a new package
uv add package-name

# Add a development dependency
uv add --dev package-name

# Example: Add plotly
uv add plotly
```

## Installed Libraries

This project includes the following key libraries:

### Bayesian Modeling
- **PyMC** (v5.20+) - Probabilistic programming framework
- **ArviZ** (v0.20+) - Bayesian visualization and diagnostics
- **PyTensor** - Backend for PyMC

### Data Manipulation
- **Pandas** - DataFrame manipulation
- **Polars** - Fast DataFrame library with S3 support
- **NumPy** - Numerical computing

### Visualization
- **Matplotlib** - Core plotting library
- **Seaborn** - Statistical visualization

### Scientific Computing
- **SciPy** - Scientific algorithms

### Notebooks
- **Jupyter** - Notebook interface
- **JupyterLab** - Advanced notebook environment
- **IPython** - Interactive Python shell

## Development Workflow

### Working with Notebooks

1. **Start with basic examples**: Explore `basic_examples/` to understand PyMC patterns
2. **Progress to advanced models**: Check `variant_models/` for hierarchical models
3. **Document your work**: Add markdown cells explaining concepts and results
4. **Save outputs**: Store figures and results in the `results/` directory

### Best Practices

- **Data management**: Place raw data in `data/` (gitignored for large files)
- **Script organization**: Move reusable code from notebooks to `scripts/`
- **Version control**: Use meaningful commit messages: `"project-name: brief description"`
- **Model diagnostics**: Always check trace plots, rhat, and effective sample size

## Bayesian Modeling with PyMC

### Basic Workflow

```python
import pymc as pm
import arviz as az

# 1. Define the model
with pm.Model() as model:
    # Define priors
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    # Define likelihood
    mu = alpha + beta * X
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # 2. Sample from posterior
    trace = pm.sample(draws=1000, tune=1000, cores=4)

    # 3. Posterior predictive checks
    post_pred = pm.sample_posterior_predictive(trace)

# 4. Analyze results
az.summary(trace)
az.plot_trace(trace)
az.plot_ppc(post_pred)
```

### Key Concepts in This Repository

- **Hierarchical Models**: Location-variant specific parameters with partial pooling
- **Multinomial Regression**: Modeling COVID-19 variant proportions over time
- **Posterior Predictive Checks**: Validating model fit with observed data
- **Softmax Link Function**: Ensuring probabilities sum to 1 in categorical models

## Updating Dependencies

To update all dependencies to their latest compatible versions:

```bash
# Update all packages
uv sync --upgrade

# Update a specific package
uv add package-name --upgrade
```

## Data Sources

### COVID Variant Hub
Access variant data from the US COVID Variant Nowcast Hub via S3:
- See `variant_models/s3_hubdata.ipynb` for examples
- Parquet files with model outputs and oracle data
- Uses Polars for efficient data loading

## Troubleshooting

### Environment Issues

If you encounter environment issues:

```bash
# Remove the virtual environment
rm -rf .venv

# Recreate from scratch
uv sync
```

### Kernel Not Found in Jupyter

1. Make sure the environment is activated
2. Restart Jupyter Lab/Notebook
3. In the notebook, select Kernel → Change Kernel → Python (.venv)

### PyMC Sampling Issues

- **Divergences**: Increase `target_accept` (e.g., 0.95) or reparameterize
- **Max tree depth warnings**: Increase `max_treedepth` or simplify the model
- **High rhat**: Run longer chains or check for multimodality

## Additional Resources

- [PyMC Documentation](https://www.pymc.io/projects/docs/en/stable/)
- [ArviZ Documentation](https://python.arviz.org/)
- [Bayesian Analysis Recipes](https://github.com/ericmjl/bayesian-analysis-recipes)
- [COVID Variant Hub](https://github.com/reichlab/variant-nowcast-hub)

## Contributing

This is a personal research repository. For questions or suggestions, please open an issue.

## License

This project is for research and educational purposes.
