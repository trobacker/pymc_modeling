# PyMC BART Model - Setup Instructions

## Initial Setup (Do This Once)

Follow these steps to set up the model-specific virtual environment:

```bash
# 1. Navigate to the pymc_bart model directory
cd /Users/trobacker/GitHub/pymc_modeling/hub_submissions/models/pymc_bart

# 2. Create model-specific virtual environment
python -m venv .venv

# 3. Activate the environment
source .venv/bin/activate

# You should see (.venv) in your prompt now

# 4. Install dependencies (this will take 5-10 minutes)
pip install -r requirements.txt

# 5. Verify installation
python -c "import pymc; import pymc_bart; print('✓ Environment ready!')"
```

## Running the Model (Every Time)

Each time you want to run the model in a new terminal session:

```bash
# 1. Navigate to model directory
cd /Users/trobacker/GitHub/pymc_modeling/hub_submissions/models/pymc_bart

# 2. Activate environment
source .venv/bin/activate

# 3. Run workflow from scripts directory
cd scripts
python run_workflow.py --nowcast-date 2024-11-13 --mode test
```

## Quick Test

To verify everything is set up correctly:

```bash
# From pymc_bart directory with .venv activated
python -c "
import pymc
import pymc_bart
import polars
import arviz
print('✓ All required packages installed!')
print(f'PyMC: {pymc.__version__}')
print(f'PyMC-BART: {pymc_bart.__version__}')
"
```

## Troubleshooting

### "No module named 'pymc_bart'"

**Problem:** You're not using the model-specific `.venv`

**Solution:**
```bash
cd /Users/trobacker/GitHub/pymc_modeling/hub_submissions/models/pymc_bart
source .venv/bin/activate  # Make sure you see (.venv) in prompt
cd scripts
python run_workflow.py --nowcast-date 2024-11-13 --mode test
```

### Environment is activated but packages missing

**Problem:** Dependencies not installed in this `.venv`

**Solution:**
```bash
# Make sure you're in pymc_bart directory
cd /Users/trobacker/GitHub/pymc_modeling/hub_submissions/models/pymc_bart
source .venv/bin/activate
pip install -r requirements.txt
```

### Which environment am I using?

Check with:
```bash
which python
# Should show: /Users/trobacker/GitHub/pymc_modeling/hub_submissions/models/pymc_bart/.venv/bin/python

echo $VIRTUAL_ENV
# Should show: /Users/trobacker/GitHub/pymc_modeling/hub_submissions/models/pymc_bart/.venv
```

## Deactivating

When done working with the model:

```bash
deactivate  # Returns to system Python
```

## Why Model-Specific Environments?

Each model in this repository can have different dependency requirements:
- `pymc_bart` needs `pymc-bart` (which you're setting up now)
- `pymc_hmlr` models don't need `pymc-bart`
- Different models might need different versions of packages

By using model-specific `.venv` directories, we avoid conflicts and ensure each model has exactly what it needs.
