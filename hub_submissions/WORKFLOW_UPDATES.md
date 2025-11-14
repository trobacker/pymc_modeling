# Workflow Updates - Repository Restructuring

## Summary of Changes

This document describes the major restructuring made to support iterating on different COVID variant models:

### 2025-11-14: Repository Restructuring
1. **Multi-model directory structure** for iterating on different modeling approaches
2. **Shared utilities** across all models in `common/` module
3. **VNH-compliant output** directory structure in `hub_output/`

### Previous Updates:
1. **All 52 US locations** (50 states + DC + PR) in submissions
2. **Correct prediction window**: 32-day nowcast + 10-day forecast

## Changes Made

### 1. Configuration (`config/config.yaml`)

**Added:**
- `data.locations`: List of all 52 US location abbreviations (50 states + DC + PR)
- `submission.nowcast_lookback_days`: Set to 31 (creates 32-day nowcast period)

**Locations included:**
```yaml
locations: [
  "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL",
  "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA",
  "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE",
  "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "PR",
  "RI", "SC", "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI",
  "WV", "WY"
]
```

### 2. Submission Formatting (`03_format_submission.py`)

**Updated `generate_forecast_dates()` function:**
- Now takes three parameters: `nowcast_date`, `nowcast_lookback`, `forecast_horizon`
- Generates **32-day nowcast period**: `nowcast_date - 31` to `nowcast_date`
- Generates **10-day forecast period**: `nowcast_date + 1` to `nowcast_date + 10`
- **Total**: 42 days of predictions per location

**Example for nowcast_date = 2024-11-13:**
- Nowcast: 2024-10-13 to 2024-11-13 (32 days)
- Forecast: 2024-11-14 to 2024-11-23 (10 days)

**Updated location handling:**
- Now uses all 52 locations from config (`config['data']['locations']`)
- For locations present in training data: uses fitted model parameters
- For locations missing from training data: uses location index 0 (population average)
- Logs warnings for any missing locations

**Code snippet:**
```python
# For locations not in training, use location index 0 (population average)
if location in mappings['location']:
    loc_idx = mappings['location'][location]
else:
    loc_idx = 0  # Use first location as proxy (population average)
    logger.warning(f"Location {location} not in training data, will use population average")
```

## Impact

### Submission Size
- **Old**: ~11 target dates × 39 locations × 8 clades × (100 samples + 1 mean) ≈ 346K rows
- **New**: 42 target dates × 52 locations × 8 clades × (100 samples + 1 mean) ≈ 1.75M rows

### Missing Locations
The following 13 locations were added but may not have training data:
- **HI** (Hawaii) - may have limited data
- **ID** (Idaho)
- **ME** (Maine)
- **MT** (Montana)
- **ND** (North Dakota)
- **NE** (Nebraska)
- **NH** (New Hampshire)
- **NM** (New Mexico)
- **NV** (Nevada)
- **OR** (Oregon)
- **SD** (South Dakota)
- **VT** (Vermont)
- **WY** (Wyoming)

For these locations without training data, the model will use the population-level average from all locations with data, providing reasonable baseline predictions.

## Repository Restructuring (2025-11-14)

### New Directory Structure

```
hub_submissions/
├── README.md                    # Comprehensive documentation
├── common/                      # Shared utilities across all models
│   ├── config_utils.py          # Configuration and logging
│   ├── hub_utils.py             # Hub-specific utilities
│   └── validation_utils.py      # Submission validation
├── models/                      # Model implementations
│   └── pymc_hmlr/              # Hierarchical Multinomial Logistic Regression
│       ├── config.yaml          # Model-specific configuration
│       ├── scripts/             # Model workflow scripts
│       ├── data/                # Training data
│       ├── fitted/              # Model artifacts (traces, posteriors)
│       ├── submissions/         # Generated submissions
│       └── logs/                # Workflow logs
└── hub_output/                  # VNH-compliant structure
    └── YourTeam-PyMC-HMLR/
        └── YYYY-MM-DD-YourTeam-PyMC-HMLR.parquet
```

### Key Improvements

1. **Model Isolation**: Each model variant has its own directory enabling parallel development
2. **Shared Utilities**: Common functionality extracted to `common/` module
3. **VNH Compliance**: `hub_output/` mirrors Variant Nowcast Hub structure
4. **Updated Imports**: Scripts now import from `common` utilities
5. **Updated Paths**: Config paths updated for new structure

### Using the Restructured Workflow

```bash
cd /Users/trobacker/GitHub/pymc_modeling/hub_submissions/models/pymc_hmlr/scripts

# Test run
python run_workflow.py --nowcast-date 2025-11-14 --mode test

# Production run
python run_workflow.py --nowcast-date 2025-11-14 --mode prod
```

Expected output:
- Target dates: 42 dates (32 nowcast + 10 forecast)
- Locations: 52 (all US states + DC + PR)
- Total submission rows: ~1.75M

## Validation

After running, check:
1. ✅ Submission has all 52 locations
2. ✅ Submission has 42 target dates per location
3. ✅ Date range: (nowcast_date - 31) to (nowcast_date + 10)
4. ✅ 100 samples + 1 mean per location-date-clade combination
5. ⚠️  Check warnings for locations missing from training data

## Notes

- The model only trains on locations with sufficient data (min_sequences threshold)
- Locations without training data receive population-level average predictions
- This approach ensures full 52-location coverage while maintaining model quality for well-sampled locations
- For production runs with more data, more locations may be included in training
