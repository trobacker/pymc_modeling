"""Validation utilities for hub submissions"""

import polars as pl
import logging


def validate_submission_schema(df, logger=None):
    """
    Validate that submission dataframe matches hub requirements.

    Args:
        df: Polars DataFrame to validate
        logger: Optional logger instance

    Returns:
        Tuple of (is_valid: bool, messages: list)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    messages = []
    is_valid = True

    # Required columns
    required_cols = [
        'nowcast_date', 'target_date', 'location', 'clade',
        'output_type', 'output_type_id', 'value'
    ]

    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        messages.append(f"Missing required columns: {missing_cols}")
        is_valid = False
        return is_valid, messages

    # Check for nulls in required columns
    for col in required_cols:
        null_count = df[col].null_count()
        if null_count > 0:
            messages.append(f"Column '{col}' has {null_count} null values")
            is_valid = False

    # Check output_type values
    valid_output_types = {'mean', 'quantile', 'sample'}
    output_types = set(df['output_type'].unique().to_list())
    invalid_types = output_types - valid_output_types
    if invalid_types:
        messages.append(f"Invalid output_type values: {invalid_types}")
        is_valid = False

    # Check value range [0, 1] for proportions
    value_min = df['value'].min()
    value_max = df['value'].max()
    if value_min < 0 or value_max > 1:
        messages.append(f"Values out of range [0,1]: min={value_min}, max={value_max}")
        is_valid = False

    # Check date formats
    try:
        df['nowcast_date'].str.strptime(pl.Date, "%Y-%m-%d")
    except Exception as e:
        messages.append(f"Invalid nowcast_date format: {e}")
        is_valid = False

    try:
        df['target_date'].str.strptime(pl.Date, "%Y-%m-%d")
    except Exception as e:
        messages.append(f"Invalid target_date format: {e}")
        is_valid = False

    # Log results
    if is_valid:
        logger.info("✓ Submission validation passed")
    else:
        logger.error("✗ Submission validation failed:")
        for msg in messages:
            logger.error(f"  - {msg}")

    return is_valid, messages
