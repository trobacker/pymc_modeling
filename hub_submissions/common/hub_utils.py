"""Variant Nowcast Hub specific utilities"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import logging


def get_modeled_clades(hub_repo_path, nowcast_date, logger=None):
    """
    Load the required clades from the hub's modeled-clades JSON file.

    According to hub rules, submissions use the modeled-clades file from 2 days prior
    to the nowcast date. For example, a nowcast on 2025-11-14 uses 2025-11-12.json.

    Args:
        hub_repo_path: Path to variant-nowcast-hub repository
        nowcast_date: Date string in YYYY-MM-DD format
        logger: Optional logger instance

    Returns:
        List of clade names to model
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    nowcast_dt = datetime.strptime(nowcast_date, "%Y-%m-%d")
    modeled_clades_date = nowcast_dt - timedelta(days=2)
    modeled_clades_date_str = modeled_clades_date.strftime("%Y-%m-%d")

    modeled_clades_file = Path(hub_repo_path) / "auxiliary-data" / "modeled-clades" / f"{modeled_clades_date_str}.json"

    if not modeled_clades_file.exists():
        raise FileNotFoundError(
            f"Modeled clades file not found: {modeled_clades_file}\n"
            f"Expected date: {modeled_clades_date_str} (2 days before nowcast date {nowcast_date})"
        )

    with open(modeled_clades_file, 'r') as f:
        data = json.load(f)

    clades = data.get('clades', [])
    logger.info(f"Loaded {len(clades)} modeled clades from {modeled_clades_date_str}: {clades}")

    return clades


def get_s3_data_url(date_str):
    """
    Get S3 URL for COVID clade count data.

    Data is uploaded weekly on Mondays.

    Args:
        date_str: Date string in YYYY-MM-DD format (should be a Monday)

    Returns:
        S3 URL string
    """
    return f"https://covid-clade-counts.s3.amazonaws.com/{date_str}_covid_clade_counts.parquet"


def find_most_recent_monday(date_str):
    """
    Find the most recent Monday on or before the given date.

    Args:
        date_str: Date string in YYYY-MM-DD format

    Returns:
        Date string in YYYY-MM-DD format (Monday)
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")

    # Monday is 0 in weekday()
    days_since_monday = dt.weekday()
    monday_dt = dt - timedelta(days=days_since_monday)

    return monday_dt.strftime("%Y-%m-%d")
