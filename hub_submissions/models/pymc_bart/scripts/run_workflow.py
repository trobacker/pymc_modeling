#!/usr/bin/env python3
"""
Run Complete Workflow for BART Variant Nowcast Model
=============================================================================

This script orchestrates the complete workflow:
1. Fetch training data from S3
2. Fit BART model
3. Format submission for hub

Prerequisites:
    Activate the virtual environment first:
    $ source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows

Usage:
    python run_workflow.py --nowcast-date 2024-11-13
    python run_workflow.py --nowcast-date 2024-11-13 --mode prod
    python run_workflow.py --nowcast-date 2024-11-13 --skip-data --skip-model

Author: Auto-generated workflow
Date: 2025-12-31
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a shell command and handle errors"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Run complete BART variant nowcast workflow"
    )
    parser.add_argument(
        "--nowcast-date",
        type=str,
        required=True,
        help="Nowcast date in YYYY-MM-DD format (e.g., 2024-11-13)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (default: ../config.yaml)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['test', 'prod'],
        default=None,
        help="Modeling mode: test or prod (overrides config)"
    )
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data fetching step (use existing data)"
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model fitting step (use existing model)"
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=None,
        help="Forecast horizon in days (overrides config)"
    )

    args = parser.parse_args()
    logger = setup_logging()

    # Determine config path
    if args.config is None:
        # Default: look for config.yaml in parent directory (if running from scripts/)
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / "config.yaml"
        if not config_path.exists():
            # Fallback: current directory
            config_path = Path("config.yaml")
        args.config = str(config_path)
    else:
        config_path = Path(args.config)
        args.config = str(config_path)

    # Get script directory
    script_dir = Path(__file__).parent

    logger.info("=" * 70)
    logger.info("BART VARIANT NOWCAST WORKFLOW")
    logger.info("=" * 70)
    logger.info(f"Nowcast date: {args.nowcast_date}")
    logger.info(f"Mode: {args.mode or 'from config'}")
    logger.info(f"Config: {args.config}")
    logger.info("=" * 70)

    start_time = datetime.now()

    # Step 1: Fetch data
    if not args.skip_data:
        logger.info("\n[Step 1/3] Fetching training data...")
        cmd = [
            sys.executable,
            str(script_dir / "01_fetch_data.py"),
            "--nowcast-date", args.nowcast_date,
            "--config", args.config
        ]
        if not run_command(cmd, "Data fetching"):
            logger.error("Workflow failed at data fetching step")
            return 1
    else:
        logger.info("\n[Step 1/3] Skipping data fetching (--skip-data)")

    # Step 2: Fit model
    if not args.skip_model:
        logger.info("\n[Step 2/3] Fitting BART model...")
        cmd = [
            sys.executable,
            str(script_dir / "02_fit_model.py"),
            "--nowcast-date", args.nowcast_date,
            "--config", args.config
        ]
        if args.mode:
            cmd.extend(["--mode", args.mode])

        if not run_command(cmd, "Model fitting"):
            logger.error("Workflow failed at model fitting step")
            return 1
    else:
        logger.info("\n[Step 2/3] Skipping model fitting (--skip-model)")

    # Step 3: Format submission
    logger.info("\n[Step 3/3] Formatting submission...")
    cmd = [
        sys.executable,
        str(script_dir / "03_format_submission.py"),
        "--nowcast-date", args.nowcast_date,
        "--config", args.config
    ]
    if args.forecast_horizon:
        cmd.extend(["--forecast-horizon", str(args.forecast_horizon)])

    if not run_command(cmd, "Submission formatting"):
        logger.error("Workflow failed at submission formatting step")
        return 1

    # Success!
    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("\n" + "=" * 70)
    logger.info("✓ WORKFLOW COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Nowcast date: {args.nowcast_date}")
    logger.info(f"Total duration: {duration}")
    logger.info(f"Submission ready in: submissions/")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
