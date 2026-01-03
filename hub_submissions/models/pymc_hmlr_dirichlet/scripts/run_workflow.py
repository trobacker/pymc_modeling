#!/usr/bin/env python3
"""
Run Complete Variant Nowcast Hub Submission Workflow
=============================================================================

This script orchestrates the complete workflow:
1. Fetch training data from hub
2. Fit PyMC model
3. Format and validate submission

Prerequisites:
    Activate the virtual environment first:
    $ source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows

Usage:
    python run_workflow.py --nowcast-date 2024-11-13
    python run_workflow.py --nowcast-date 2024-11-13 --mode prod
    python run_workflow.py --nowcast-date 2024-11-13 --mode prod --skip-fetch

Author: Auto-generated workflow
Date: 2025-11-14
"""

import argparse
import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import common utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from common.config_utils import setup_logging, load_config


def run_command(cmd, step_name):
    """Run a subprocess command and handle errors"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running: {step_name}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )

        # Print stdout if there was any
        if result.stdout:
            print(result.stdout)

        logger.info(f"✓ {step_name} completed successfully")
        return 0

    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {step_name} failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return e.returncode


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Run complete variant nowcast hub submission workflow"
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
        default="../config.yaml",
        help="Path to configuration file (relative to scripts directory)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['test', 'prod'],
        default=None,
        help="Modeling mode: test (quick) or prod (full run)"
    )
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="Skip data fetching (use existing training data)"
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model fitting (use existing model artifacts)"
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=None,
        help="Forecast horizon in days"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    logger = setup_logging(config['logging']['level'])

    print("\n" + "=" * 70)
    print("VARIANT NOWCAST HUB SUBMISSION WORKFLOW")
    print("=" * 70)
    print(f"Nowcast date: {args.nowcast_date}")
    print(f"Mode: {args.mode or config['modeling']['mode']}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    scripts_dir = Path(__file__).parent
    exit_code = 0

    try:
        # Step 1: Fetch data
        if not args.skip_fetch:
            print("\n" + "─" * 70)
            print("STEP 1: FETCHING TRAINING DATA")
            print("─" * 70)

            cmd = [
                "python3",
                str(scripts_dir / "01_fetch_data.py"),
                "--nowcast-date", args.nowcast_date,
                "--config", args.config
            ]

            exit_code = run_command(cmd, "Data fetching")
            if exit_code != 0:
                return exit_code
        else:
            logger.info("Skipping data fetching (--skip-fetch)")

        # Step 2: Fit model
        if not args.skip_model:
            print("\n" + "─" * 70)
            print("STEP 2: FITTING PyMC MODEL")
            print("─" * 70)

            cmd = [
                "python3",
                str(scripts_dir / "02_fit_model.py"),
                "--nowcast-date", args.nowcast_date,
                "--config", args.config
            ]

            if args.mode:
                cmd.extend(["--mode", args.mode])

            exit_code = run_command(cmd, "Model fitting")
            if exit_code != 0:
                return exit_code
        else:
            logger.info("Skipping model fitting (--skip-model)")

        # Step 3: Format submission
        print("\n" + "─" * 70)
        print("STEP 3: FORMATTING SUBMISSION")
        print("─" * 70)

        cmd = [
            "python3",
            str(scripts_dir / "03_format_submission.py"),
            "--nowcast-date", args.nowcast_date,
            "--config", args.config
        ]

        if args.forecast_horizon:
            cmd.extend(["--forecast-horizon", str(args.forecast_horizon)])

        exit_code = run_command(cmd, "Submission formatting")
        if exit_code != 0:
            return exit_code

        # Success!
        print("\n" + "=" * 70)
        print("✓ WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        model_name = f"{config['model']['team']}-{config['model']['name']}"
        submission_file = (Path(config['paths']['submissions']) /
                           f"{args.nowcast_date}-{model_name}.{config['submission']['format']}")

        print(f"\nSubmission file ready:")
        print(f"  {submission_file}")
        print(f"\nNext steps:")
        print(f"  1. Review the submission file")
        print(f"  2. Copy to hub repository: {config['hub']['repo_path']}/model-output/{model_name}/")
        print(f"  3. Create pull request to submit")
        print("=" * 70 + "\n")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nWorkflow interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error in workflow: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
