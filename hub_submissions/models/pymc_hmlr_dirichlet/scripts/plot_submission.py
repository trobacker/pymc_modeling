#!/usr/bin/env python3
"""
Plot Model Submission Results
=============================================================================

Generate visualizations of clade trajectory predictions from a submission file.

Usage:
    python plot_submission.py --submission submissions/2025-11-14-YourTeam-PyMC-HMLR.parquet

Author: Auto-generated workflow
"""

import argparse
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def load_submission(submission_path):
    """Load submission parquet file"""
    df = pl.read_parquet(submission_path)
    print(f"Loaded submission: {df.shape}")
    print(f"  Locations: {df['location'].n_unique()}")
    print(f"  Clades: {df['clade'].unique().to_list()}")
    print(f"  Date range: {df['target_date'].min()} to {df['target_date'].max()}")
    return df


def plot_clade_trajectories_by_location(df, locations, output_path):
    """
    Plot clade trajectories for selected locations.

    Each subplot shows all clades for one location over time.
    """
    # Filter to mean predictions only
    df_mean = df.filter(pl.col("output_type") == "mean")

    # Convert target_date to datetime
    df_mean = df_mean.with_columns(
        pl.col("target_date").str.strptime(pl.Date, "%Y-%m-%d").alias("date")
    )

    # Get all clades
    clades = sorted(df_mean["clade"].unique().to_list())

    # Create color map for clades
    colors = plt.cm.tab10(np.linspace(0, 1, len(clades)))
    clade_colors = dict(zip(clades, colors))

    # Create subplots
    n_locs = len(locations)
    fig, axes = plt.subplots(n_locs, 1, figsize=(12, 4 * n_locs))

    if n_locs == 1:
        axes = [axes]

    for ax, location in zip(axes, locations):
        # Filter for this location
        df_loc = df_mean.filter(pl.col("location") == location)

        if len(df_loc) == 0:
            ax.text(0.5, 0.5, f"No data for {location}",
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Plot each clade
        for clade in clades:
            df_clade = df_loc.filter(pl.col("clade") == clade).sort("date")

            dates = df_clade["date"].to_list()
            values = df_clade["value"].to_list()

            ax.plot(dates, values, label=clade, color=clade_colors[clade],
                   linewidth=2, marker='o', markersize=3)

        ax.set_title(f"Clade Trajectories: {location}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Proportion", fontsize=11)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved location trajectories to: {output_path}")
    plt.close()


def plot_clade_trajectories_national(df, output_path):
    """
    Plot national-level clade trajectories (averaged across all locations).
    """
    # Filter to mean predictions only
    df_mean = df.filter(pl.col("output_type") == "mean")

    # Convert target_date to datetime
    df_mean = df_mean.with_columns(
        pl.col("target_date").str.strptime(pl.Date, "%Y-%m-%d").alias("date")
    )

    # Compute national average by date-clade
    df_national = df_mean.group_by(["date", "clade"]).agg(
        pl.col("value").mean().alias("mean_proportion")
    ).sort(["clade", "date"])

    # Get all clades
    clades = sorted(df_national["clade"].unique().to_list())

    # Create color map for clades
    colors = plt.cm.tab10(np.linspace(0, 1, len(clades)))
    clade_colors = dict(zip(clades, colors))

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    for clade in clades:
        df_clade = df_national.filter(pl.col("clade") == clade).sort("date")

        dates = df_clade["date"].to_list()
        values = df_clade["mean_proportion"].to_list()

        ax.plot(dates, values, label=clade, color=clade_colors[clade],
               linewidth=2.5, marker='o', markersize=4)

    ax.set_title("National Clade Trajectories (Mean Across All Locations)",
                fontsize=16, fontweight='bold')
    ax.set_xlabel("Date", fontsize=13)
    ax.set_ylabel("Proportion", fontsize=13)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved national trajectories to: {output_path}")
    plt.close()


def compute_growth_rates(df):
    """
    Compute growth rates for each clade (comparing first and last date).
    """
    # Filter to mean predictions only
    df_mean = df.filter(pl.col("output_type") == "mean")

    # Get first and last dates
    dates = sorted(df_mean["target_date"].unique().to_list())
    first_date = dates[0]
    last_date = dates[-1]

    # Compute national averages at first and last dates
    df_first = df_mean.filter(pl.col("target_date") == first_date).group_by("clade").agg(
        pl.col("value").mean().alias("prop_first")
    )

    df_last = df_mean.filter(pl.col("target_date") == last_date).group_by("clade").agg(
        pl.col("value").mean().alias("prop_last")
    )

    # Join and compute percent change
    df_growth = df_first.join(df_last, on="clade").with_columns(
        ((pl.col("prop_last") - pl.col("prop_first")) / pl.col("prop_first") * 100).alias("percent_change")
    ).sort("percent_change", descending=True)

    print("\n" + "=" * 70)
    print("CLADE GROWTH RATES (National Average)")
    print("=" * 70)
    print(f"Period: {first_date} to {last_date}")
    print(f"Duration: {len(dates)} days\n")

    for row in df_growth.iter_rows(named=True):
        clade = row["clade"]
        prop_first = row["prop_first"]
        prop_last = row["prop_last"]
        pct_change = row["percent_change"]

        direction = "↑" if pct_change > 0 else "↓" if pct_change < 0 else "→"

        print(f"  {clade:15s} {direction} {pct_change:+7.1f}%  "
              f"({prop_first:.3f} → {prop_last:.3f})")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Plot model submission results"
    )
    parser.add_argument(
        "--submission",
        type=str,
        required=True,
        help="Path to submission parquet file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/trobacker/Desktop",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--locations",
        type=str,
        nargs="+",
        default=["CA", "NY", "FL", "TX"],
        help="Locations to plot individually"
    )

    args = parser.parse_args()

    # Load submission
    print(f"Loading submission from: {args.submission}")
    df = load_submission(args.submission)

    # Extract nowcast date from filename
    submission_path = Path(args.submission)
    nowcast_date = submission_path.stem.split("-")[0]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute and print growth rates
    compute_growth_rates(df)

    # Plot national trajectories
    print("\nGenerating national trajectory plot...")
    national_output = output_dir / f"national_trajectories_{nowcast_date}.png"
    plot_clade_trajectories_national(df, national_output)

    # Plot individual location trajectories
    print("\nGenerating location trajectory plots...")
    location_output = output_dir / f"location_trajectories_{nowcast_date}.png"
    plot_clade_trajectories_by_location(df, args.locations, location_output)

    print("\n✓ All plots generated successfully!")


if __name__ == "__main__":
    main()
