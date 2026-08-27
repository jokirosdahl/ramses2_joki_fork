#!/usr/bin/env python3
"""
Plot sink particle mass (log scale) versus lookback time (linear scale)
for all sink particle CSV files in a given sinklog directory.

Usage:
    python3 utils/py/plot_sinklog.py [--dir sinklog] [--out sinks.png] [--unit Gyr] [--show]
"""

import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def natural_sort_key(s):
    """Sort strings containing numbers naturally (e.g. sink_2 before sink_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot sink particle mass (log scale) vs lookback time (linear scale)."
    )
    parser.add_argument(
        "--dir", "-d",
        type=str,
        default="sinklog",
        help="Path to directory containing sink_*.csv files (default: 'sinklog')."
    )
    parser.add_argument(
        "--out", "-o",
        type=str,
        default="sink_mass_vs_lookback.png",
        help="Output plot image filename (default: 'sink_mass_vs_lookback.png')."
    )
    parser.add_argument(
        "--unit", "-u",
        type=str,
        choices=["Gyr", "Myr", "yr"],
        default="Gyr",
        help="Time unit for lookback time on x-axis (default: 'Gyr')."
    )
    parser.add_argument(
        "--sinks",
        type=int,
        nargs="+",
        default=None,
        help="Specific sink IDs to plot (e.g., --sinks 1 2 5). Default: plot all."
    )
    parser.add_argument(
        "--forward-time",
        action="store_true",
        help="Invert x-axis so that cosmic time flows from left to right (larger lookback to smaller lookback)."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively with plt.show()."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    sink_dir = args.dir

    if not os.path.isdir(sink_dir):
        raise FileNotFoundError(f"Sinklog directory not found: {sink_dir}")

    # Find all sink CSV files
    file_pattern = os.path.join(sink_dir, "sink_*.csv")
    csv_files = sorted(glob.glob(file_pattern), key=natural_sort_key)

    if not csv_files:
        raise FileNotFoundError(f"No sink_*.csv files found in '{sink_dir}'.")

    # Time conversion factor from years (RAMSES sinklog time unit)
    unit_factors = {
        "yr": 1.0,
        "Myr": 1.0e6,
        "Gyr": 1.0e9,
    }
    time_scale = unit_factors[args.unit]

    # Filter sinks if specified
    if args.sinks is not None:
        target_sinks = set(args.sinks)
        filtered_files = []
        for f in csv_files:
            match = re.search(r"sink_(\d+)\.csv", os.path.basename(f))
            if match and int(match.group(1)) in target_sinks:
                filtered_files.append(f)
        csv_files = filtered_files
        if not csv_files:
            raise ValueError(f"No matching sink CSV files for specified sink IDs: {args.sinks}")

    n_files = len(csv_files)
    print(f"Found {n_files} sink file(s) in '{sink_dir}'.")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)

    # Color map for multiple sinks
    colors = cm.turbo(np.linspace(0.05, 0.95, max(n_files, 1)))

    plotted_count = 0
    for idx, filepath in enumerate(csv_files):
        filename = os.path.basename(filepath)
        sink_match = re.search(r"sink_(\d+)", filename)
        sink_id = int(sink_match.group(1)) if sink_match else idx + 1

        try:
            # Strip whitespace in column names
            df = pd.read_csv(filepath, skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]

            if "time" not in df.columns or "mass" not in df.columns:
                print(f"Warning: Skipping {filename} (missing 'time' or 'mass' column).")
                continue

            if len(df) == 0:
                print(f"Warning: Skipping empty file {filename}.")
                continue

            # Lookback time: in RAMSES cosmological runs, time is negative years (t = 0 at z = 0).
            # Lookback time = |time| / unit
            raw_time = df["time"].values
            lookback_time = np.abs(raw_time) / time_scale
            mass = df["mass"].values

            # Sort by lookback time if needed (time usually advances, lookback decreases)
            sort_idx = np.argsort(lookback_time)
            lookback_time = lookback_time[sort_idx]
            mass = mass[sort_idx]

            label = f"Sink {sink_id}"
            ax.plot(
                lookback_time,
                mass,
                label=label,
                color=colors[idx],
                linewidth=1.5,
                alpha=0.85,
                marker="o" if len(mass) < 30 else None,
                markersize=3,
            )
            plotted_count += 1

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if plotted_count == 0:
        print("No valid sink data was plotted.")
        return

    # Set log scale on y-axis and linear on x-axis
    ax.set_yscale("log")
    ax.set_xlabel(f"Lookback Time [{args.unit}]", fontsize=12)
    ax.set_ylabel(r"Sink Mass [$M_\odot$]", fontsize=12)
    ax.set_title(f"Sink Particle Mass Growth vs Lookback Time ({plotted_count} sinks)", fontsize=13, fontweight="bold")

    if args.forward_time:
        # Invert x-axis so earlier universe (larger lookback) is on the left
        ax.invert_xaxis()
        ax.annotate(
            "Cosmic time $\\rightarrow$",
            xy=(0.02, 0.03),
            xycoords="axes fraction",
            fontsize=10,
            color="gray",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.6)
        )

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    # Place legend
    if plotted_count <= 20:
        ncol = 2 if plotted_count > 10 else 1
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=9,
            frameon=True,
            ncol=ncol,
            title="Sink ID"
        )
    else:
        # Too many sinks for complete legend
        ax.legend(
            loc="best",
            fontsize=8,
            frameon=True,
            title="Sinks"
        )

    plt.tight_layout()

    # Save output
    if args.out:
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
        print(f"Plot saved to '{args.out}'")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
