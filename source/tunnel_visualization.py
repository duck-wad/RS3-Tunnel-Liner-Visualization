"""
CLI entry for tunnel visualization.

Prefer the web viewer for RS3 model extract:
    python run_tunnel_viewer.py

This script keeps the original CSV workflow (prompt for a filename, open Plotly).
"""

from __future__ import annotations

import os
import sys

from tunnel_viz.plotting import build_tunnel_figure, dataframe_from_csv


def main() -> int:
    csv_filename = input("Enter name of CSV file: ").strip()
    if not csv_filename:
        print("No filename provided.")
        return 1
    if not csv_filename.endswith(".csv"):
        csv_filename += ".csv"
    if not os.path.isfile(csv_filename):
        print(f"Error: File '{csv_filename}' not found in current directory.")
        return 1

    print(f"Loading {csv_filename}...")
    df = dataframe_from_csv(csv_filename)
    print(f"Loaded {len(df)} points from CSV")
    print("Building interactive visualization...")
    fig = build_tunnel_figure(df, progress_callback=print)
    print("Launching interactive visualization...")
    fig.show()
    return 0


if __name__ == "__main__":
    # Allow `python tunnel_visualization.py` from the source/ directory.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    raise SystemExit(main())
