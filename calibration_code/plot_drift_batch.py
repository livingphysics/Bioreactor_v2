"""
Example usage:
    python -m calibration_code.plot_drift_batch A [--combined]

Plots the most recent drift results for a given letter (A, B, C, or D) for all flow rates (4, 6, 8, ..., 20).
By default, each flow rate is plotted in a separate subplot (3x3 grid). Use --combined to plot all trends on the same axes.
If a file is missing for a flow rate, the subplot is left blank with a message.
"""
import os
import re
import sys
import importlib.util
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
from calibration_code.calibration_utils import plot_with_fit_and_bands, find_most_recent_file
import pandas as pd

# --- Parse command-line argument (letter and mode) ---
def get_letter_and_mode():
    parser = argparse.ArgumentParser()
    parser.add_argument('letter', type=str, help='A, B, C, or D')
    parser.add_argument('--combined', action='store_true', help='Plot all flow rates on the same axes')
    args = parser.parse_args()
    letter = args.letter.upper()
    if letter not in {'A', 'B', 'C', 'D'}:
        print("Letter must be one of: A, B, C, D")
        sys.exit(1)
    return letter, args.combined

# --- Main CLI app ---
def main():
    letter, combined = get_letter_and_mode()
    folder = os.path.join(os.path.dirname(__file__), '..', 'drift_results')
    folder = os.path.abspath(folder)
    flow_rates = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    files = []
    dates = []
    print(f"Looking for most recent files for letter {letter} and flow rates {flow_rates}...")
    pattern_template = "{date}_drift_{letter}_{flow}_0_results.csv"
    for flow in flow_rates:
        f, d = find_most_recent_file(folder, pattern_template, {"letter": letter, "flow": flow})
        files.append(f)
        dates.append(d)
    if all(f is None for f in files):
        print(f"No drift result files found for letter {letter}.")
        sys.exit(1)
    if combined:
        print("Generating combined plot...")
        import matplotlib.cm as cm
        cmap = cm.get_cmap('tab10', len(flow_rates))
        fig, ax = plt.subplots(figsize=(12, 8))
        for idx, (flow, path, date) in enumerate(zip(flow_rates, files, dates)):
            if path is not None:
                try:
                    df = pd.read_csv(path, comment='#')
                    x = df['time_s'].values
                    y = df['delta_mass_g'].values
                    label = f"{flow} uL/s (Date: {date})"
                    color = cmap(idx)
                    plot_with_fit_and_bands(x, y, ax=ax, label=label, color=color, show_title=False, show_stats_table=False)
                except Exception as e:
                    print(f"Error plotting {flow} uL/s: {e}")
            else:
                print(f"No data for {flow} uL/s")
        ax.set_title(f"Drift Results for {letter}", fontsize=20)
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Generating separate plots...")
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()
        for i, (flow, path, date) in enumerate(zip(flow_rates, files, dates)):
            ax = axes[i]
            if path is not None:
                try:
                    df = pd.read_csv(path, comment='#')
                    x = df['time_s'].values
                    y = df['delta_mass_g'].values
                    label = f"{flow} uL/s (Date: {date})"
                    plot_with_fit_and_bands(x, y, ax=ax, label=label, color='b', show_title=True, show_stats_table=False)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
                    ax.set_title(f"{letter}, {flow} uL/s\n(Date: {date})")
            else:
                ax.text(0.5, 0.5, f"No data for {flow} uL/s", ha='center', va='center', fontsize=14)
                ax.set_title(f"{letter}, {flow} uL/s\n(No file)")
        plt.tight_layout()
        plt.suptitle(f"Drift Results for {letter}", fontsize=20, y=1.02)
        plt.show()

if __name__ == "__main__":
    main()
