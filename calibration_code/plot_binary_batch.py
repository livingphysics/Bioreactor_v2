import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from calibration_code.calibration_utils import plot_with_fit_and_bands, find_most_recent_file

LETTERS = ['A', 'B', 'C', 'D']
STEP_RATES = [100, 200, 300, 400, 500]  # Example step rates; adjust as needed
BINARY_DRIFT_DIR = 'binary_drift_results'
PATTERN_TEMPLATE = '{date}_binary_drift_{letter}_{in_rate}_{out_rate}_results.csv'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--combined', action='store_true', help='Plot all data on one plot (default: separate subplots)')
    args = parser.parse_args()
    files = []
    meta = []
    print("Searching for most recent binary drift results...")
    for letter in LETTERS:
        for steps_rate in STEP_RATES:
            # Find the most recent file for this letter and steps_rate (in_rate == steps_rate)
            # out_rate is unknown, so we need to search for all files matching letter and in_rate
            found = False
            for fname in sorted(os.listdir(BINARY_DRIFT_DIR), reverse=True):
                if f'_binary_drift_{letter}_{steps_rate}_' in fname and fname.endswith('_results.csv'):
                    files.append(os.path.join(BINARY_DRIFT_DIR, fname))
                    meta.append((letter, steps_rate, fname))
                    found = True
                    break
            if not found:
                files.append(None)
                meta.append((letter, steps_rate, None))
    n = len(LETTERS) * len(STEP_RATES)
    if args.combined:
        print("Plotting all data in combined mode...")
        fig, ax = plt.subplots(figsize=(12, 8))
        for (letter, steps_rate, fname), path in zip(meta, files):
            if path is not None:
                df = pd.read_csv(path, comment='#')
                x = df['time_s'].values
                y = df['delta_mass_g'].values
                label = f"{letter}, {steps_rate} (file: {os.path.basename(path)})"
                plot_with_fit_and_bands(x, y, ax=ax, label=label, show_title=False, show_stats_table=False)
            else:
                print(f"No data for {letter} at {steps_rate}")
        ax.set_title("Binary Drift Results (Combined)")
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Plotting all data in separate subplots...")
        nrows = len(LETTERS)
        ncols = len(STEP_RATES)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
        for idx, ((letter, steps_rate, fname), path) in enumerate(zip(meta, files)):
            row = LETTERS.index(letter)
            col = STEP_RATES.index(steps_rate)
            ax = axes[row][col]
            if path is not None:
                df = pd.read_csv(path, comment='#')
                x = df['time_s'].values
                y = df['delta_mass_g'].values
                label = f"{letter}, {steps_rate}"
                plot_with_fit_and_bands(x, y, ax=ax, label=label, show_title=True, show_stats_table=False)
            else:
                ax.text(0.5, 0.5, f"No data for {letter}, {steps_rate}", ha='center', va='center')
            ax.set_title(f"{letter}, {steps_rate}")
        plt.tight_layout()
        plt.suptitle("Binary Drift Results (Separate)", fontsize=20, y=1.02)
        plt.show()

if __name__ == "__main__":
    main() 
