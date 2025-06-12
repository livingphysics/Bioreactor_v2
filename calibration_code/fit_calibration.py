"""
Example usage:
    python -m calibration_code.fit_calibration

This script will automatically find all CSV calibration files in the 'calibration_data' directory and plot each as a separate subplot in a single matplotlib window.
If there are more than 8 files, use the Next/Previous buttons to flip between pages of plots.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import t
import glob
from matplotlib.widgets import Button
import argparse
from src.config import Config as cfg


def analyze_and_plot(ax, input_csv, title=None):
    # Load experimental data
    df = pd.read_csv(input_csv, comment='#')
    X = df[['steps_rate']].values
    y = df['ml_rate'].values

    # Convert units
    X_thousands = X.flatten() / 1000  # thousands of steps/sec
    y_ul_s = y * 1000  # microlitres/sec

    # Fit linear model
    model = LinearRegression()
    model.fit(X_thousands.reshape(-1, 1), y_ul_s)
    gradient = model.coef_[0]
    intercept = model.intercept_
    y_pred = model.predict(X_thousands.reshape(-1, 1))

    # Fit quality metrics
    r2 = model.score(X_thousands.reshape(-1, 1), y_ul_s)
    rmse = np.sqrt(np.mean((y_ul_s - y_pred) ** 2))

    # --- Wild bootstrap prediction band ---
    x_grid = np.linspace(X_thousands.min(), X_thousands.max(), 200)
    X_grid = x_grid.reshape(-1, 1)
    n_boot = 2000
    y_boot = np.empty((n_boot, x_grid.size))
    residuals = y_ul_s - y_pred
    X_col = X_thousands.reshape(-1, 1)
    rng = np.random.default_rng()
    for b in range(n_boot):
        v = rng.choice([-1, 1], size=len(y_ul_s))
        y_star = y_pred + residuals * v
        model_b = LinearRegression().fit(X_col, y_star)
        res_b = y_star - model_b.predict(X_col)
        idx = rng.integers(0, len(res_b), size=x_grid.size)
        eps_new = rng.choice([-1, 1], size=x_grid.size) * np.abs(res_b[idx])
        y_boot[b] = model_b.predict(X_grid) + eps_new
    band_low, band_high = np.percentile(y_boot, [2.5, 97.5], axis=0)

    # --- Homoskedastic (classic) prediction band ---
    n = len(X_thousands)
    dof = n - 2
    s = np.sqrt(np.sum((y_ul_s - y_pred)**2) / dof)
    x_bar = X_thousands.mean()
    Sxx = np.sum((X_thousands - x_bar)**2)
    y_grid = intercept + gradient * x_grid
    se_pred = s * np.sqrt(1 + 1/n + (x_grid - x_bar)**2 / Sxx)
    t_crit = t.ppf(0.975, dof)
    low = y_grid - t_crit * se_pred
    high = y_grid + t_crit * se_pred

    # Plot both bands
    ax.scatter(X_thousands, y_ul_s, color='royalblue', marker='+', label='data')
    ax.plot(x_grid, intercept + gradient * x_grid, color='crimson', label='least-squares line')
    ax.fill_between(x_grid, low, high, color='gold', alpha=0.25, label='95 % prediction band (classic)')
    ax.fill_between(x_grid, band_low, band_high, color='lightcoral', alpha=0.25, label='95 % prediction band (wild bootstrap)')
    ax.set_xlabel('Step rate (thousands s⁻¹)')
    ax.set_ylabel('Actual rate (µL s⁻¹)')
    eqn = f"y = {gradient:.8f}x + {intercept:.8f}\nR²={r2:.3f}, RMSE={rmse:.2f}"
    ax.text(0.05, 0.95, eqn, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    if title:
        ax.set_title(title)
    # Only add legend to the first subplot (handled in main)


def main():
    parser = argparse.ArgumentParser(description='Plot calibration fits for up to 8 CSVs.')
    parser.add_argument('csvs', nargs='*', help='Calibration CSV files to plot (max 8).')
    args = parser.parse_args()

    # If user provides CSVs, use those (up to 8)
    if args.csvs:
        files = args.csvs[:8]
        if len(args.csvs) > 8:
            print('Warning: More than 8 files provided. Only the first 8 will be plotted.')
        titles = [os.path.basename(f).replace('pump_', 'Pump ').replace('_', ' ').replace('.csv', '').title() for f in files]
    else:
        # No files provided: auto-select most recent for each pump
        pump_keys = list(cfg.PUMPS.keys())
        calib_dir = 'calibration_results'
        files = []
        titles = []
        for pump in pump_keys:
            # Find all matching CSVs for this pump (forward direction)
            pattern = f"{calib_dir}/*calibration_{pump}_forward_results.csv"
            matches = sorted(glob.glob(pattern))
            if matches:
                most_recent = max(matches, key=os.path.getmtime)
                files.append(most_recent)
                titles.append(pump.replace('_', ' ').title())
            if len(files) == 8:
                break
        if not files:
            print(f"No calibration CSVs found in {calib_dir}")
            sys.exit(1)

    n_files = len(files)
    # Choose layout: 1xN, 2x4, or 4x2 depending on n_files
    if n_files <= 2:
        nrows, ncols = 1, n_files
    elif n_files <= 4:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), squeeze=False)
    axes = axes.flatten()
    for i, (file, title) in enumerate(zip(files, titles)):
        analyze_and_plot(axes[i], file, title=title)
        axes[i].legend()
        axes[i].set_visible(True)
    # Hide unused axes
    for j in range(n_files, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Calibration Plots', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == '__main__':
    main()
