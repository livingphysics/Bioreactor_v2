"""
Example usage:
    python fit_calibration.py

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
    calib_dir = 'weekend_calibration'
    files = sorted(glob.glob(os.path.join(calib_dir, '*.csv')))
    n_files = len(files)
    if n_files == 0:
        print(f"No CSV files found in {calib_dir}")
        sys.exit(1)

    plots_per_page = 8
    n_pages = int(np.ceil(n_files / plots_per_page))

    # Precompute all axes and plot content
    fig, axes = plt.subplots(2, 4, figsize=(24, 8), squeeze=False)
    axes = axes.flatten()
    plot_handles = []
    titles = []
    for i, file in tqdm(enumerate(files)):
        base = os.path.basename(file)
        title = base.replace('pump_', 'Pump ').replace('_', ' ').replace('.csv', '').title()
        titles.append(title)
        analyze_and_plot(axes[i % plots_per_page], file, title=title)
        if i % plots_per_page == 0:
            axes[i % plots_per_page].legend()
        plot_handles.append(axes[i % plots_per_page])
    # Hide all axes initially
    for ax in axes:
        ax.set_visible(False)

    # Button callback logic
    current_page = [0]  # Use list for mutability in closure

    def show_page(page):
        for i, ax in enumerate(axes):
            ax.clear()
            ax.set_visible(False)
        start = page * plots_per_page
        end = min(start + plots_per_page, n_files)
        for i, file_idx in enumerate(range(start, end)):
            analyze_and_plot(axes[i], files[file_idx], title=titles[file_idx])
            if i == 0:
                axes[i].legend()
            axes[i].set_visible(True)
        fig.suptitle(f'Calibration Plots (Page {page+1} of {n_pages})', fontsize=16)
        plt.draw()

    def next_page(event):
        if current_page[0] < n_pages - 1:
            current_page[0] += 1
            show_page(current_page[0])

    def prev_page(event):
        if current_page[0] > 0:
            current_page[0] -= 1
            show_page(current_page[0])

    # Add buttons
    axprev = plt.axes([0.4, 0.01, 0.1, 0.05])
    axnext = plt.axes([0.5, 0.01, 0.1, 0.05])
    bnext = Button(axnext, 'Next')
    bprev = Button(axprev, 'Previous')
    bnext.on_clicked(next_page)
    bprev.on_clicked(prev_page)

    show_page(0)
    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    plt.show()

if __name__ == '__main__':
    main()
