"""
Example usage:
    python -m calibration_code.plot_drift <csv_file> [x_min] [x_max]

Plots drift data from a CSV file, with optional time range selection.
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import t
import matplotlib.colors as mcolors


def get_drift_stats(csv_file, x_min=None, x_max=None):
    """
    Returns (rmse, avg_homoskedastic_band_width/2) for the linear fit to the drift data in csv_file.
    """
    df = pd.read_csv(csv_file, comment='#')
    if not {'time_s', 'delta_mass_g'}.issubset(df.columns):
        raise ValueError("CSV must contain 'time_s' and 'delta_mass_g' columns.")
    x = df['time_s'].values
    y = df['delta_mass_g'].values

    if x_min is not None:
        mask = x >= x_min
    else:
        mask = np.ones_like(x, dtype=bool)
    if x_max is not None:
        mask = mask & (x <= x_max)
    x_fit = x[mask]
    y_fit = y[mask]

    model = LinearRegression()
    X_fit = x_fit.reshape(-1, 1)
    model.fit(X_fit, y_fit)
    y_pred_fit = model.predict(X_fit)
    rmse = np.sqrt(np.mean((y_fit - y_pred_fit) ** 2))

    # Homoskedastic (classic) prediction band
    x_grid = np.linspace(x_fit.min(), x_fit.max(), 200)
    n = len(x_fit)
    dof = n - 2
    s = np.sqrt(np.sum((y_fit - y_pred_fit) ** 2) / dof)
    x_bar = x_fit.mean()
    Sxx = np.sum((x_fit - x_bar) ** 2)
    se_pred = s * np.sqrt(1 + 1/n + (x_grid - x_bar) ** 2 / Sxx)
    t_crit = t.ppf(0.975, dof)
    y_grid = model.predict(x_grid.reshape(-1, 1))
    low = y_grid - t_crit * se_pred
    high = y_grid + t_crit * se_pred
    avg_band_width = np.mean(high - low)
    return rmse, avg_band_width / 2


def plot_drift(csv_file, x_min=None, x_max=None, ax=None, label=None, show_title=True, color='b'):
    # Read the CSV file
    df = pd.read_csv(csv_file, comment='#')
    # Check if required columns exist
    if not {'time_s', 'delta_mass_g'}.issubset(df.columns):
        raise ValueError("CSV must contain 'time_s' and 'delta_mass_g' columns.")
    x = df['time_s'].values
    y = df['delta_mass_g'].values

    # Select range for regression if specified
    if x_min is not None:
        mask = x >= x_min
    else:
        mask = np.ones_like(x, dtype=bool)
    if x_max is not None:
        mask = mask & (x <= x_max)
    x_fit = x[mask]
    y_fit = y[mask]

    # Fit linear regression on selected range
    model = LinearRegression()
    X_fit = x_fit.reshape(-1, 1)
    model.fit(X_fit, y_fit)
    gradient = model.coef_[0]
    intercept = model.intercept_
    y_pred_fit = model.predict(X_fit)
    r2 = model.score(X_fit, y_fit)
    rmse = np.sqrt(np.mean((y_fit - y_pred_fit) ** 2))

    # Grid for smooth line and bands (over fit range)
    x_grid = np.linspace(x_fit.min(), x_fit.max(), 200)
    X_grid = x_grid.reshape(-1, 1)
    y_grid = model.predict(X_grid)

    # --- Wild bootstrap (heteroskedastic) prediction band ---
    n_boot = 2000
    y_boot = np.empty((n_boot, x_grid.size))
    residuals = y_fit - y_pred_fit
    rng = np.random.default_rng()
    for b in range(n_boot):
        v = rng.choice([-1, 1], size=len(y_fit))
        y_star = y_pred_fit + residuals * v
        model_b = LinearRegression().fit(X_fit, y_star)
        res_b = y_star - model_b.predict(X_fit)
        idx = rng.integers(0, len(res_b), size=x_grid.size)
        eps_new = rng.choice([-1, 1], size=x_grid.size) * np.abs(res_b[idx])
        y_boot[b] = model_b.predict(X_grid) + eps_new
    band_low, band_high = np.percentile(y_boot, [2.5, 97.5], axis=0)

    # --- Homoskedastic (classic) prediction band ---
    n = len(x_fit)
    dof = n - 2
    s = np.sqrt(np.sum((y_fit - y_pred_fit) ** 2) / dof)
    x_bar = x_fit.mean()
    Sxx = np.sum((x_fit - x_bar) ** 2)
    se_pred = s * np.sqrt(1 + 1/n + (x_grid - x_bar) ** 2 / Sxx)
    t_crit = t.ppf(0.975, dof)
    low = y_grid - t_crit * se_pred
    high = y_grid + t_crit * se_pred

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_ax = True
    # Get a light version of the color for the data
    base_color = mcolors.to_rgb(color)
    light_color = tuple(0.6 + 0.4 * c for c in base_color)  # blend with white
    # Plot all data
    ax.plot(x, y, marker='+', linestyle='-', color=light_color, label=label if label else None)
    # Plot fit and bands only over fit range
    ax.plot(x_grid, y_grid, color=color)
    ax.fill_between(x_grid, low, high, color='gold', alpha=0.25)
    ax.fill_between(x_grid, band_low, band_high, color='lightcoral', alpha=0.25)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Delta Mass (g)')
    if show_title:
        ax.set_title('Drift: Delta Mass vs Time')
        eqn = f"y = {gradient:.8f}x + {intercept:.8f}\nR²={r2:.3f}, RMSE={rmse:.4f}"
        ax.text(0.05, 0.95, eqn, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    ax.grid(True)
    if label or not show_title:
        ax.legend()
    if created_ax:
        plt.tight_layout()
        plt.show()


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print(f"Usage: python {sys.argv[0]} <csv_file> [x_min] [x_max]")
        sys.exit(1)
    csv_file = sys.argv[1]
    x_min = float(sys.argv[2]) if len(sys.argv) > 2 else None
    x_max = float(sys.argv[3]) if len(sys.argv) > 3 else None
    plot_drift(csv_file, x_min, x_max)


if __name__ == "__main__":
    main() 
