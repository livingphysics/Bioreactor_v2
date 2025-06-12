import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import t


def plot_drift(csv_file, x_min=None, x_max=None):
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

    # Plot all data
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='+', linestyle='-', color='b', label='Data')
    # Plot fit and bands only over fit range
    plt.plot(x_grid, y_grid, color='crimson', label='Best fit (selected range)')
    plt.fill_between(x_grid, low, high, color='gold', alpha=0.25, label='95% pred. band (classic)')
    plt.fill_between(x_grid, band_low, band_high, color='lightcoral', alpha=0.25, label='95% pred. band (wild bootstrap)')
    plt.xlabel('Time (s)')
    plt.ylabel('Delta Mass (g)')
    plt.title('Drift: Delta Mass vs Time')
    eqn = f"y = {gradient:.8f}x + {intercept:.8f}\nR²={r2:.3f}, RMSE={rmse:.4f}"
    plt.text(0.05, 0.95, eqn, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    plt.grid(True)
    plt.legend()
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
