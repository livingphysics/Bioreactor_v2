import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
from tqdm import trange


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <input_csv>")
        sys.exit(1)
    input_csv = sys.argv[1]

    # Load experimental data
    df = pd.read_csv(input_csv)
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
    print(f"Gradient: {gradient}, Intercept: {intercept}")
    print(f"R^2: {r2:.4f}, RMSE: {rmse:.4f} (microlitres/sec)")

    # --- Wild bootstrap prediction band ---
    # 1. Set up a grid of x-values to draw a smooth ribbon on
    x_grid = np.linspace(X_thousands.min(), X_thousands.max(), 200)
    X_grid = x_grid.reshape(-1, 1)

    # 2. Wild-bootstrap resampling
    n_boot = 2000
    y_boot = np.empty((n_boot, x_grid.size))
    residuals = y_ul_s - y_pred
    X_col = X_thousands.reshape(-1, 1)
    rng = np.random.default_rng()
    for b in trange(n_boot, desc='Wild bootstrap'):
        v = rng.choice([-1, 1], size=len(y_ul_s))
        y_star = y_pred + residuals * v
        model_b = LinearRegression().fit(X_col, y_star)
        y_boot[b] = model_b.predict(X_grid)

    # 3. 95% percentile envelope (prediction band)
    band_low, band_high = np.percentile(y_boot, [2.5, 97.5], axis=0)

    # 4. Plot
    plt.figure(figsize=(6,4))
    plt.scatter(X_thousands, y_ul_s, color='royalblue', label='data')
    plt.plot(x_grid, intercept + gradient * x_grid, color='crimson', label='least-squares line')
    plt.fill_between(x_grid, band_low, band_high, color='lightcoral', alpha=0.25, label='95 % prediction band (wild bootstrap)')
    plt.xlabel('Step rate (thousands s⁻¹)')
    plt.ylabel('Actual rate (µL s⁻¹)')
    plt.title('Heteroskedastic-robust prediction band via wild bootstrap')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
