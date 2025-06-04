import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
import os
import numpy as np


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

    # Standard error of predictions (for error bars)
    residuals = y_ul_s - y_pred
    dof = len(y_ul_s) - 2
    residual_std = np.sqrt(np.sum(residuals ** 2) / dof)
    # For each x, the standard error of the fit is:
    x_mean = np.mean(X_thousands)
    Sxx = np.sum((X_thousands - x_mean) ** 2)
    y_err = residual_std * np.sqrt(1/len(X_thousands) + (X_thousands - x_mean) ** 2 / Sxx)

    # Plot data and regression line with error bars
    plt.errorbar(X_thousands, y_ul_s, yerr=y_err, fmt='o', color='blue', label='Data ± fit error')
    x_line = np.linspace(X_thousands.min(), X_thousands.max(), 100)
    y_line = gradient * x_line + intercept
    plt.plot(x_line, y_line, color='red', label='Regression line')
    plt.xlabel('Step Rate (thousands/sec)')
    plt.ylabel('Actual Rate (μL/sec)')
    plt.title('Linear Regression Calibration')
    eqn = f"y = {gradient:.8f}x + {intercept:.8f}"
    plt.legend()
    plt.text(0.05, 0.95, eqn + f"\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.2f} μL/s", transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
