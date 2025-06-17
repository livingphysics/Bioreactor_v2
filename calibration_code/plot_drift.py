"""
Example usage:
    python -m calibration_code.plot_drift <csv_file> [x_min] [x_max]

Plots drift data from a CSV file, with optional time range selection.
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from calibration_code.calibration_utils import plot_with_fit_and_bands, get_fit_stats


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

    gradient, intercept, r2, rmse, model = get_fit_stats(x_fit, y_fit)
    # For homoskedastic band width, use calibration_utils
    x_grid = np.linspace(x_fit.min(), x_fit.max(), 200)
    from calibration_code.calibration_utils import calculate_homoskedastic_band
    low, high = calculate_homoskedastic_band(x_fit, y_fit, model, x_grid)
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

    plot_with_fit_and_bands(x_fit, y_fit, ax=ax, label=label, color=color, show_title=show_title)


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
