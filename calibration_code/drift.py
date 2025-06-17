"""
Example usage:
    python -m calibration_code.drift <letter: A/B/C/D> <flow_rate_ul_s>

Example:
    python -m calibration_code.drift A 10.0

This will run the dual pump experiment for vial A at 10.0 uL/s.
"""
import sys
from src.config import Config as cfg
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import trange
import matplotlib.pyplot as plt
import re
from math import floor
import time
from glob import glob
from calibration_code.calibration_utils import run_drift


def load_and_fit(csv_path):
    df = pd.read_csv(csv_path, comment='#')
    X = df[['steps_rate']].values
    y = df['ml_rate'].values
    X_thousands = X.flatten() / 1000  # thousands of steps/sec
    y_ul_s = y * 1000  # microlitres/sec
    model = LinearRegression()
    model.fit(X_thousands.reshape(-1, 1), y_ul_s)
    gradient = model.coef_[0]
    intercept = model.intercept_
    y_pred = model.predict(X_thousands.reshape(-1, 1))
    residuals = y_ul_s - y_pred
    return gradient, intercept, X_thousands, y_ul_s, residuals, model


def wild_bootstrap_at_x(X_thousands, y_ul_s, residuals, model, x_target, n_boot=2000):
    X_col = X_thousands.reshape(-1, 1)
    y_pred = model.predict(X_col)
    rng = np.random.default_rng()
    y_boot = np.empty(n_boot)
    for b in trange(n_boot, desc="Wild bootstrap – prediction", leave=False):
        v = rng.choice([-1, 1], size=len(y_ul_s))
        y_star = y_pred + residuals * v
        model_b = LinearRegression().fit(X_col, y_star)
        res_b = y_star - model_b.predict(X_col)
        idx = rng.integers(0, len(res_b))
        eps_new = rng.choice([-1, 1]) * np.abs(res_b[idx])
        y_boot[b] = model_b.predict(np.array([[x_target]]))[0] + eps_new
    return y_boot


# --- Scale reading utilities (from calibrate_pump.py) ---
SERIAL_PORT_SCALE = "/dev/ttyUSB0"  # Update if needed
BAUDRATE = 9600
TOLERANCE = 0.001
READ_TIMEOUT = 1


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <letter: A/B/C/D> <flow_rate_ul_s>")
        sys.exit(1)
    letter = sys.argv[1].upper()
    if letter not in ['A', 'B', 'C', 'D']:
        print("Error: Letter must be one of A, B, C, D.")
        sys.exit(1)
    try:
        flow_rate_ul_s = float(sys.argv[2])
    except ValueError:
        print("Error: flow_rate_ul_s must be a number.")
        sys.exit(1)
    # Check for at most 1 decimal place
    flow_str = sys.argv[2]
    if '.' in flow_str:
        decimal_part = flow_str.split('.')[-1]
        if len(decimal_part) > 1:
            print("Error: Please input flow_rate_ul_s with at most 1 decimal place.")
            sys.exit(1)
    in_key = f'{letter}_in'
    out_key = f'{letter}_out'
    in_pump = cfg.PUMPS[in_key]['serial']
    out_pump = cfg.PUMPS[out_key]['serial']

    # Find most recent calibration CSVs in calibration_results
    def find_latest_calib_csv(letter, direction):
        pattern = f"calibration_results/*_calibration_{letter}_{direction}_forward_results.csv"
        files = glob(pattern)
        if not files:
            return None
        # Extract date from filename and sort
        def extract_date(f):
            m = re.search(r'(\\d{8})_calibration', f)
            return m.group(1) if m else ''
        files.sort(key=extract_date, reverse=True)
        return files[0]

    csv_in = find_latest_calib_csv(letter, 'in')
    csv_out = find_latest_calib_csv(letter, 'out')
    if not csv_in or not csv_out:
        print(f"Could not find calibration CSVs: {csv_in}, {csv_out}")
        return

    # Fit models
    grad_in, int_in, X_in, y_in, res_in, model_in = load_and_fit(csv_in)
    grad_out, int_out, X_out, y_out, res_out, model_out = load_and_fit(csv_out)

    # Calculate steps_rate for desired flow (in thousands of steps/sec)
    # y = grad * x + int => x = (y - int) / grad
    x_in_target = (flow_rate_ul_s - int_in) / grad_in
    x_out_target = (flow_rate_ul_s - int_out) / grad_out
    print(f"x_in_target: {x_in_target}, x_out_target: {x_out_target}")

    # Wild bootstrap distributions at x_target
    y_in_boot = wild_bootstrap_at_x(X_in, y_in, res_in, model_in, x_in_target)
    y_out_boot = wild_bootstrap_at_x(X_out, y_out, res_out, model_out, x_out_target)

    # Distribution of the difference
    diff_boot = y_in_boot - y_out_boot
    ci_low, ci_high = np.percentile(diff_boot, [2.5, 97.5])
    mean_diff = np.mean(diff_boot)
    print(f"\n95% confidence interval for the difference in flow rates (in_pump - out_pump) at {flow_rate_ul_s} uL/s:")
    print(f"  in_pump: {in_pump}")
    print(f"  out_pump: {out_pump}")
    print(f"  Mean difference: {mean_diff:.3f} uL/s")
    print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}] uL/s")

    # --- Run dual experiment for both pumps ---
    times_masses = run_drift(
        in_pump, x_in_target*1000, out_pump, x_out_target*1000,
        duration=1800, measurement_times=None, # Drift mode: every 15s
        csv_output_path=None, log_progress=True
    )

if __name__ == "__main__":
    main()
