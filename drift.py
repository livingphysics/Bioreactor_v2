# Set your pump serials and flow rate here
in_pump = '00473498'   # Example: '00473517'
out_pump = '00473497' # Example: '00473508'
flow_rate_ul_s = 20.0    # Example: 15 (microlitres/sec)

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import trange


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


def main():
    # Find CSVs
    calib_dir = 'calibration_data'
    csv_in = os.path.join(calib_dir, f"pump_{in_pump}_forward.csv")
    csv_out = os.path.join(calib_dir, f"pump_{out_pump}_forward.csv")
    if not os.path.exists(csv_in) or not os.path.exists(csv_out):
        print(f"Could not find calibration CSVs: {csv_in}, {csv_out}")
        return

    # Fit models
    grad_in, int_in, X_in, y_in, res_in, model_in = load_and_fit(csv_in)
    grad_out, int_out, X_out, y_out, res_out, model_out = load_and_fit(csv_out)

    # Calculate steps_rate for desired flow (in thousands of steps/sec)
    # y = grad * x + int => x = (y - int) / grad
    x_in_target = (flow_rate_ul_s - int_in) / grad_in
    x_out_target = (flow_rate_ul_s - int_out) / grad_out

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

if __name__ == "__main__":
    main()
