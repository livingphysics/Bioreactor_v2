import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import t
import statsmodels.api as sm

# Load data
old = pd.read_csv('old.csv')
new = pd.read_csv('calibration_data/pump_00473498_forward.csv')

# Prepare data
X_old = old['steps_rate'].values / 1000  # thousands of steps/sec
Y_old = old['ml_rate'].values * 1000     # microlitres/sec
X_new = new['steps_rate'].values / 1000
Y_new = new['ml_rate'].values * 1000

# Combined data
X_comb = np.concatenate([X_old, X_new])
Y_comb = np.concatenate([Y_old, Y_new])

# Fit individual models
model_old = LinearRegression().fit(X_old.reshape(-1,1), Y_old)
model_new = LinearRegression().fit(X_new.reshape(-1,1), Y_new)
model_comb = LinearRegression().fit(X_comb.reshape(-1,1), Y_comb)

# Compute metrics and print
for label, X, Y, model in [
    ("old.csv", X_old, Y_old, model_old),
    ("pump_00473498_forward.csv", X_new, Y_new, model_new),
    ("combined", X_comb, Y_comb, model_comb)
]:
    y_pred = model.predict(X.reshape(-1,1))
    gradient = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X.reshape(-1,1), Y)
    rmse = np.sqrt(np.mean((Y - y_pred) ** 2))

    # Classic OLS standard errors
    n = len(X)
    x_mean = X.mean()
    Sxx = np.sum((X - x_mean) ** 2)
    residuals = Y - y_pred
    s2 = np.sum(residuals ** 2) / (n - 2)
    se_gradient = np.sqrt(s2 / Sxx)
    se_intercept = np.sqrt(s2 * (1/n + x_mean**2 / Sxx))

    # HC3 robust standard errors using statsmodels
    X_sm = sm.add_constant(X.reshape(-1,1))
    model_sm = sm.OLS(Y, X_sm).fit(cov_type='HC3')
    se_intercept_hc3 = model_sm.bse[0]
    se_gradient_hc3 = model_sm.bse[1]

    print(f"Fit for {label}:")
    print(f"  Equation: y = {gradient:.8f}x + {intercept:.8f}")
    print(f"    Classic uncertainty: ±{se_gradient:.8f} (gradient), ±{se_intercept:.8f} (intercept)")
    print(f"    HC3 robust uncertainty: ±{se_gradient_hc3:.8f} (gradient), ±{se_intercept_hc3:.8f} (intercept)")
    print(f"  R^2: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f} (microlitres/sec)")
    print()

# Prediction lines
x_grid = np.linspace(min(X_comb.min(), X_old.min(), X_new.min()), max(X_comb.max(), X_old.max(), X_new.max()), 200)
y_pred_old = model_old.predict(x_grid.reshape(-1,1))
y_pred_new = model_new.predict(x_grid.reshape(-1,1))
y_pred_comb = model_comb.predict(x_grid.reshape(-1,1))

# Wild bootstrap for combined fit
n_boot = 2000
residuals = Y_comb - model_comb.predict(X_comb.reshape(-1,1))
y_pred_comb_train = model_comb.predict(X_comb.reshape(-1,1))
y_boot = np.empty((n_boot, x_grid.size))
rng = np.random.default_rng()
for b in range(n_boot):
    v = rng.choice([-1, 1], size=len(Y_comb))
    y_star = y_pred_comb_train + residuals * v
    model_b = LinearRegression().fit(X_comb.reshape(-1,1), y_star)
    res_b = y_star - model_b.predict(X_comb.reshape(-1,1))
    idx = rng.integers(0, len(res_b), size=x_grid.size)
    eps_new = rng.choice([-1, 1], size=x_grid.size) * np.abs(res_b[idx])
    y_boot[b] = model_b.predict(x_grid.reshape(-1,1)) + eps_new
band_low, band_high = np.percentile(y_boot, [2.5, 97.5], axis=0)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(X_old, Y_old, color='royalblue', marker='+', label='old.csv')
plt.scatter(X_new, Y_new, color='darkorange', marker='x', label='pump_00473498_forward.csv')
plt.plot(x_grid, y_pred_old, color='blue', linestyle='--', label='Best fit (old)')
plt.plot(x_grid, y_pred_new, color='orange', linestyle='--', label='Best fit (pump_00473498_forward)')
plt.plot(x_grid, y_pred_comb, color='crimson', label='Best fit (combined)')
plt.fill_between(x_grid, band_low, band_high, color='lightcoral', alpha=0.3, label='95% wild bootstrap band (combined)')
plt.xlabel('Step rate (thousands s⁻¹)')
plt.ylabel('Actual rate (µL s⁻¹)')
plt.legend()
plt.title('Pump Calibration: Data, Fits, and Wild Bootstrap Band')
plt.tight_layout()
plt.show()
