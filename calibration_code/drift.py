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
import serial
import re
from math import floor
import time
from ticlib import TicUSB
from glob import glob


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


def parse_weight(s: str) -> float:
    m = re.search(r'[-+]?\d*\.\d+|\d+', s)
    return float(m.group()) if m else None

def read_stable_weight():
    ser = serial.Serial(port=SERIAL_PORT_SCALE,
                        baudrate=BAUDRATE,
                        bytesize=serial.EIGHTBITS,
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        timeout=READ_TIMEOUT)
    prev = None
    while True:
        ser.write(b'w')
        raw = ser.read(18)
        try:
            text = raw.decode('ascii', errors='ignore')
        except:
            continue
        w = parse_weight(text)
        if w is None:
            continue
        if prev is not None and abs(w - prev) < TOLERANCE:
            ser.close()
            del ser
            return w
        prev = w
        time.sleep(3.0)

# --- Pump driving utilities (from debubblify.py) ---
STEP_MODE = 3
STEPS_PER_PULSE = 0.5 ** STEP_MODE

def init_pump(pump_serial):
    pump = TicUSB(serial_number=pump_serial)
    pump.energize()
    pump.exit_safe_start()
    pump.set_step_mode(STEP_MODE)
    pump.set_current_limit(32)
    return pump

def stop_pump(pump):
    pump.set_target_velocity(0)
    pump.deenergize()
    pump.enter_safe_start()
    del pump

def run_dual_experiment(in_pump_serial, in_steps_rate, out_pump_serial, out_steps_rate, duration=3600, measurement_interval=15, csv_output_path=None, flow_rate_ul_s=None, letter=None):
    if csv_output_path is None:
        from datetime import datetime
        date_str = datetime.now().strftime('%y%m%d')
        flow_str = f"{flow_rate_ul_s:.1f}".replace('.', '_')
        csv_output_path = f"drift_results/{date_str}_drift_{letter}_{flow_str}_results.csv"
    print(f"\nRunning dual experiment for {duration//60} minutes...")
    # Setup both pumps
    in_velocity = int(floor(in_steps_rate / STEPS_PER_PULSE))
    out_velocity = int(floor(out_steps_rate / STEPS_PER_PULSE))

    # Data storage
    times = [0.0]
    masses = [0.0,]
    data_rows = [(0.0, 0.0)]  # For CSV: (time, delta_mass)
    initial_mass = read_stable_weight()
    t_exp = 0.0
    measurement_number = 0
    while t_exp < duration:
        in_pump = init_pump(in_pump_serial)
        out_pump = init_pump(out_pump_serial)

        # Start both pumps
        t_measurement_start = time.time()
        while time.time() - t_measurement_start <= measurement_interval:
            in_pump.set_target_velocity(in_velocity)
            out_pump.set_target_velocity(out_velocity)
        t_exp += time.time() - t_measurement_start
        
        # Stop both pumps
        stop_pump(in_pump)
        stop_pump(out_pump)
        
        # Pause for measurement
        mass = read_stable_weight()
        times.append(t_exp)
        delta_mass = mass - initial_mass
        masses.append(delta_mass)
        data_rows.append((t_exp, delta_mass))
        print(f"  t={t_exp:.1f}s, delta_mass={delta_mass:.4f}g")
    
    # Write to CSV
    df = pd.DataFrame(data_rows, columns=["time_s", "delta_mass_g"])
    df.to_csv(csv_output_path, index=False)
    print(f"Experiment data saved to {csv_output_path}")
    
    return times, masses


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
    times, masses = run_dual_experiment(in_pump, x_in_target*1000, out_pump, x_out_target*1000, duration=3600, measurement_interval=15, flow_rate_ul_s=flow_rate_ul_s, letter=letter)
    
    # # Plot
    # plt.figure(figsize=(10,6))
    # plt.plot(times, masses, '+-', label=f'Δmass')
    # # Overlay CI lines: y = grad*x and y = grad*sqrt(x/60) for grad_in and grad_out
    # x_grid = np.linspace(0, max(times), 100)
    # for grad, color in zip([grad_in, grad_out], ['red', 'blue']):
        # plt.plot(x_grid, grad * x_grid / 1e6, '--', color=color, label=f'y={grad:.3f}·x (uL/s)')
        # plt.plot(x_grid, grad * np.sqrt(x_grid/60) / 1e6, ':', color=color, label=f'y={grad:.3f}·sqrt(x/60) (uL/s)')
    # plt.xlabel('Experiment time (s)')
    # plt.ylabel('Δmass (g)')
    # plt.title(f'Dual Pump Δmass vs. time at {x_in_target:.1f} & {x_out_target:.1f} steps/sec')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    main()
