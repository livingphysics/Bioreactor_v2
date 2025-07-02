"""
calibration_utils.py

Shared utilities for pump control, mass measurement, experiment timing, CSV logging, and plotting for drift and binary drift experiments.
"""
import os
import re
import time
import serial
import numpy as np
import pandas as pd
from math import floor
from datetime import datetime
from ticlib import TicUSB
from sklearn.linear_model import LinearRegression
from scipy.stats import t
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- Pump control utilities ---
STEP_MODE = 3
STEPS_PER_PULSE = 0.5 ** STEP_MODE

def init_pump(pump_serial):
    """Initialize and configure a pump given its serial number."""
    pump = TicUSB(serial_number=pump_serial)
    pump.energize()
    pump.exit_safe_start()
    pump.set_step_mode(STEP_MODE)
    pump.set_current_limit(32)
    return pump

def stop_pump(pump):
    """Stop and de-energize a pump."""
    pump.set_target_velocity(0)
    pump.deenergize()
    pump.enter_safe_start()
    del pump

# --- Mass measurement utilities ---
SERIAL_PORT_SCALE = "/dev/ttyUSB0"  # Update if needed
BAUDRATE = 9600
TOLERANCE = 0.001
READ_TIMEOUT = 1

def parse_weight(s: str) -> float:
    """Parse a float weight from the scale's ASCII output string."""
    m = re.search(r'[-+]?\d*\.\d+|\d+', s)
    return float(m.group()) if m else None

def read_stable_weight():
    """Read a stable weight from the scale, waiting for two consecutive readings within tolerance."""
    prev = None
    max_retries = 10
    retry_count = 0
    
    while retry_count < max_retries:
        ser = None
        try:
            ser = serial.Serial(port=SERIAL_PORT_SCALE,
                              baudrate=BAUDRATE,
                              bytesize=serial.EIGHTBITS,
                              parity=serial.PARITY_NONE,
                              stopbits=serial.STOPBITS_ONE,
                              timeout=READ_TIMEOUT)
            ser.write(b'w')
            raw = ser.read(18)
            text = raw.decode('ascii', errors='ignore')
            w = parse_weight(text)
            if w is not None:
                if prev is not None and abs(w - prev) < TOLERANCE:
                    return w
                prev = w
        except (serial.SerialException, OSError) as e:
            print(f"Serial connection error: {e}")
            retry_count += 1
            time.sleep(1.0)  # Wait before retrying
        finally:
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
        time.sleep(3.0)
    
    raise Exception(f"Failed to read stable weight after {max_retries} retries")

# --- Experiment running utilities ---
def run_drift(in_pump_serial, in_steps_rate, out_pump_serial, out_steps_rate, duration=1800, measurement_times=None, csv_output_path=None, log_progress=True):
    """
    Run a dual-pump experiment. If measurement_times is None, measure every 15s (drift mode).
    If measurement_times is a list, measure at those times (in seconds, e.g., [0, 180, 900] for binary drift).
    Returns: list of (time, delta_mass)
    """
    if measurement_times is None:
        # Drift mode: measure every 15s
        measurement_times = list(range(0, duration+1, 15))
    else:
        # Ensure 0 and duration are included
        if 0 not in measurement_times:
            measurement_times = [0] + measurement_times
        if duration not in measurement_times:
            measurement_times = measurement_times + [duration]
        measurement_times = sorted(set(measurement_times))
    if log_progress:
        print(f"\nRunning experiment for {duration//60} minutes...")
    in_velocity = int(floor(in_steps_rate / STEPS_PER_PULSE))
    out_velocity = int(floor(out_steps_rate / STEPS_PER_PULSE))
    data_rows = []
    initial_mass = read_stable_weight()
    t_exp = 0.0
    next_measure_idx = 0
    
    # Initialize pump variables for cleanup
    in_pump = None
    out_pump = None
    
    try:
        while t_exp < duration or next_measure_idx < len(measurement_times):
            # If it's time for a measurement
            if next_measure_idx < len(measurement_times) and t_exp >= measurement_times[next_measure_idx]:
                mass = read_stable_weight()
                delta_mass = mass - initial_mass
                data_rows.append((t_exp, delta_mass))
                if log_progress:
                    print(f"  t={t_exp:.1f}s, delta_mass={delta_mass:.4f}g")
                next_measure_idx += 1
            # If experiment is done
            if t_exp >= duration:
                break
            # Run pumps until next measurement or end
            next_time = measurement_times[next_measure_idx] if next_measure_idx < len(measurement_times) else duration
            run_time = min(next_time - t_exp, duration - t_exp)
            if run_time > 0:
                # Initialize pumps with timeout handling
                try:
                    in_pump = init_pump(in_pump_serial)
                    out_pump = init_pump(out_pump_serial)
                except Exception as e:
                    print(f"Failed to initialize pumps: {e}")
                    raise
                
                t_measurement_start = time.time()
                while time.time() - t_measurement_start <= run_time:
                    try:
                        in_pump.set_target_velocity(in_velocity)
                        out_pump.set_target_velocity(out_velocity)
                    except Exception as e:
                        print(f"Pump control error: {e}")
                        raise
                t_exp += time.time() - t_measurement_start
                
                # Clean up pumps after each run
                if in_pump:
                    stop_pump(in_pump)
                    in_pump = None
                if out_pump:
                    stop_pump(out_pump)
                    out_pump = None
    except Exception as e:
        print(f"Experiment error: {e}")
        raise
    finally:
        # Ensure pumps are cleaned up even if an exception occurs
        if in_pump:
            try:
                stop_pump(in_pump)
            except Exception as e:
                print(f"Error cleaning up in_pump: {e}")
        if out_pump:
            try:
                stop_pump(out_pump)
            except Exception as e:
                print(f"Error cleaning up out_pump: {e}")
    
    # Write to CSV if path provided
    if csv_output_path is not None:
        df = pd.DataFrame(data_rows, columns=["time_s", "delta_mass_g"])
        df.to_csv(csv_output_path, index=False)
        if log_progress:
            print(f"Experiment data saved to {csv_output_path}")
    return data_rows

# --- CSV/Batch utilities ---
def log_to_csv(data, columns, path):
    """Write data (list of tuples) to CSV with given columns and path."""
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(path, index=False)

def find_most_recent_file(folder, pattern_template, pattern_dict):
    """
    Find the most recent file in folder matching the pattern_template, filled with pattern_dict.
    pattern_template: e.g. '{date}_drift_{letter}_{flow}_0_results.csv'
    pattern_dict: e.g. {'letter': 'A', 'flow': 4}
    Returns (filepath, date_str) or (None, None)
    """
    # Build regex pattern from template
    regex = pattern_template
    # Replace {date} with a regex group
    regex = regex.replace('{date}', r'(\\d{6})')
    # Replace other fields with their values or regex
    for key, value in pattern_dict.items():
        regex = regex.replace('{' + key + '}', str(value))
    # Escape dots
    regex = regex.replace('.', r'\\.')
    pattern = re.compile(f'^{regex}$')
    most_recent_date = None
    most_recent_file = None
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            date = m.group(1)
            if (most_recent_date is None) or (date > most_recent_date):
                most_recent_date = date
                most_recent_file = os.path.join(folder, fname)
    return most_recent_file, most_recent_date

# --- Plotting/Stats utilities ---
def get_fit_stats(x, y):
    """Fit a linear model and return gradient, intercept, r2, rmse, model."""
    model = LinearRegression()
    X = np.array(x).reshape(-1, 1)
    model.fit(X, y)
    y_pred = model.predict(X)
    gradient = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    return gradient, intercept, r2, rmse, model

def calculate_homoskedastic_band(x, y, model, x_grid):
    """Calculate classic (homoskedastic) prediction band."""
    X = np.array(x).reshape(-1, 1)
    y_pred = model.predict(X)
    n = len(x)
    dof = n - 2
    s = np.sqrt(np.sum((y - y_pred) ** 2) / dof)
    x_bar = np.mean(x)
    Sxx = np.sum((x - x_bar) ** 2)
    se_pred = s * np.sqrt(1 + 1/n + (x_grid - x_bar) ** 2 / Sxx)
    t_crit = t.ppf(0.975, dof)
    y_grid = model.predict(x_grid.reshape(-1, 1))
    low = y_grid - t_crit * se_pred
    high = y_grid + t_crit * se_pred
    return low, high

def calculate_heteroskedastic_band(x, y, model, x_grid, n_boot=2000):
    """Calculate wild bootstrap (heteroskedastic) prediction band."""
    X = np.array(x).reshape(-1, 1)
    y_pred = model.predict(X)
    residuals = y - y_pred
    rng = np.random.default_rng()
    y_boot = np.empty((n_boot, len(x_grid)))
    for b in range(n_boot):
        v = rng.choice([-1, 1], size=len(y))
        y_star = y_pred + residuals * v
        model_b = LinearRegression().fit(X, y_star)
        res_b = y_star - model_b.predict(X)
        idx = rng.integers(0, len(res_b), size=len(x_grid))
        eps_new = rng.choice([-1, 1], size=len(x_grid)) * np.abs(res_b[idx])
        y_boot[b] = model_b.predict(x_grid.reshape(-1, 1)) + eps_new
    band_low, band_high = np.percentile(y_boot, [2.5, 97.5], axis=0)
    return band_low, band_high

def plot_with_fit_and_bands(x, y, ax=None, label=None, color='b', show_title=True, show_stats_table=True):
    """Plot data, fit line, and both error bands. Flexible for drift and binary drift."""
    gradient, intercept, r2, rmse, model = get_fit_stats(x, y)
    x_grid = np.linspace(np.min(x), np.max(x), 200)
    y_grid = model.predict(x_grid.reshape(-1, 1))
    band_low, band_high = calculate_heteroskedastic_band(np.array(x), np.array(y), model, x_grid)
    low, high = calculate_homoskedastic_band(np.array(x), np.array(y), model, x_grid)
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))  # Increased height for table
        created_ax = True
    base_color = mcolors.to_rgb(color)
    light_color = tuple(0.6 + 0.4 * c for c in base_color)
    ax.plot(x, y, marker='+', linestyle='-', color=light_color, label=label if label else None)
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
    
    # Add statistics table if requested
    if show_stats_table and created_ax:
        # Create table data
        table_data = [
            ['Gradient', f'{gradient:.15f}'],
            ['Intercept', f'{intercept:.8f}'],
            ['R²', f'{r2:.4f}'],
            ['RMSE', f'{rmse:.6f}'],
            ['Data Points', f'{len(x)}']
        ]
        
        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=['Statistic', 'Value'],
                        cellLoc='left',
                        loc='bottom',
                        bbox=[0, -0.3, 1, 0.2])  # Position below plot
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style data rows
        for i in range(1, len(table_data) + 1):
            for j in range(2):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f2f2f2')
                else:
                    table[(i, j)].set_facecolor('white')
        
        # Adjust layout to accommodate table
        plt.subplots_adjust(bottom=0.25)
    
    if created_ax:
        plt.tight_layout()
        plt.show() 
