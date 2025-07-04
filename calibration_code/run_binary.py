"""
python -m calibration_code.run_binary <letter> <rate> [--mode {steps,flow}]

This script runs a binary search for the optimal out pump rate for a given in pump rate.
Usage: 
  python -m calibration_code.run_binary A 148802 --mode steps  # Steps mode (default)
  python -m calibration_code.run_binary A 20.0 --mode flow     # Flow mode (µL/s)
"""

import sys
import os
import time
import argparse
import glob
from datetime import datetime
from sklearn.linear_model import LinearRegression
from src.config import Config as cfg
from calibration_code.calibration_utils import run_drift, log_to_csv
import pandas as pd

BINARY_SEARCH_DIR = 'binary_search_results'
BINARY_DRIFT_DIR = 'binary_drift_results'


def format_flow_rate_for_filename(flow_rate):
    """
    Format flow rate for filename: 20 or 20.0 -> 20_0, 14.5 -> 14_5
    """
    # Convert to string with one decimal place
    formatted = f"{flow_rate:.1f}"
    # Replace decimal point with underscore
    return formatted.replace('.', '_')


def get_most_recent_calibration(letter, pump_type):
    """
    Find the most recent calibration file for a given letter and pump type (in/out).
    Returns the file path, gradient (ml_per_step), and intercept from the calibration.
    """
    calib_dir = 'calibration_results'
    
    # Find all calibration files for this pump
    pattern = f"{calib_dir}/*calibration_{letter}_{pump_type}_forward_results.csv"
    matches = glob.glob(pattern)
    
    if not matches:
        raise FileNotFoundError(f"No calibration files found for {letter}_{pump_type}")
    
    # Get the most recent file
    most_recent_file = max(matches, key=os.path.getmtime)
    
    # Read the calibration data and fit to get ml_per_step
    df = pd.read_csv(most_recent_file)
    X = df['steps_rate'].values / 1000  # Convert to thousands of steps/sec
    y = df['ml_rate'].values * 1000     # Convert to microlitres/sec
    
    # Fit linear model
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    gradient = model.coef_[0]  # This is ml_per_step in µL/s per 1000 steps/s
    intercept = model.intercept_  # This is the intercept in µL/s
    
    return most_recent_file, gradient, intercept


def calculate_step_rates_from_flow(letter, target_flow_rate):
    """
    Calculate step rates for in and out pumps based on calibration data.
    target_flow_rate is in µL/s.
    Returns (in_steps_rate, out_steps_rate)
    """
    print(f"Calculating step rates for pump {letter} at flow rate {target_flow_rate} µL/s...")
    
    # Get most recent calibrations
    in_calib_file, in_gradient, in_intercept = get_most_recent_calibration(letter, 'in')
    out_calib_file, out_gradient, out_intercept = get_most_recent_calibration(letter, 'out')
    
    print(f"  In pump calibration: {os.path.basename(in_calib_file)}")
    print(f"  Out pump calibration: {os.path.basename(out_calib_file)}")
    print(f"  In pump gradient: {in_gradient:.6f} µL/s per 1000 steps/s")
    print(f"  Out pump gradient: {out_gradient:.6f} µL/s per 1000 steps/s")
    print(f"  In pump intercept: {in_intercept:.6f} µL/s")
    print(f"  Out pump intercept: {out_intercept:.6f} µL/s")
    
    # Calculate step rates accounting for intercept
    # Using the formula: steps_rate = (target_flow_rate - intercept) / gradient
    in_steps_rate = ((target_flow_rate - in_intercept) / in_gradient) * 1000  # Convert back to steps/s
    out_steps_rate = ((target_flow_rate - out_intercept) / out_gradient) * 1000
    
    print(f"  Calculated in pump rate: {in_steps_rate:.1f} steps/s")
    print(f"  Calculated out pump rate: {out_steps_rate:.1f} steps/s")
    
    return in_steps_rate, out_steps_rate


def run_binary(letter, rate, mode='steps'):
    """
    Run binary search for optimal out pump rate.
    
    Args:
        letter: Pump letter (A, B, C, D)
        rate: Either steps per second (steps mode) or flow rate in µL/s (flow mode)
        mode: 'steps' or 'flow'
    """
    in_key = f'{letter}_in'
    out_key = f'{letter}_out'
    in_pump = cfg.PUMPS[in_key]['serial']
    out_pump = cfg.PUMPS[out_key]['serial']
    date_str = datetime.now().strftime('%y%m%d')
    os.makedirs(BINARY_SEARCH_DIR, exist_ok=True)
    os.makedirs(BINARY_DRIFT_DIR, exist_ok=True)
    
    # Calculate initial step rates based on mode
    if mode == 'flow':
        in_rate, out_rate = calculate_step_rates_from_flow(letter, rate)
        print(f"Using calculated step rates: in={in_rate:.1f}, out={out_rate:.1f} steps/s")
    else:  # steps mode
        in_rate = rate
        out_rate = rate
        print(f"Using provided step rate: {rate} steps/s for both pumps")
    
    # 1. Initial test: both pumps at calculated rates for 15 min
    print(f"Running initial test: in pump at {in_rate:.1f}, out pump at {out_rate:.1f} for 15 min...")
    search_rows = []
    
    # Run initial test
    meas_times = [0, 180, 900]
    data = run_drift(in_pump, in_rate, out_pump, out_rate, duration=900, measurement_times=meas_times, csv_output_path=None, log_progress=True)
    masses = [row[1] for row in data]
    initial_mass, mass_3min, end_mass = masses[0], masses[1], masses[2]
    delta_mass = end_mass - mass_3min
    search_rows.append([out_rate, in_rate, initial_mass, mass_3min, end_mass, delta_mass])
    print(f"Initial: Δmass (end-3min) = {delta_mass:.4f}g")
    # 2. Adjust out pump rate until sign change
    direction = 1 if delta_mass > 0 else -1 if delta_mass < 0 else 0
    if direction == 0:
        out_rates = [out_rate + 1000, out_rate - 1000]
    else:
        while True:
            out_rate += 1000 * direction
            print(f"Testing out pump at {out_rate:.1f} (in at {in_rate:.1f})...")
            data = run_drift(in_pump, in_rate, out_pump, out_rate, duration=900, measurement_times=meas_times, csv_output_path=None, log_progress=True)
            masses = [row[1] for row in data]
            initial_mass, mass_3min, end_mass = masses[0], masses[1], masses[2]
            delta_mass_new = end_mass - mass_3min
            search_rows.append([out_rate, in_rate, initial_mass, mass_3min, end_mass, delta_mass_new])
            print(f"Δmass (end-3min) = {delta_mass_new:.4f}g")
            if (direction > 0 and delta_mass_new < 0) or (direction < 0 and delta_mass_new > 0):
                # Crossed zero
                out_rates = [out_rate - 1000 * direction, out_rate]
                break
    # 3. Binary search
    print(f"Starting binary search between {out_rates[0]:.1f} and {out_rates[1]:.1f}...")
    tol = 0.005
    low, high = min(out_rates), max(out_rates)
    best_out_rate = None
    best_delta = None
    while low <= high:
        mid = (low + high) // 2
        print(f"Testing out pump at {mid:.1f} (in at {in_rate:.1f}) for 30 min...")
        data = run_drift(in_pump, in_rate, out_pump, mid, duration=1800, measurement_times=[0, 180, 1800], csv_output_path=None, log_progress=True)
        masses = [row[1] for row in data]
        initial_mass, mass_3min, end_mass = masses[0], masses[1], masses[2]
        delta_mass = end_mass - mass_3min
        search_rows.append([mid, in_rate, initial_mass, mass_3min, end_mass, delta_mass])
        print(f"Δmass (end-3min) = {delta_mass:.4f}g")
        if abs(delta_mass) <= tol:
            best_out_rate = mid
            best_delta = delta_mass
            break
        elif delta_mass > 0:
            low = mid + 1
        else:
            high = mid - 1
    if best_out_rate is None:
        # Pick the closest
        diffs = [(abs(row[5]), row[0]) for row in search_rows if row[0] >= min(out_rates) and row[0] <= max(out_rates)]
        diffs.sort()
        best_out_rate = diffs[0][1]
        best_delta = [row[5] for row in search_rows if row[0] == best_out_rate][0]
    
    print(f"Best out pump rate: {best_out_rate:.1f} (Δmass={best_delta:.4f}g)")
    
    # 4. Save binary search results
    search_cols = ['out_pump_steps_rate', 'in_pump_steps_rate', 'initial_mass', 'mass_3min', 'end_mass', 'delta_mass_end_minus_3min']
    
    # Create filename based on mode with proper formatting
    if mode == 'flow':
        flow_rate_formatted = format_flow_rate_for_filename(rate)
        search_csv = os.path.join(BINARY_SEARCH_DIR, f"{date_str}_binary_search_{letter}_flow_{flow_rate_formatted}_results.csv")
    else:
        search_csv = os.path.join(BINARY_SEARCH_DIR, f"{date_str}_binary_search_{letter}_{rate}_results.csv")
    
    log_to_csv(search_rows, search_cols, search_csv)
    print(f"Binary search results saved to {search_csv}")
    # 5. Final drift experiment
    print(f"Running final drift experiment for 30 min: in {in_rate}, out {best_out_rate}")
    drift_csv = os.path.join(BINARY_DRIFT_DIR, f"{date_str}_binary_drift_{letter}_{in_rate}_{best_out_rate}_results.csv")
    run_drift(in_pump, in_rate, out_pump, best_out_rate, duration=1800, measurement_times=None, csv_output_path=drift_csv, log_progress=True)
    print(f"Drift results saved to {drift_csv}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run binary search for optimal out pump rate')
    parser.add_argument('letter', type=str, help='Pump letter (A, B, C, D)')
    parser.add_argument('rate', type=float, help='Rate: steps per second (steps mode) or flow rate in µL/s (flow mode)')
    parser.add_argument('--mode', type=str, choices=['steps', 'flow'], default='steps',
                       help='Mode: steps (default) or flow')
    
    args = parser.parse_args()
    
    letter = args.letter.upper()
    rate = args.rate
    mode = args.mode
    
    # Validate inputs
    if letter not in ['A', 'B', 'C', 'D']:
        print(f"Error: Letter must be A, B, C, or D. Got: {letter}")
        sys.exit(1)
    
    if rate <= 0:
        print(f"Error: Rate must be positive. Got: {rate}")
        sys.exit(1)
    
    if mode == 'steps' and rate != int(rate):
        print(f"Error: Steps rate must be an integer. Got: {rate}")
        sys.exit(1)
    
    print(f"Running binary search experiment: Pump {letter}")
    if mode == 'flow':
        print(f"Mode: Flow rate = {rate} µL/s")
    else:
        print(f"Mode: Steps rate = {int(rate)} steps/s")
    print(f"{'='*50}")
    
    try:
        start_time = time.time()
        run_binary(letter, rate, mode)
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = (elapsed_seconds % 3600) % 60
        print(f"Experiment completed successfully in {hours}h {minutes}m {seconds:.1f}s!")
    except Exception as e:
        print(f"Error in experiment (Pump {letter}, {rate} {mode}): {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
