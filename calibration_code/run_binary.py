"""
python -m calibration_code.run_binary <letter> <steps_rate>

This script runs a binary search for the optimal out pump rate for a given in pump rate.
Usage: python -m calibration_code.run_binary A 148802
"""

import sys
import os
import time
import argparse
from datetime import datetime
from src.config import Config as cfg
from calibration_code.calibration_utils import run_drift, log_to_csv
import pandas as pd

BINARY_SEARCH_DIR = 'binary_search_results'
BINARY_DRIFT_DIR = 'binary_drift_results'


def run_binary(letter, steps_rate):
    in_key = f'{letter}_in'
    out_key = f'{letter}_out'
    in_pump = cfg.PUMPS[in_key]['serial']
    out_pump = cfg.PUMPS[out_key]['serial']
    date_str = datetime.now().strftime('%y%m%d')
    os.makedirs(BINARY_SEARCH_DIR, exist_ok=True)
    os.makedirs(BINARY_DRIFT_DIR, exist_ok=True)
    # 1. Initial test: both pumps at steps_rate for 15 min
    print(f"Running initial test: in/out pumps at {steps_rate} for 15 min...")
    search_rows = []
    out_rate = steps_rate
    in_rate = steps_rate
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
            print(f"Testing out pump at {out_rate} (in at {in_rate})...")
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
    print(f"Starting binary search between {out_rates[0]} and {out_rates[1]}...")
    tol = 0.005
    low, high = min(out_rates), max(out_rates)
    best_out_rate = None
    best_delta = None
    while low <= high:
        mid = (low + high) // 2
        print(f"Testing out pump at {mid} (in at {in_rate}) for 30 min...")
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
    print(f"Best out pump rate: {best_out_rate} (Δmass={best_delta:.4f}g)")
    # 4. Save binary search results
    search_cols = ['out_pump_steps_rate', 'in_pump_steps_rate', 'initial_mass', 'mass_3min', 'end_mass', 'delta_mass_end_minus_3min']
    search_csv = os.path.join(BINARY_SEARCH_DIR, f"{date_str}_binary_search_{letter}_{steps_rate}_results.csv")
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
    parser.add_argument('steps_rate', type=int, help='Steps per second rate')
    
    args = parser.parse_args()
    
    letter = args.letter.upper()
    steps_rate = args.steps_rate
    
    # Validate inputs
    if letter not in ['A', 'B', 'C', 'D']:
        print(f"Error: Letter must be A, B, C, or D. Got: {letter}")
        sys.exit(1)
    
    if steps_rate <= 0:
        print(f"Error: Steps rate must be positive. Got: {steps_rate}")
        sys.exit(1)
    
    print(f"Running binary search experiment: Pump {letter} at {steps_rate} steps/s")
    print(f"{'='*50}")
    
    try:
        start_time = time.time()
        run_binary(letter, steps_rate)
        end_time = time.time()
        elapsed_seconds = end_time - start_time
        hours = int(elapsed_seconds // 3600)
        minutes = int((elapsed_seconds % 3600) // 60)
        seconds = (elapsed_seconds % 3600) % 60
        print(f"Experiment completed successfully in {hours}h {minutes}m {seconds:.1f}s!")
    except Exception as e:
        print(f"Error in experiment (Pump {letter}, {steps_rate} steps/s): {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
