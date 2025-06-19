"""
Example usage:
    python -m calibration_code.plot_corrected_drift <letter> <flow_rate>

This script performs drift correction by:
1. Finding the most recent calibrations for in and out pumps
2. Calculating step rates based on the target flow rate (in µL/s)
3. Finding the most recent drift result and fitting a linear model
4. Calculating the drift rate and correcting the out pump rate
5. Running a new drift experiment with the corrected rates
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime
import glob
from calibration_code.calibration_utils import (
    find_most_recent_file, get_fit_stats, run_drift, plot_with_fit_and_bands
)
from src.config import Config as cfg


def get_most_recent_calibration(letter, pump_type):
    """
    Find the most recent calibration file for a given letter and pump type (in/out).
    Returns the file path and the gradient (ml_per_step) from the calibration.
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
    
    return most_recent_file, gradient


def get_most_recent_drift_result(letter, flow_rate):
    """
    Find the most recent drift result file for a given letter and flow rate.
    Returns the file path and the drift rate (gradient of linear fit).
    """
    drift_dir = 'drift_results'
    
    # Find all drift files for this letter and flow rate
    pattern = f"{drift_dir}/*drift_{letter}_{flow_rate}_0_results.csv"
    matches = glob.glob(pattern)
    
    if not matches:
        raise FileNotFoundError(f"No drift result files found for {letter} at flow rate {flow_rate}")
    
    # Get the most recent file
    most_recent_file = max(matches, key=os.path.getmtime)
    
    # Read the drift data and fit to get drift rate
    df = pd.read_csv(most_recent_file)
    x = df['time_s'].values
    y = df['delta_mass_g'].values
    
    # Fit linear model to get drift rate
    gradient, intercept, r2, rmse, model = get_fit_stats(x, y)
    
    return most_recent_file, gradient


def format_flow_rate_for_filename(flow_rate):
    """
    Format flow rate for filename: 20 or 20.0 -> 20_0, 14.5 -> 14_5
    """
    # Convert to string with one decimal place
    formatted = f"{flow_rate:.1f}"
    # Replace decimal point with underscore
    return formatted.replace('.', '_')


def calculate_corrected_rates(letter, target_flow_rate):
    """
    Calculate corrected pump rates based on recent calibrations and drift analysis.
    target_flow_rate is in µL/s.
    Returns (in_steps_rate, corrected_out_steps_rate, drift_rate, ml_per_step_out)
    """
    print(f"Calculating corrected rates for pump {letter} at flow rate {target_flow_rate} µL/s...")
    
    # Get most recent calibrations
    in_calib_file, in_gradient = get_most_recent_calibration(letter, 'in')
    out_calib_file, out_gradient = get_most_recent_calibration(letter, 'out')
    
    print(f"  In pump calibration: {os.path.basename(in_calib_file)}")
    print(f"  Out pump calibration: {os.path.basename(out_calib_file)}")
    print(f"  In pump gradient: {in_gradient:.6f} µL/s per 1000 steps/s")
    print(f"  Out pump gradient: {out_gradient:.6f} µL/s per 1000 steps/s")
    
    # Calculate naive step rates (target_flow_rate is already in µL/s)
    in_steps_rate = (target_flow_rate / in_gradient) * 1000  # Convert back to steps/s
    out_steps_rate = (target_flow_rate / out_gradient) * 1000
    
    print(f"  Naive in pump rate: {in_steps_rate:.1f} steps/s")
    print(f"  Naive out pump rate: {out_steps_rate:.1f} steps/s")
    
    # Get most recent drift result and calculate drift rate
    try:
        drift_file, drift_rate = get_most_recent_drift_result(letter, target_flow_rate)
        print(f"  Drift result: {os.path.basename(drift_file)}")
        print(f"  Drift rate: {drift_rate:.6f} g/s")
        
        # Calculate correction
        # drift_rate is in g/s, out_gradient is in µL/s per 1000 steps/s
        # We need to convert drift_rate to µL/s (assuming 1 g ≈ 1 mL for water)
        drift_rate_ul_s = drift_rate * 1000  # Convert g/s to µL/s
        
        # Calculate the correction in steps/s
        correction_steps_s = (drift_rate_ul_s / out_gradient) * 1000
        
        # Apply correction to out pump rate
        corrected_out_steps_rate = out_steps_rate + correction_steps_s
        
        print(f"  Drift correction: {correction_steps_s:.1f} steps/s")
        print(f"  Corrected out pump rate: {corrected_out_steps_rate:.1f} steps/s")
        
        return in_steps_rate, corrected_out_steps_rate, drift_rate, out_gradient
        
    except FileNotFoundError:
        print(f"  Warning: No drift result found for {letter} at flow rate {target_flow_rate}")
        print(f"  Using naive rates without correction")
        return in_steps_rate, out_steps_rate, 0.0, out_gradient


def run_corrected_drift(letter, target_flow_rate, duration=1800):
    """
    Run a drift experiment with corrected pump rates.
    target_flow_rate is in µL/s.
    """
    # Get pump serial numbers
    in_pump_serial = cfg.PUMPS[f'{letter}_in']['serial']
    out_pump_serial = cfg.PUMPS[f'{letter}_out']['serial']
    
    # Calculate corrected rates
    in_steps_rate, out_steps_rate, drift_rate, ml_per_step_out = calculate_corrected_rates(letter, target_flow_rate)
    
    # Create drift_corrected_results directory if it doesn't exist
    output_dir = 'drift_corrected_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename with new naming convention
    timestamp = datetime.now().strftime("%y%m%d")
    flow_rate_formatted = format_flow_rate_for_filename(target_flow_rate)
    output_file = f"{output_dir}/{timestamp}_drift_corrected_{letter}_{flow_rate_formatted}_results.csv"
    
    print(f"\nRunning corrected drift experiment...")
    print(f"  Duration: {duration} seconds ({duration//60} minutes)")
    print(f"  Output file: {output_file}")
    
    # Run the drift experiment
    data_rows = run_drift(
        in_pump_serial=in_pump_serial,
        in_steps_rate=in_steps_rate,
        out_pump_serial=out_pump_serial,
        out_steps_rate=out_steps_rate,
        duration=duration,
        csv_output_path=output_file,
        log_progress=True
    )
    
    return output_file, data_rows


def plot_corrected_drift(csv_file, ax=None, label=None, show_title=True):
    """
    Plot the corrected drift data with fit and bands.
    """
    df = pd.read_csv(csv_file, comment='#')
    if not {'time_s', 'delta_mass_g'}.issubset(df.columns):
        raise ValueError("CSV must contain 'time_s' and 'delta_mass_g' columns.")
    
    x = df['time_s'].values
    y = df['delta_mass_g'].values
    
    # Plot with fit and bands
    plot_with_fit_and_bands(x, y, ax=ax, label=label, show_title=show_title)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <letter> <flow_rate>")
        print("Example: python plot_corrected_drift.py A 20")
        sys.exit(1)
    
    letter = sys.argv[1].upper()
    flow_rate = float(sys.argv[2])
    
    # Validate inputs
    if letter not in ['A', 'B', 'C', 'D']:
        print("Error: Letter must be A, B, C, or D")
        sys.exit(1)
    
    if flow_rate <= 0:
        print("Error: Flow rate must be positive")
        sys.exit(1)
    
    try:
        # Run the corrected drift experiment
        output_file, data_rows = run_corrected_drift(letter, flow_rate)
        
        # Plot the results
        print(f"\nPlotting results...")
        plot_corrected_drift(output_file, label=f"Corrected drift {letter} {flow_rate} µL/s")
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
