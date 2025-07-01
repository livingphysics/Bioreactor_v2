import pandas as pd
import numpy as np
import os
import sys
import subprocess
import time
from sklearn.linear_model import LinearRegression
from datetime import datetime
from calibration_code.calibration_utils import run_drift

def run_drift_experiment(letter, in_steps_rate, out_steps_rate, duration=1800):
    """Run a drift experiment and return the drift rate from linear fit."""
    print(f"Running drift experiment for pump {letter}...")
    print(f"  In steps rate: {in_steps_rate}")
    print(f"  Out steps rate: {out_steps_rate}")
    
    # Get pump serial numbers from config
    from src.config import Config as cfg
    in_key = f'{letter}_in'
    out_key = f'{letter}_out'
    in_pump_serial = cfg.PUMPS[in_key]['serial']
    out_pump_serial = cfg.PUMPS[out_key]['serial']
    
    # Create output filename
    timestamp = datetime.now().strftime("%y%m%d")
    output_filename = f"{timestamp}_drift_{letter}_check_results.csv"
    output_path = os.path.join("drift_results", output_filename)
    
    try:
        # Run the drift experiment
        times_masses = run_drift(
            in_pump_serial, in_steps_rate, out_pump_serial, out_steps_rate,
            duration=duration, measurement_times=None,  # Every 15s
            csv_output_path=output_path, log_progress=True
        )
        
        # Fit linear model to the data
        times = np.array([t for t, _ in times_masses])
        masses = np.array([m for _, m in times_masses])
        
        # Reshape for sklearn
        X = times.reshape(-1, 1)
        y = masses
        
        # Fit linear model with intercept
        model = LinearRegression()
        model.fit(X, y)
        
        # Get the gradient (drift rate)
        drift_rate = model.coef_[0]
        
        print(f"  Drift experiment completed. Drift rate: {drift_rate:.9f} g/s")
        return drift_rate
        
    except Exception as e:
        print(f"  Error running drift experiment: {e}")
        return None

def process_binary_results():
    """Read compiled binary results and add drift analysis."""
    input_path = "calibration_code/binary_results_compiled.csv"
    output_path = "calibration_code/binary_results_compiled_with_drift.csv"
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found!")
        print("Please run binary_compiler.py first to generate the compiled results.")
        return
    
    # Read the compiled results
    print(f"Reading compiled results from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Add drift_rate column
    df['drift_rate'] = np.nan
    
    # Process each row
    for index, row in df.iterrows():
        print(f"\n{'='*50}")
        print(f"Processing {index + 1}/{len(df)}: Pump {row['pump']} at {row['flow_rate']} µL/s")
        print(f"{'='*50}")
        
        # Run drift experiment
        drift_rate = run_drift_experiment(
            row['pump'], 
            row['in_steps_rate'], 
            row['out_steps_rate']
        )
        
        if drift_rate is not None:
            df.at[index, 'drift_rate'] = round(drift_rate, 9)
            print(f"  Updated drift_rate: {drift_rate:.9f}")
        else:
            print(f"  Failed to get drift rate for {row['pump']} at {row['flow_rate']} µL/s")
        
        # Small delay between experiments
        time.sleep(2)
    
    # Save the results
    df.to_csv(output_path, index=False)
    print(f"\n{'='*50}")
    print(f"Results saved to {output_path}")
    print(f"{'='*50}")
    
    # Print summary
    print("\nSummary:")
    for _, row in df.iterrows():
        if pd.notna(row['drift_rate']):
            print(f"  {row['pump']}{row['flow_rate']}: drift_rate = {row['drift_rate']:.9f}")
        else:
            print(f"  {row['pump']}{row['flow_rate']}: drift_rate = FAILED")

if __name__ == "__main__":
    process_binary_results()
