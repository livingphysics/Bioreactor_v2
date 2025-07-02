#!/usr/bin/env python3
"""
drift_final_single.py

Run a single drift experiment for final calibration.
Usage: python drift_final_single.py <letter> <flow_rate>
Example: python drift_final_single.py A 4
"""

import sys
import os
import pandas as pd
from datetime import datetime
from calibration_utils import run_drift

def format_flow_rate_for_filename(flow_rate):
    """
    Format flow rate for filename: 20 or 20.0 -> 20_0, 14.5 -> 14_5
    """
    # Convert to string with one decimal place
    formatted = f"{flow_rate:.1f}"
    # Replace decimal point with underscore
    return formatted.replace('.', '_')

def main():
    if len(sys.argv) != 3:
        print("Usage: python drift_final_single.py <letter> <flow_rate>")
        print("Example: python drift_final_single.py A 4")
        sys.exit(1)
    
    letter = sys.argv[1].upper()
    flow_rate = float(sys.argv[2])
    
    # Validate inputs
    if letter not in ['A', 'B', 'C', 'D']:
        print(f"Error: Letter must be A, B, C, or D. Got: {letter}")
        sys.exit(1)
    
    if flow_rate not in [4, 8, 12, 16, 20]:
        print(f"Error: Flow rate must be 4, 8, 12, 16, or 20. Got: {flow_rate}")
        sys.exit(1)
    
    # Read the compiled results
    try:
        df = pd.read_csv('binary_results_compiled.csv')
    except FileNotFoundError:
        print("Error: binary_results_compiled.csv not found in current directory")
        sys.exit(1)
    
    # Find the relevant row
    row = df[(df['pump'] == letter) & (df['flow_rate'] == flow_rate)]
    
    if row.empty:
        print(f"Error: No data found for pump {letter} with flow rate {flow_rate}")
        sys.exit(1)
    
    # Extract parameters
    in_steps_rate = row['in_steps_rate'].iloc[0]
    out_steps_rate = row['out_steps_rate'].iloc[0]
    
    print(f"Found parameters for pump {letter}, flow rate {flow_rate}:")
    print(f"  In steps rate: {in_steps_rate}")
    print(f"  Out steps rate: {out_steps_rate}")
    
    # Create output directory
    output_dir = "final_calibration_drift_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    date_str = datetime.now().strftime("%y%m%d")
    flow_rate_formatted = format_flow_rate_for_filename(flow_rate)
    output_filename = f"{date_str}_drift_results_{letter}_{flow_rate_formatted}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"Output will be saved to: {output_path}")
    
    # Run the drift experiment
    print(f"\nStarting drift experiment for pump {letter} at flow rate {flow_rate}...")
    try:
        data_rows = run_drift(
            in_pump_serial=f"003{letter}",
            in_steps_rate=in_steps_rate,
            out_pump_serial=f"003{letter}",
            out_steps_rate=out_steps_rate,
            duration=1800,  # 30 minutes
            csv_output_path=output_path,
            log_progress=True
        )
        print(f"\nDrift experiment completed successfully!")
        print(f"Data saved to: {output_path}")
        print(f"Number of data points: {len(data_rows)}")
        
    except Exception as e:
        print(f"Error during drift experiment: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
