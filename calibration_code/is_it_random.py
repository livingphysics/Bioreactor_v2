#!/usr/bin/env python3
"""
is_it_random.py

This script runs a sequence of experiments to test if pump behavior is random:
1. Debubblify for 20 minutes
2. Run a binary drift (search + drift)
3. Sleep for 5 minutes
4. Debubblify for 20 minutes
5. Run a drift at the in/out rates found by the binary drift

Usage:
    python -m calibration_code.is_it_random <pump_letter> <flow_rate> [--mode {steps,flow}]

Example:
    python -m calibration_code.is_it_random A 20.0 --mode flow
    python -m calibration_code.is_it_random A 148802 --mode steps
"""

import sys
import os
import time
import argparse
import subprocess
from datetime import datetime
from src.config import Config as cfg
from calibration_code.calibration_utils import run_drift, log_to_csv
from calibration_code.run_binary import run_binary, calculate_step_rates_from_flow
import pandas as pd
import re


def run_debubblify(letter, duration_minutes=20):
    """
    Run debubblify for a specified duration.
    
    Args:
        letter: Pump letter (A, B, C, D)
        duration_minutes: Duration to run debubblify in minutes
    """
    print(f"\n{'='*60}")
    print(f"Running debubblify for pump {letter} for {duration_minutes} minutes...")
    print(f"{'='*60}")
    
    # Start debubblify in background
    try:
        # Run debubblify for the specified pump
        PYTHON_CMD = "/home/michele/venv/bin/python3"
        process = subprocess.Popen([PYTHON_CMD, "-m", "hardware_testing.debubblify"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Let it run for the specified duration
        time.sleep(duration_minutes * 60)
        
        # Stop the process
        process.terminate()
        process.wait(timeout=10)
        
        print(f"Debubblify completed for pump {letter}")
        
    except subprocess.TimeoutExpired:
        print("Force killing debubblify process...")
        process.kill()
    except Exception as e:
        print(f"Error running debubblify: {e}")
        if process.poll() is None:
            process.terminate()


def extract_binary_drift_rates(search_csv_path):
    """
    Extract the optimal in/out rates from a binary search CSV file.
    
    Args:
        search_csv_path: Path to the binary search results CSV
        
    Returns:
        tuple: (in_rate, out_rate) in steps/sec
    """
    try:
        df = pd.read_csv(search_csv_path)
        
        # Find the row with the smallest absolute delta_mass
        df['abs_delta'] = abs(df['delta_mass_end_minus_3min'])
        best_row = df.loc[df['abs_delta'].idxmin()]
        
        in_rate = best_row['in_pump_steps_rate']
        out_rate = best_row['out_pump_steps_rate']
        
        print(f"Extracted rates from binary search: in={in_rate:.1f}, out={out_rate:.1f} steps/s")
        return in_rate, out_rate
        
    except Exception as e:
        print(f"Error extracting rates from {search_csv_path}: {e}")
        return None, None


def find_most_recent_binary_search(letter, rate, mode='steps'):
    """
    Find the most recent binary search CSV file for the given parameters.
    
    Args:
        letter: Pump letter
        rate: Flow rate or step rate
        mode: 'steps' or 'flow'
        
    Returns:
        str: Path to the most recent binary search CSV file
    """
    from calibration_code.run_binary import format_flow_rate_for_filename
    
    binary_search_dir = 'binary_search_results'
    
    if mode == 'flow':
        flow_rate_formatted = format_flow_rate_for_filename(rate)
        pattern = f"{binary_search_dir}/*_binary_search_{letter}_flow_{flow_rate_formatted}_results.csv"
    else:
        pattern = f"{binary_search_dir}/*_binary_search_{letter}_{rate}_results.csv"
    
    import glob
    matches = glob.glob(pattern)
    
    if not matches:
        raise FileNotFoundError(f"No binary search files found matching pattern: {pattern}")
    
    # Return the most recent file
    return max(matches, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description='Run a sequence of experiments to test pump randomness')
    parser.add_argument('letter', choices=['A', 'B', 'C', 'D'], help='Pump letter')
    parser.add_argument('rate', type=float, help='Flow rate (µL/s) or step rate (steps/s)')
    parser.add_argument('--mode', choices=['steps', 'flow'], default='flow', 
                       help='Mode: steps (step rate) or flow (flow rate in µL/s)')
    
    args = parser.parse_args()
    
    letter = args.letter.upper()
    rate = args.rate
    mode = args.mode
    
    print(f"Starting is_it_random experiment for pump {letter}")
    print(f"Rate: {rate} ({mode} mode)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get pump serials
    in_key = f'{letter}_in'
    out_key = f'{letter}_out'
    in_pump = cfg.PUMPS[in_key]['serial']
    out_pump = cfg.PUMPS[out_key]['serial']
    
    # Step 1: Debubblify for 20 minutes
    print(f"\nStep 1: Debubblify for 20 minutes")
    run_debubblify(letter, duration_minutes=20)
    
    # Step 2: Run binary drift
    print(f"\nStep 2: Running binary drift")
    print(f"{'='*60}")
    
    # Run the binary search and drift
    run_binary(letter, rate, mode)
    
    # Step 3: Sleep for 5 minutes
    print(f"\nStep 3: Sleeping for 5 minutes")
    print(f"{'='*60}")
    for i in range(5, 0, -1):
        print(f"Sleeping... {i} minutes remaining")
        time.sleep(60)
    print("Sleep completed")
    
    # Step 4: Debubblify for 20 minutes again
    print(f"\nStep 4: Debubblify for 20 minutes (second time)")
    run_debubblify(letter, duration_minutes=20)
    
    # Step 5: Run drift at the rates found by binary drift
    print(f"\nStep 5: Running drift at binary drift rates")
    print(f"{'='*60}")
    
    try:
        # Find the most recent binary search results
        search_csv = find_most_recent_binary_search(letter, rate, mode)
        print(f"Using binary search results from: {search_csv}")
        
        # Extract the optimal rates
        in_rate, out_rate = extract_binary_drift_rates(search_csv)
        
        if in_rate is None or out_rate is None:
            print("Could not extract rates from binary search. Exiting.")
            return
        
        # Run the final drift experiment
        date_str = datetime.now().strftime('%y%m%d')
        drift_csv = f"drift_results/{date_str}_drift_{letter}_{in_rate:.0f}_{out_rate:.0f}_results.csv"
        
        # Ensure drift_results directory exists
        os.makedirs('drift_results', exist_ok=True)
        
        print(f"Running drift experiment:")
        print(f"  In pump: {in_pump} at {in_rate:.1f} steps/s")
        print(f"  Out pump: {out_pump} at {out_rate:.1f} steps/s")
        print(f"  Duration: 30 minutes")
        print(f"  Output: {drift_csv}")
        
        # Run the drift experiment
        run_drift(
            in_pump, in_rate, out_pump, out_rate,
            duration=1800,  # 30 minutes
            measurement_times=None,  # Drift mode: every 15s
            csv_output_path=drift_csv,
            log_progress=True
        )
        
        print(f"\nFinal drift experiment completed!")
        print(f"Results saved to: {drift_csv}")
        
    except Exception as e:
        print(f"Error in final drift step: {e}")
        return
    
    print(f"\n{'='*60}")
    print(f"is_it_random experiment completed!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 
