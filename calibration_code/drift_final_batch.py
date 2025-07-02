#!/usr/bin/env python3
"""
drift_final_batch.py

Run drift experiments for all pumps and flow rates with retry logic.
Calls drift_final_single.py for each combination.
"""

import subprocess
import sys
import time
from datetime import datetime

def run_single_drift(letter, flow_rate, max_retries=3):
    """Run a single drift experiment with retry logic."""
    # Use the exact same PYTHON_CMD as in run_drift_batch.py
    PYTHON_CMD = "/home/michele/venv/bin/python3"
    cmd = [PYTHON_CMD, "-m", "calibration_code.drift_final_single", letter, str(flow_rate)]
    
    for attempt in range(max_retries):
        print(f"\n{'='*60}")
        print(f"Running drift for pump {letter}, flow rate {flow_rate} (attempt {attempt + 1}/{max_retries})")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(cmd, capture_output=False, text=True, check=True)
            print(f"\n✓ Successfully completed drift for pump {letter}, flow rate {flow_rate}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n✗ Attempt {attempt + 1} failed for pump {letter}, flow rate {flow_rate}")
            print(f"Return code: {e.returncode}")
            
            if attempt < max_retries - 1:
                print(f"Waiting 30 seconds before retry...")
                time.sleep(30)
            else:
                print(f"✗ All {max_retries} attempts failed for pump {letter}, flow rate {flow_rate}")
                return False
    
    return False

def main():
    letters = ['A', 'B', 'C', 'D']
    flow_rates = [4, 8, 12, 16, 20]
    
    print(f"Starting batch drift experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pumps: {letters}")
    print(f"Flow rates: {flow_rates}")
    print(f"Total experiments: {len(letters) * len(flow_rates)}")
    
    # Track results
    successful = []
    failed = []
    
    # Run experiments
    for letter in letters:
        for flow_rate in flow_rates:
            success = run_single_drift(letter, flow_rate)
            
            if success:
                successful.append((letter, flow_rate))
            else:
                failed.append((letter, flow_rate))
            
            # Brief pause between experiments
            print(f"\nPausing 10 seconds before next experiment...")
            time.sleep(10)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETION SUMMARY")
    print(f"{'='*60}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successful: {len(successful)}/{len(letters) * len(flow_rates)}")
    print(f"Failed: {len(failed)}/{len(letters) * len(flow_rates)}")
    
    if successful:
        print(f"\nSuccessful experiments:")
        for letter, flow_rate in successful:
            print(f"  ✓ Pump {letter}, flow rate {flow_rate}")
    
    if failed:
        print(f"\nFailed experiments:")
        for letter, flow_rate in failed:
            print(f"  ✗ Pump {letter}, flow rate {flow_rate}")
    
    if failed:
        print(f"\nSome experiments failed. Consider re-running failed combinations manually.")
        sys.exit(1)
    else:
        print(f"\nAll experiments completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main() 
