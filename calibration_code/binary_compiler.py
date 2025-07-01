import pandas as pd
import os
import glob
import re
from datetime import datetime

# Define the list of experiments as tuples (letter, flow_rate) from run_binary_batch.py
FLOW_RATES = [4, 8, 12, 16, 20]
LETTERS = ['A', 'B', 'C', 'D']

EXPERIMENTS = []
for flow_rate in FLOW_RATES:
    for letter in LETTERS:
        EXPERIMENTS.append((letter, flow_rate))

def find_binary_search_file(letter, flow_rate):
    """Find the most recent binary search file for a given letter and flow rate."""
    pattern = f"*_binary_search_{letter}_flow_{flow_rate}_0_results.csv"
    files = glob.glob(os.path.join("binary_search_results", pattern))
    
    if not files:
        return None
    
    # Sort by date (extract date from filename) and return the most recent
    def extract_date(filename):
        match = re.search(r'(\d{6})_binary_search', filename)
        return match.group(1) if match else '000000'
    
    files.sort(key=extract_date, reverse=True)
    return files[0]

def compile_binary_results():
    """Compile binary search results into a single CSV file."""
    results = []
    
    for letter, flow_rate in EXPERIMENTS:
        print(f"Processing {letter} at {flow_rate} µL/s...")
        
        # Find the binary search file
        file_path = find_binary_search_file(letter, flow_rate)
        
        if file_path is None:
            print(f"  Warning: No file found for {letter} at {flow_rate} µL/s")
            continue
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Get the last row
            last_row = df.iloc[-1]
            
            # Calculate estimated_drift_rate
            delta_mass = last_row['delta_mass_end_minus_3min']
            estimated_drift_rate = delta_mass / (1800 - 180)  # 1620 seconds
            
            # Create result row
            result_row = {
                'pump': letter,
                'flow_rate': flow_rate,
                'in_steps_rate': last_row['in_pump_steps_rate'],
                'out_steps_rate': last_row['out_pump_steps_rate'],
                'estimated_drift_rate': round(estimated_drift_rate, 9)
            }
            
            results.append(result_row)
            print(f"  Success: Found file {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
    
    # Create DataFrame and save to CSV
    if results:
        results_df = pd.DataFrame(results)
        output_path = "calibration_code/binary_results_compiled.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nCompiled {len(results)} results to {output_path}")
        
        # Print summary
        print("\nSummary:")
        for _, row in results_df.iterrows():
            print(f"  {row['pump']}{row['flow_rate']}: drift_rate = {row['estimated_drift_rate']:.9f}")
    else:
        print("No results found!")

if __name__ == "__main__":
    compile_binary_results() 
