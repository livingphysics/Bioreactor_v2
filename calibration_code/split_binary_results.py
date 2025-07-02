import pandas as pd
import os

def split_binary_results():
    """Split the compiled binary results into separate CSV files for each pump."""
    input_path = "calibration_code/binary_results_compiled.csv"
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found!")
        return
    
    # Read the compiled results
    print(f"Reading compiled results from {input_path}...")
    df = pd.read_csv(input_path)
    
    # Split by pump letter
    for pump_letter in ['A', 'B', 'C', 'D']:
        # Filter data for this pump
        pump_data = df[df['pump'] == pump_letter].copy()
        
        if len(pump_data) == 0:
            print(f"No data found for pump {pump_letter}")
            continue
        
        # Reorder columns to put flow_rate first
        columns_order = ['flow_rate', 'in_steps_rate', 'out_steps_rate', 'estimated_drift_rate']
        pump_data = pump_data[columns_order]
        
        # Sort by flow_rate
        pump_data = pump_data.sort_values('flow_rate')
        
        # Create output filename
        output_filename = f"binary_results_pump_{pump_letter}.csv"
        output_path = os.path.join("calibration_code", output_filename)
        
        # Save to CSV
        pump_data.to_csv(output_path, index=False)
        
        print(f"Created {output_filename} with {len(pump_data)} rows")
        print(f"  Flow rates: {list(pump_data['flow_rate'])}")
    
    print("\nAll pump-specific CSV files created in calibration_code directory!")

if __name__ == "__main__":
    split_binary_results() 
