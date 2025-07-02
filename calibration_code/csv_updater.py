#!/usr/bin/env python3
"""
csv_updater.py

Update binary_results_compiled.csv with actual drift rates from final calibration experiments.
Finds the most recent drift file for each pump/flow rate combination, fits a linear model,
and adds the gradient as 'actual_drift_rate' to the compiled results.
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from calibration_utils import find_most_recent_file

def format_flow_rate_for_filename(flow_rate):
    """
    Format flow rate for filename: 20 or 20.0 -> 20_0, 14.5 -> 14_5
    """
    # Convert to string with one decimal place
    formatted = f"{flow_rate:.1f}"
    # Replace decimal point with underscore
    return formatted.replace('.', '_')

def fit_drift_rate(filepath):
    """Fit a linear model to drift data and return the gradient (drift rate)."""
    try:
        df = pd.read_csv(filepath)
        
        if len(df) < 2:
            print(f"Warning: Insufficient data points in {filepath}")
            return None
        
        # Extract time and delta mass
        x = df['time_s'].values
        y = df['delta_mass_g'].values
        
        # Fit linear model
        model = LinearRegression()
        X = x.reshape(-1, 1)
        model.fit(X, y)
        
        gradient = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(X, y)
        
        print(f"  Gradient (drift rate): {gradient:.10f}")
        print(f"  Intercept: {intercept:.6f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Data points: {len(x)}")
        
        return gradient
        
    except Exception as e:
        print(f"Error fitting drift rate for {filepath}: {e}")
        return None

def main():
    letters = ['A', 'B', 'C', 'D']
    flow_rates = [4, 8, 12, 16, 20]
    drift_results_dir = "final_calibration_drift_results"
    
    print(f"Updating binary_results_compiled.csv with actual drift rates...")
    print(f"Looking for drift files in: {drift_results_dir}")
    
    # Check if drift results directory exists
    if not os.path.exists(drift_results_dir):
        print(f"Error: Directory {drift_results_dir} not found")
        return
    
    # Read the compiled results
    try:
        df = pd.read_csv('binary_results_compiled.csv')
        print(f"Loaded binary_results_compiled.csv with {len(df)} rows")
    except FileNotFoundError:
        print("Error: binary_results_compiled.csv not found")
        return
    
    # Add actual_drift_rate column if it doesn't exist
    if 'actual_drift_rate' not in df.columns:
        df['actual_drift_rate'] = np.nan
        print("Added 'actual_drift_rate' column")
    
    # Process each pump/flow rate combination
    updated_count = 0
    missing_count = 0
    
    for letter in letters:
        for flow_rate in flow_rates:
            print(f"\n{'='*50}")
            print(f"Processing pump {letter}, flow rate {flow_rate}")
            print(f"{'='*50}")
            
            # Find the most recent drift file
            flow_rate_formatted = format_flow_rate_for_filename(flow_rate)
            pattern_template = "{date}_drift_results_{letter}_{flow_rate}.csv"
            pattern_dict = {'letter': letter, 'flow_rate': flow_rate_formatted}
            
            filepath, date_str = find_most_recent_file(drift_results_dir, pattern_template, pattern_dict)
            
            if filepath is None:
                print(f"  No drift file found for pump {letter}, flow rate {flow_rate}")
                missing_count += 1
                continue
            
            print(f"  Found file: {os.path.basename(filepath)} (date: {date_str})")
            
            # Fit drift rate
            drift_rate = fit_drift_rate(filepath)
            
            if drift_rate is not None:
                # Update the dataframe
                mask = (df['pump'] == letter) & (df['flow_rate'] == flow_rate)
                if mask.any():
                    df.loc[mask, 'actual_drift_rate'] = drift_rate
                    updated_count += 1
                    print(f"  ✓ Updated drift rate: {drift_rate:.10f}")
                else:
                    print(f"  Warning: No matching row found in compiled results")
            else:
                print(f"  ✗ Failed to fit drift rate")
                missing_count += 1
    
    # Save updated results
    df.to_csv('binary_results_compiled.csv', index=False)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"UPDATE SUMMARY")
    print(f"{'='*60}")
    print(f"Total combinations: {len(letters) * len(flow_rates)}")
    print(f"Successfully updated: {updated_count}")
    print(f"Missing/failed: {missing_count}")
    
    if updated_count > 0:
        print(f"\nUpdated binary_results_compiled.csv with actual drift rates")
        print(f"New column 'actual_drift_rate' contains the fitted gradients")
        
        # Show summary of updated values
        print(f"\nUpdated drift rates:")
        for _, row in df[df['actual_drift_rate'].notna()].iterrows():
            print(f"  Pump {row['pump']}, flow {row['flow_rate']}: {row['actual_drift_rate']:.10f}")
    
    if missing_count > 0:
        print(f"\nWarning: {missing_count} combinations could not be updated")
        print("Consider running drift experiments for missing combinations")

if __name__ == "__main__":
    main() 
