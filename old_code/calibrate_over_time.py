import sys
import pandas as pd
import numpy as np

# Usage: python calibrate_over_time.py <csv_file> [duration_column] [mass_column]

def main():
    if len(sys.argv) < 2:
        print("Usage: python calibrate_over_time.py <csv_file> [duration_column] [mass_column]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    duration_col = sys.argv[2] if len(sys.argv) > 2 else 'duration'

    df = pd.read_csv(csv_file)
    if duration_col not in df.columns or 'ml_rate' not in df.columns:
        print(f"CSV must contain columns '{duration_col}' and 'ml_rate'")
        sys.exit(1)

    t = df[duration_col].to_numpy(dtype=float)
    ml_rate = df['ml_rate'].to_numpy(dtype=float)
    m = t * ml_rate
    n = len(t)

    mu = np.sum(m) / np.sum(t)
    sigma_squared = np.sum(((m - mu * t) ** 2) / t) / n

    print(f"mu = {mu}")
    print(f"sigma_squared = {sigma_squared}")

if __name__ == "__main__":
    main()
