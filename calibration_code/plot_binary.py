import sys
import pandas as pd
from calibration_code.calibration_utils import plot_with_fit_and_bands

def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <csv_file>")
        sys.exit(1)
    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file, comment='#')
    if not {'time_s', 'delta_mass_g'}.issubset(df.columns):
        print("CSV must contain 'time_s' and 'delta_mass_g' columns.")
        sys.exit(1)
    x = df['time_s'].values
    y = df['delta_mass_g'].values
    plot_with_fit_and_bands(x, y, show_title=True)

if __name__ == "__main__":
    main() 
