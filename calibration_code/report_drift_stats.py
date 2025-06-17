"""
Example usage:
    python -m calibration_code.report_drift_stats
    python -m calibration_code.report_drift_stats --band

Reports a table of RMSEs (or half the average width of the homoskedastic error band, if --band is given)
for the linear model fits of the most recent drift result CSVs for each pump pair (A-D) at each flow rate (4-20).
"""

import os
import re
import sys
import importlib.util
import argparse
import numpy as np

# --- Helper to import plot_drift.py as a module ---
PLOT_DRIFT_PATH = os.path.join(os.path.dirname(__file__), 'plot_drift.py')
spec = importlib.util.spec_from_file_location('plot_drift', PLOT_DRIFT_PATH)
plot_drift_mod = importlib.util.module_from_spec(spec)
sys.modules['plot_drift'] = plot_drift_mod
spec.loader.exec_module(plot_drift_mod)

LETTERS = ['A', 'B', 'C', 'D']
FLOW_RATES = [4, 6, 8, 10, 12, 14, 16, 18, 20]

def find_most_recent_file(folder, letter, flow):
    pattern = re.compile(r"(\d{{6}})_drift_{}_{}_0_results\.csv".format(letter, flow))
    most_recent_date = None
    most_recent_file = None
    for fname in os.listdir(folder):
        m = pattern.match(fname)
        if m:
            date = m.group(1)
            if (most_recent_date is None) or (date > most_recent_date):
                most_recent_date = date
                most_recent_file = os.path.join(folder, fname)
    return most_recent_file, most_recent_date

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--band', action='store_true', help='Report half the average width of the homoskedastic error band instead of RMSE')
    args = parser.parse_args()

    folder = os.path.join(os.path.dirname(__file__), '..', 'drift_results')
    folder = os.path.abspath(folder)

    # Table: rows = flow rates, columns = letters
    table = []
    for flow in FLOW_RATES:
        row = []
        for letter in LETTERS:
            f, _ = find_most_recent_file(folder, letter, flow)
            if f is not None:
                try:
                    rmse, band = plot_drift_mod.get_drift_stats(f)
                    value = band if args.band else rmse
                    row.append(f"{value:.4g}")
                except Exception as e:
                    row.append("ERR")
            else:
                row.append("")
        table.append(row)

    # Print table
    col_width = 12
    header = "Flow".ljust(6) + "".join(l.center(col_width) for l in LETTERS)
    print(header)
    print("-" * len(header))
    for i, flow in enumerate(FLOW_RATES):
        row_str = f"{flow:<6}" + "".join(cell.center(col_width) for cell in table[i])
        print(row_str)

if __name__ == "__main__":
    main() 
