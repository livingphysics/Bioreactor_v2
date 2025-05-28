import pandas as pd
from sklearn.linear_model import LinearRegression

# Configuration
INPUT_CSV = "pump_SERIAL3_forward.csv"
OUTPUT_TXT = "calibration_SERIAL3_forward.txt"


def main():
    # Load experimental data
    df = pd.read_csv(INPUT_CSV)
    X = df[['step_rate']].values
    y = df['actual_rate'].values

    # Fit linear model
    model = LinearRegression()
    model.fit(X, y)
    gradient = model.coef_[0]
    intercept = model.intercept_

    # Save calibration
    with open(OUTPUT_TXT, 'w') as f:
        f.write(f"gradient: {gradient}\n")
        f.write(f"intercept: {intercept}\n")

    print(f"Calibration results saved to {OUTPUT_TXT}")

if __name__ == '__main__':
    main()
