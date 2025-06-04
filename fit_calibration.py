import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
import os


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <input_csv>")
        sys.exit(1)
    input_csv = sys.argv[1]

    # Load experimental data
    df = pd.read_csv(input_csv)
    X = df[['steps_rate']].values
    y = df['ml_rate'].values

    # Fit linear model
    model = LinearRegression()
    model.fit(X, y)
    gradient = model.coef_[0]
    intercept = model.intercept_

    # Plot data and regression line
    plt.scatter(X, y, color='blue', label='Data')
    x_line = pd.Series(X.flatten()).sort_values()
    y_line = gradient * x_line + intercept
    plt.plot(x_line, y_line, color='red', label='Regression line')
    plt.xlabel('Step Rate')
    plt.ylabel('Actual Rate')
    plt.title('Linear Regression Calibration')
    eqn = f"y = {gradient:.8f}x + {intercept:.8f}"
    plt.legend()
    plt.text(0.05, 0.95, eqn, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
