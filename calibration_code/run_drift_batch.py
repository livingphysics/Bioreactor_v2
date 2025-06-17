"""
Example usage:
    /home/michele/venv/bin/python3 -m calibration_code.run_drift_batch

This script will run calibration_code.drift for all combinations of letters (A, B, C, D) and specified flow rates.
Logs are saved to run_drift_batch.log.
"""
import subprocess
import logging
import time

LETTERS = ['A', 'B', 'C', 'D']
FLOW_RATES = [4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]  # Flow rates from 4 to 20 uL/s in steps of 2
PYTHON_CMD = "/home/michele/venv/bin/python3"
MAX_RETRIES = 3

# Set up logging
logging.basicConfig(
    filename='run_drift_batch.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)

for flow in FLOW_RATES:
    for letter in LETTERS:
        attempt = 0
        success = False
        while attempt < MAX_RETRIES and not success:
            attempt += 1
            logging.info(f"Starting run: Letter={letter}, Flow={flow} uL/s, Attempt={attempt}")
            print(f"Running drift for {letter} at {flow} uL/s (Attempt {attempt})")
            try:
                subprocess.run(
                    [PYTHON_CMD, "-m", "calibration_code.drift", letter, f"{flow:.1f}"],
                    check=True
                )
                logging.info(f"Success: Letter={letter}, Flow={flow} uL/s on attempt {attempt}")
                success = True
            except subprocess.CalledProcessError as e:
                logging.error(f"Failure: Letter={letter}, Flow={flow} uL/s, Attempt={attempt}, Error={e}")
                print(f"Run failed for {letter} at {flow} uL/s (Attempt {attempt}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(1)  # Wait before retrying
        if not success:
            logging.error(f"Max retries reached: Letter={letter}, Flow={flow} uL/s. Skipping.")
            print(f"Max retries reached for {letter} at {flow} uL/s. Skipping.")
