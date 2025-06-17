import sys
import subprocess
import logging
import time

LETTERS = ['A', 'B', 'C', 'D']
STEP_RATES = [100, 200, 300, 400, 500]  # Example step rates; adjust as needed
MAX_RETRIES = 3

# Set up logging
logging.basicConfig(
    filename='run_binary_batch.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)

def main():
    for steps_rate in STEP_RATES:
        for letter in LETTERS:
            attempt = 0
            success = False
            while attempt < MAX_RETRIES and not success:
                attempt += 1
                logging.info(f"Starting run: Letter={letter}, Steps={steps_rate}, Attempt={attempt}")
                print(f"Running binary calibration for {letter} at {steps_rate} (Attempt {attempt})")
                result = subprocess.run([sys.executable, '-m', 'calibration_code.run_binary', letter, str(steps_rate)])
                if result.returncode == 0:
                    logging.info(f"Success: Letter={letter}, Steps={steps_rate} on attempt {attempt}")
                    print(f"Completed {letter} at {steps_rate}.")
                    success = True
                else:
                    logging.error(f"Failure: Letter={letter}, Steps={steps_rate}, Attempt={attempt}, ReturnCode={result.returncode}")
                    print(f"Run failed for {letter} at {steps_rate} (Attempt {attempt}).")
                    if attempt < MAX_RETRIES:
                        time.sleep(1)
            if not success:
                logging.error(f"Max retries reached: Letter={letter}, Steps={steps_rate}. Skipping.")
                print(f"Max retries reached for {letter} at {steps_rate}. Skipping.")

if __name__ == "__main__":
    main() 
