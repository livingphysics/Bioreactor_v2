import sys
import subprocess
import logging
import time

# Define the list of experiments as tuples (letter, steps_rate)
EXPERIMENTS = [
    ('A', 148802),
    ('B', 64263),
    ('C', 174417),
    ('D', 43647)
]

MAX_RETRIES = 3

# Set up logging
logging.basicConfig(
    filename='run_binary_batch.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)

def main():
    print(f"Running {len(EXPERIMENTS)} binary search experiments...")
    
    for i, (letter, steps_rate) in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*50}")
        print(f"Experiment {i}/{len(EXPERIMENTS)}: Pump {letter} at {steps_rate} steps/s")
        print(f"{'='*50}")
        
        attempt = 0
        success = False
        while attempt < MAX_RETRIES and not success:
            attempt += 1
            logging.info(f"Starting run: Letter={letter}, Steps={steps_rate}, Attempt={attempt}")
            print(f"Running binary calibration for {letter} at {steps_rate} (Attempt {attempt})")
            
            result = subprocess.run([
                sys.executable, 
                '-m', 
                'calibration_code.run_binary', 
                letter, 
                str(steps_rate)
            ])
            
            if result.returncode == 0:
                logging.info(f"Success: Letter={letter}, Steps={steps_rate} on attempt {attempt}")
                print(f"Completed {letter} at {steps_rate}.")
                success = True
            else:
                logging.error(f"Failure: Letter={letter}, Steps={steps_rate}, Attempt={attempt}, ReturnCode={result.returncode}")
                print(f"Run failed for {letter} at {steps_rate} (Attempt {attempt}).")
                if attempt < MAX_RETRIES:
                    print("Waiting 1 second before retry...")
                    time.sleep(1)
        
        if not success:
            logging.error(f"Max retries reached: Letter={letter}, Steps={steps_rate}. Skipping.")
            print(f"Max retries reached for {letter} at {steps_rate}. Skipping.")
    
    print(f"\n{'='*50}")
    print(f"All experiments completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 
