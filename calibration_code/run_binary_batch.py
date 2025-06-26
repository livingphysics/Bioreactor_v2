import sys
import subprocess
import logging
import time

# Define the list of experiments as tuples (letter, flow_rate)
# Flow rates: 4, 8, 12, 16, 20 µL/s
# Letters: A, B, C, D
# Order: A4, B4, C4, D4, A8, B8, C8, D8, A12, B12, C12, D12, A16, B16, C16, D16, A20, B20, C20, D20
FLOW_RATES = [4, 8, 12, 16, 20]
LETTERS = ['A', 'B', 'C', 'D']

EXPERIMENTS = []
for flow_rate in FLOW_RATES:
    for letter in LETTERS:
        EXPERIMENTS.append((letter, flow_rate))

MAX_RETRIES = 3

# Set up logging
logging.basicConfig(
    filename='run_binary_batch.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
)

def main():
    print(f"Running {len(EXPERIMENTS)} binary search experiments in flow mode...")
    print(f"Flow rates: {FLOW_RATES} µL/s")
    print(f"Letters: {LETTERS}")
    print(f"Order: {', '.join([f'{letter}{rate}' for letter, rate in EXPERIMENTS])}")
    
    for i, (letter, flow_rate) in enumerate(EXPERIMENTS, 1):
        print(f"\n{'='*50}")
        print(f"Experiment {i}/{len(EXPERIMENTS)}: Pump {letter} at {flow_rate} µL/s")
        print(f"{'='*50}")
        
        attempt = 0
        success = False
        while attempt < MAX_RETRIES and not success:
            attempt += 1
            logging.info(f"Starting run: Letter={letter}, Flow={flow_rate}, Attempt={attempt}")
            print(f"Running binary calibration for {letter} at {flow_rate} µL/s (Attempt {attempt})")
            
            result = subprocess.run([
                sys.executable, 
                '-m', 
                'calibration_code.run_binary', 
                letter, 
                str(flow_rate),
                '--mode', 'flow'
            ])
            
            if result.returncode == 0:
                logging.info(f"Success: Letter={letter}, Flow={flow_rate} on attempt {attempt}")
                print(f"Completed {letter} at {flow_rate} µL/s.")
                success = True
            else:
                logging.error(f"Failure: Letter={letter}, Flow={flow_rate}, Attempt={attempt}, ReturnCode={result.returncode}")
                print(f"Run failed for {letter} at {flow_rate} µL/s (Attempt {attempt}).")
                if attempt < MAX_RETRIES:
                    print("Waiting 1 second before retry...")
                    time.sleep(1)
        
        if not success:
            logging.error(f"Max retries reached: Letter={letter}, Flow={flow_rate}. Skipping.")
            print(f"Max retries reached for {letter} at {flow_rate} µL/s. Skipping.")
    
    print(f"\n{'='*50}")
    print(f"All experiments completed!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 
