import logging
import time

from src.bioreactor import Bioreactor
from src.utils import measure_and_write_sensor_data, compensated_flow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DURATION = 1000
DT = 30.0
DOSE = 10.0

def main():


    def flow_job(bioreactor, elapsed):
            compensated_flow(bioreactor,'All', 0.020, DOSE,DT, elapsed)
    jobs = [
        (flow_job, DT, DURATION)
    ]

    try:
        with Bioreactor() as bioreactor:
            bioreactor.run(jobs)
            start = time.time()
            while time.time() - start < DURATION:
                time.sleep(1)
    except Exception as e:
        logging.error(f"Error: {e}")

if __name__ == '__main__':
    main()
