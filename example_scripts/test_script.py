import logging
import time

from src.bioreactor import Bioreactor
from src.utils import measure_and_write_sensor_data, pid_controller

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DURATION = 45000
DT = 1.0

def main():
    # Set up the plot
    times = []
    temperature = [[] for _ in range(4)]

    def job(bioreactor, elapsed):
            measure_and_write_sensor_data(bioreactor, elapsed)
    jobs = [
        (job, DT, True)
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