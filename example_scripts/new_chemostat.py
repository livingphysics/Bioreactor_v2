import logging
import time

from src.bioreactor import Bioreactor
from src.utils import measure_and_write_sensor_data, pid_controller, compensated_flow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Temperature setpoint and PID gains (defaults match Bioreactor)
T_SET = 27.0
KP = 5.0
KI = 0.005
KD = 0.0
DT = 5.0  # seconds per loop
DURATION = 720000  # seconds

# Flow control parameters
FLOW_DT = 90.0
FLOW_DOSE = 14.0

# Data logging parameters
DATA_DT = 30.0  # seconds between data logging

def main():
    def temp_job(bioreactor, elapsed):
        """Temperature control job"""
        # PID control
        pid_controller(bioreactor, setpoint=T_SET, kp=KP, ki=KI, kd=KD, dt=DT)
        
    def flow_job(bioreactor, elapsed):
        """Flow control job"""
        compensated_flow(bioreactor, 'All', 0.010, FLOW_DOSE, FLOW_DT, elapsed)

    def data_job(bioreactor, elapsed):
        """Data logging job"""
        measure_and_write_sensor_data(bioreactor, elapsed)

    # Define jobs with their intervals and durations
    jobs = [
        (temp_job, DT, True),                    # Temperature control every DT seconds
        # (flow_job, FLOW_DT, DURATION),           # Flow control every FLOW_DT seconds
        (data_job, DATA_DT, True)                # Data logging every DATA_DT seconds
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
