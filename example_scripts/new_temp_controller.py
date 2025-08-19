import logging
import time

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from src.bioreactor import Bioreactor
from src.utils import measure_and_write_sensor_data, pid_controller, compensated_flow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Temperature setpoint and PID gains (defaults match Bioreactor)
T_SET = 30.0
KP = 10.0
KI = 0.1
KD = 0.0
DT = 1.0  # seconds per loop
DURATION = 360000  # seconds

FLOW_DT = 30.0
FLOW_DOSE = 5.0

def main():
    # Set up the plot
    times = []
    temperature = [[] for _ in range(5)]
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = [ax.plot([], [], label=f'Sensor {i+1}')[0] for i in range(5)]
    ax.set_xlabel('Time (s)')
    ax.set_title('Real-time Temperature Data')
    ax.legend(loc=3, fontsize=8, bbox_to_anchor=(0.2, 0.2))

    def job(bioreactor, elapsed):
        # PID control
        pid_controller(bioreactor, setpoint=T_SET, kp=KP, ki=KI, kd=KD, dt=DT)
        # Data writing
        measure_and_write_sensor_data(bioreactor, elapsed)
        # Plotting
        times.append(elapsed)
        temps = bioreactor.get_vial_temp()
        temps.append(bioreactor.get_ambient_temp())
        for i, temp in enumerate(temperature):
            temp.append(temps[i])
        for i, line in enumerate(lines):
            line.set_data(times, temperature[i])
        ax.relim()
        ax.autoscale_view()
        # Save plot to file instead of interactive display
        plt.savefig('temperature_plot.png', dpi=150, bbox_inches='tight')
        
    def flow_job(bioreactor, elapsed):
        compensated_flow(bioreactor,'All', 0.010, FLOW_DOSE,FLOW_DT, elapsed)

    jobs = [
        (job, DT, True), 
        (flow_job, FLOW_DT, DURATION)
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
