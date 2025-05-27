import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PWM_Motor import PWM_Motor, initialize_gpio, cleanup_gpio
from adafruit_ina219 import INA219
import board
import numpy as np
from bioreactor import Bioreactor
import csv
from utils import measure_and_write_sensor_data, create_csv_writer

# Map sensor indices:
# 0: tube_out, 1: tube_in, 2: flask_1, 3: flask_2

# Temperature setpoint and PID gains (defaults match Bioreactor)
T_SET = 30.0
KP = 10.0
KI = 1.0
KD = 0.0
DT = 1.0  # seconds per loop


def update_plot(lines, times, outputs, ax, fig):
    """Update the real-time plot with new data."""
    for i, line in enumerate(lines):
        line.set_data(times, outputs[i])
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    # Initialize Bioreactor and data storage
    bioreactor = Bioreactor()
    duration = 45000  # seconds
    times: list[float] = []
    temperature: list[list[float]] = [[] for _ in range(4)]

    # Set up the plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    lines = [ax.plot([], [], label=f'Sensor {i+1}')[0] for i in range(4)]
    ax.set_xlabel('Time (s)')
    ax.set_title('Real-time Temperature Data')
    ax.legend(loc=3, fontsize=8, bbox_to_anchor=(0.2, 0.2))

    # Open output file for sensor data
    with open('data/temperature_training_4.txt', 'w', newline='') as out_file:
        writer = create_csv_writer(out_file)

        start = time.time()
        elapsed = 0.0
        while elapsed < duration:
            # Measure and record sensors (populates external arrays if applicable)
            measure_and_write_sensor_data(bioreactor, writer, elapsed)

            # Temperature control via Bioreactor PID
            bioreactor.pid_temp_controller(
                setpoint=T_SET,
                kp=KP,
                ki=KI,
                kd=KD,
                dt=DT
            )

            # Plotting update
            times.append(elapsed)
            # Retrieve latest temps for plotting
            temps = bioreactor.get_vial_temp()
            for i, temp in enumerate(temperature):
                temp.append(temps[i])
            update_plot(lines, times, temperature, ax, fig)

            # Pause until next cycle
            time.sleep(DT)
            elapsed = time.time() - start

    # Clean up hardware
    bioreactor.finish()


if __name__ == '__main__':
    try:
        main()
    finally:
        cleanup_gpio()
