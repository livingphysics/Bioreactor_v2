import u3
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PWM_Motor import PWM_Motor, initialize_gpio, cleanup_gpio
import time
from adafruit_ina219 import INA219
import board
import numpy as np
from bioreactor import Bioreactor
import csv
from utils import measure_and_write_sensor_data, create_csv_writer

# 0: tube_out
# 1: tube_in
# 2: flask_1
# 3: flask_2
T_set = 30

bioreactor = Bioreactor()

duration = 45000

# Prepare data storage for plotting
times = []
temperature = [[] for _ in range(4)]
outputs = [[] for _ in range(4)]  

# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 6))
lines = [ax.plot([], [], label=f'Sensor {i+1}')[0] for i in range(4)]
ax.set_xlabel('Time (s)')

ax.set_title('Real-time Temperature Data')
ax.legend(loc=3,fontsize=8,bbox_to_anchor=(0.2, 0.2))

kp=10
ki=1
error=[]
def update_plot():
    for i, line in enumerate(lines):
        line.set_data(times, outputs[i])
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

with open('data/temperature_training_4.txt', 'w', newline='') as out_file:
    writer = create_csv_writer(out_file)

    elapsed = 0
    start = time.time()
    while elapsed < duration:
        measure_and_write_sensor_data(bioreactor, writer, elapsed)

        dT = T_set-temperature[3]
        error.append(dT)
        signal = min(100,(kp*abs(dT)+ki*sum(error)))
        if dT>0:
            bioreactor.change_peltier(signal, forward=True)
        else:
            bioreactor.change_peltier(signal, forward=False)
        print(signal)
        times.append(elapsed)
        for i, temp in enumerate(temperature):
            outputs[i].append(temp)
        update_plot()
        time.sleep(1)
        # calculates correct pausing time

        elapsed = time.time() - start

bioreactor.finish()
