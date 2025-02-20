import time
from adafruit_ina219 import INA219
import board
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Current Sensor real time plotter

times = []
dc_current = [[] for _ in range(1)]  

i2c_bus = board.I2C()
ina219 = INA219(i2c_bus)

# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax1 = plt.subplots(figsize=(10, 6))

live_current = [ax1.plot([], [], label=f'Current {i}')[0] for i in range(1)]

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Current (A)')
ax1.tick_params(axis='y', labelcolor='r')

plt.title('Real-time Current Data')
ax1.legend(loc='upper left')

def update_plot():
    for i, line in enumerate(live_current):
        line.set_data(times, dc_current[i])
    ax1.relim()
    ax1.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

start = time.time()

while True:
	current = time.time()
	elapsed = current - start
	# current in mA by default
	# print(ina219.current / 1000)
	# time.sleep(1)
	times.append(elapsed)
	for i, reading in enumerate(live_current):
		dc_current[i].append(ina219.current / 1000)
	
	# Update the plot
	update_plot()
	


