import u3
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PWM_Motor import PWM_Motor, initialize_gpio, cleanup_gpio
import time
from adafruit_ina219 import INA219
import board
import numpy as np

# 0: tube_out
# 1: tube_in
# 2: flask_1
# 3: flask_2
T_set = 30

inputs = [0, 1, 2, 3]
gain = [3.15,3.15,1,1] 
d = u3.U3()

i2c_bus = board.I2C()
ina219 = INA219(i2c_bus)

out_file = open('data/temperature_training_4.txt', 'w')
title = ['time','current', 'tube_out', 'tube_in', 'flask_1', 'flask_2']
out_file.write(','.join(title) + '\n')

initialize_gpio()
peltier = PWM_Motor(24, 25, 1000)

duration = 45000
start = time.time()


# Prepare data storage for plotting
times = []
temperature = [[] for _ in range(len(inputs))]
outputs = [[] for _ in range(len(inputs))]  

# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 6))
lines = [ax.plot([], [], label=f'Sensor {i+1}')[0] for i,_ in enumerate(inputs)]
ax.set_xlabel('Time (s)')

ax.set_title('Real-time Temperature Data')
ax.legend(loc=3,fontsize=8,bbox_to_anchor=(0.2, 0.2))
current_time = time.time()
elapsed = (current_time - start)
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
    
while current_time - start < duration:
	try:
		current = ina219.current / 1000
		# 4 temperature readings
		ainValues = [d.getAIN(sens) for sens in inputs]
		for i, sens in enumerate(inputs):
			temperature[i]=((((d.getAIN(sens))/gain[i])*100-32)*(5/9))
		data = f'{elapsed},{current},{",".join(map(str,temperature))}'
	except:
		tmp = ['NaN']*5
		data = f'{elapsed},{",".join(tmp)}'
	finally:
		current_time = time.time()
		elapsed = (current_time - start)
		dT = T_set-temperature[3]
		error.append(dT)
		signal = min(100,(kp*abs(dT)+ki*sum(error)))
		if dT>0:
			peltier.run(signal, forward=True)
		else:
			peltier.run(signal, forward=False)
		print(signal)
		times.append(elapsed)
		for i, temp in enumerate(temperature):
			outputs[i].append(temp)
		update_plot()
		out_file.write(data + '\n')
		time.sleep(1) 

print(current_time-start)
out_file.close()
peltier.cleanup()
cleanup_gpio()
