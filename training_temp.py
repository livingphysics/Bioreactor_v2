# training data collection
# collecting temperature data for 2 in-flask thermometers
# and 2 thermometers in cooling/heating stream
# current reading from INA219

from PWM_Motor import PWM_Motor, initialize_gpio, cleanup_gpio
from adafruit_ina219 import INA219
import board
import u3
import time
import threading
import numpy as np

# 0: tube_out
# 1: tube_in
# 2: flask_1
# 3: flask_2

inputs = [0, 1, 2, 3]
gain = [3.15,3.15,1,1] 
d = u3.U3()

i2c_bus = board.I2C()
ina219 = INA219(i2c_bus)

out_file = open('data/temperature_training_4.txt', 'w')
title = ['current', 'tube_out', 'tube_in', 'flask_1', 'flask_2']
out_file.write(','.join(title) + '\n')

initialize_gpio()
peltier = PWM_Motor(24, 25, 1000)
def peltier_modulation():
	for i in range(90):
		# 900s run, current step decrease (100 to 10)
		peltier.run(100 - i, forward=True)
		time.sleep(500)
	# for i in range(180):
		# # 900s run, current step decrease (10 to 100)
		# peltier.run(100 - i // 2, forward=False)
		# time.sleep(5)
		
	# time.sleep(1800)
	
	# for i in range(180):
		# # 1800s run, current step increase (10 to 100)
		# peltier.run(10 + i // 2, forward=True)
		# time.sleep(10)
	# for i in range(180):
		# # 1800s run, current step increase (10 to 100)
		# peltier.run(10 + i // 2, forward=False)
		# time.sleep(10)
		
	# time.sleep(1800)
	
	# for i in range(180):
		# # 1800s run, constant forward current
		# peltier.run(50, forward=True)
		# time.sleep(10)
	# for i in range(180):
		# # 1800s run, constant reverse current
		# peltier.run(50, forward=False)
		# time.sleep(10)
	# for i in range(360):
		# # 1800s run, constant forward current
		# peltier.run(20, forward=True)
		# time.sleep(5)
	# for i in range(360):
		# # 1800s run, constant reverse current
		# peltier.run(20, forward=False)
		# time.sleep(5)
		
	# time.sleep(1800)
	
duration = 45000
start = time.time()
heating_thread = threading.Thread(target=peltier_modulation)
heating_thread.start()

current_time = time.time()
while current_time - start < duration:
	
	start_loop = time.time()
	# current by default in mA
	try:
		current = ina219.current / 1000
	
		# 4 temperature readings
		temperature = []
		ainValues = [d.getAIN(sens) for sens in inputs]
		for i, sens in enumerate(inputs):
			temperature.append((((d.getAIN(sens))/gain[i])*100-32)*(5/9))

		line = f'{current},{",".join(map(str,temperature))}'
	except:
		line = ",".join(['NaN']*5)
	finally:
		end_loop = time.time()
		
		# calculates correct pausing time
		diff = end_loop - start_loop
		dt = max(np.ceil(diff),diff) - diff
		pausing = dt + diff
		out_file.write(line + '\n')
		
		# sets interval between measurements
		interval = 10
		time.sleep(interval - pausing + dt)
		
		current_time = time.time()

print(current_time-start)
out_file.close()
peltier.cleanup()
cleanup_gpio()
