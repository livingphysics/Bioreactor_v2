import u3
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Real time theremometer readings 

inputs = [0, 1, 2, 3]
gain = [3.15,3.15,1,1] 
d = u3.U3()

# Prepare data storage for plotting
max_points = 200  # Maximum number of points to display on the plot
times = []
temperature = [[] for _ in range(len(inputs))]
outputs = [[] for _ in range(len(inputs))]  

# Set up the plot
plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots(figsize=(10, 6))
lines = [ax.plot([], [], label=f'Sensor {i+1}')[0] for i,_ in enumerate(inputs)]
ax.set_xlabel('Time (h)')
# ax.set_ylabel('Temperature (°C)')
ax.set_title('Real-time Temperature Data')
ax.legend(loc=3,fontsize=8,bbox_to_anchor=(0.2, 0.2))

# Function to update the plot
def update_plot():
    for i, line in enumerate(lines):
	     line.set_data(times, outputs[i])
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
# d.getCalibrationData()
# ain2bits = d.getFeedback(u3.AIN(2))
# print(ain2bits)
# ainValue = d.binaryToCalibratedAnalogVoltage(ain2bits, isLowVoltage = False, channelNumber = 2)
start = time.time()
while True:
	ainValues = [d.getAIN(sens) for sens in inputs]
	for i, sens in enumerate(inputs):
		temperature[i] = (((d.getAIN(sens))/gain[i])*100-32)*(5/9)
	current = time.time()
	elapsed = (current - start)/3600
	times.append(elapsed)
	for i, temp in enumerate(temperature):
		outputs[i].append(temp)
	# for i, voltage in enumerate(ainValues):
		# outputs[i].append(voltage)
	
	# # Keep only the last max_points
	# if len(times) > max_points:
		# times = times[-max_points:]
		# for i in range(len(inputs)):
			# outputs[i] = outputs[i][-max_points:]
	# Update the plot
	update_plot()
	time.sleep(1)
