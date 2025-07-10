import logging
import time
import os
import subprocess
import threading

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from src.bioreactor import Bioreactor
from src.utils import measure_and_write_sensor_data, pid_controller

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Temperature setpoint and PID gains (defaults match Bioreactor)
T_SET = 30.0
KP = 10.0
KI = 1.0
KD = 0.0
DT = 1.0  # seconds per loop
DURATION = 45000  # seconds

class LivePlotter:
    def __init__(self, plot_filename='temperature_plot.png'):
        self.plot_filename = plot_filename
        self.times = []
        self.temperature = [[] for _ in range(4)]
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.lines = [self.ax.plot([], [], label=f'Sensor {i+1}')[0] for i in range(4)]
        self.ax.set_xlabel('Time (s)')
        self.ax.set_title('Real-time Temperature Data')
        self.ax.legend(loc=3, fontsize=8, bbox_to_anchor=(0.2, 0.2))
        
        # Start image viewer in background
        self.start_image_viewer()
        
    def start_image_viewer(self):
        """Start an image viewer that will auto-refresh the plot"""
        try:
            # Try to start an image viewer (works on macOS, Linux)
            if os.name == 'posix':  # macOS/Linux
                # Try different image viewers
                viewers = ['open', 'xdg-open', 'display']
                for viewer in viewers:
                    try:
                        subprocess.Popen([viewer, self.plot_filename])
                        print(f"Started image viewer with {viewer}")
                        break
                    except FileNotFoundError:
                        continue
            else:  # Windows
                subprocess.Popen(['start', self.plot_filename], shell=True)
        except Exception as e:
            print(f"Could not start image viewer: {e}")
            print("You can manually open the plot file to see live updates")
    
    def update_plot(self, elapsed, temps):
        """Update the plot with new data"""
        self.times.append(elapsed)
        for i, temp in enumerate(temps):
            self.temperature[i].append(temp)
        
        # Update plot lines
        for i, line in enumerate(self.lines):
            line.set_data(self.times, self.temperature[i])
        
        # Auto-scale and save
        self.ax.relim()
        self.ax.autoscale_view()
        plt.savefig(self.plot_filename, dpi=150, bbox_inches='tight')
        
        # Print current values for console monitoring
        print(f"Time: {elapsed:.1f}s, Temps: {[f'{t:.2f}°C' for t in temps]}")

def main():
    plotter = LivePlotter()
    
    def job(bioreactor, elapsed):
        # PID control
        pid_controller(bioreactor, setpoint=T_SET, kp=KP, ki=KI, kd=KD, dt=DT)
        # Data writing
        measure_and_write_sensor_data(bioreactor, elapsed)
        # Update plot with new data
        temps = bioreactor.get_vial_temp()
        plotter.update_plot(elapsed, temps)

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
    finally:
        plt.close('all')

if __name__ == '__main__':
    main() 