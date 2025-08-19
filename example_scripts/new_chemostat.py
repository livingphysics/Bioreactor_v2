import logging
import time
import numpy as np

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
DURATION = 720000  # seconds

# Flow control parameters
FLOW_DT = 360.0
FLOW_DOSE = 28.0

# OD measurement parameters
OD_CALIBRATION_135 = 1.0  # Calibration factor for 135° sensors
OD_CALIBRATION_180 = 1.0  # Calibration factor for 180° sensors
OD_OFFSET = 0.0  # Offset for OD calculation

# Data logging parameters
DATA_DT = 10.0  # seconds between data logging

def calculate_od(photodiode_reading, calibration_factor=2.0, offset=0.1):
    """
    Calculate OD from photodiode reading using simple linear calibration.
    Args:
        photodiode_reading: Raw voltage reading from photodiode
        calibration_factor: Calibration factor (default 2.0)
        offset: Offset for OD calculation (default 0.1)
    Returns:
        float: Calculated OD value
    """
    od = calibration_factor * photodiode_reading - offset
    return max(0.0, od)

def main():
    # Set up the temperature plot
    temp_times = []
    temperature = [[] for _ in range(5)]
    temp_fig, temp_ax = plt.subplots(figsize=(10, 6))
    temp_lines = [temp_ax.plot([], [], label=f'Sensor {i+1}')[0] for i in range(5)]
    temp_ax.set_xlabel('Time (s)')
    temp_ax.set_title('Real-time Temperature Data')
    temp_ax.legend(loc=3, fontsize=8, bbox_to_anchor=(0.2, 0.2))

    # Set up the OD plot
    od_times = []
    od_135 = [[] for _ in range(4)]  # 4 vials (A, B, C, D)
    od_180 = [[] for _ in range(4)]  # 4 vials (A, B, C, D)
    od_fig, od_ax = plt.subplots(figsize=(10, 6))
    od_lines_135 = [od_ax.plot([], [], label=f'Vial {chr(65+i)} 135°')[0] for i in range(4)]
    od_lines_180 = [od_ax.plot([], [], label=f'Vial {chr(65+i)} 180°', linestyle='--')[0] for i in range(4)]
    od_ax.set_xlabel('Time (s)')
    od_ax.set_ylabel('Optical Density (OD)')
    od_ax.set_title('Real-time OD Data')
    od_ax.legend(loc='upper left', fontsize=8)
    od_ax.grid(True)

    def temp_job(bioreactor, elapsed):
        """Temperature control job"""
        # PID control
        pid_controller(bioreactor, setpoint=T_SET, kp=KP, ki=KI, kd=KD, dt=DT)
        
        # Update temperature plot
        temp_times.append(elapsed)
        temps = bioreactor.get_vial_temp()
        temps.append(bioreactor.get_ambient_temp())
        for i, temp in enumerate(temperature):
            temp.append(temps[i])
        for i, line in enumerate(temp_lines):
            line.set_data(temp_times, temperature[i])
        temp_ax.relim()
        temp_ax.autoscale_view()
        # Save temperature plot
        temp_fig.savefig('temperature_plot.png', dpi=150, bbox_inches='tight')
        
    def flow_job(bioreactor, elapsed):
        """Flow control job"""
        compensated_flow(bioreactor, 'All', 0.010, FLOW_DOSE, FLOW_DT, elapsed)

    def od_and_data_job(bioreactor, elapsed):
        """Combined OD measurement, plotting, and data logging job"""
        # Get photodiode readings (shared between OD calculation and data logging)
        photodiodes = bioreactor.get_photodiodes()
        
        # Data logging (uses the same photodiode readings)
        io_temps = bioreactor.get_io_temp()
        vial_temps = bioreactor.get_vial_temp()
        ambient_temp = bioreactor.get_ambient_temp()
        peltier_current = bioreactor.get_peltier_curr()

        # Pad lists to ensure correct length
        photodiodes_padded = photodiodes + [float('nan')] * (12 - len(photodiodes))
        io_temps += [float('nan')] * (2 - len(io_temps))
        vial_temps += [float('nan')] * (4 - len(vial_temps))

        # Write data to CSV
        from src.config import Config as cfg
        data_row = {
            'time': elapsed,
            cfg.SENSOR_LABELS['photodiode_1']: photodiodes_padded[0],
            cfg.SENSOR_LABELS['photodiode_2']: photodiodes_padded[1],
            cfg.SENSOR_LABELS['photodiode_3']: photodiodes_padded[2],
            cfg.SENSOR_LABELS['photodiode_4']: photodiodes_padded[3],
            cfg.SENSOR_LABELS['photodiode_5']: photodiodes_padded[4],
            cfg.SENSOR_LABELS['photodiode_6']: photodiodes_padded[5],
            cfg.SENSOR_LABELS['photodiode_7']: photodiodes_padded[6],
            cfg.SENSOR_LABELS['photodiode_8']: photodiodes_padded[7],
            cfg.SENSOR_LABELS['photodiode_9']: photodiodes_padded[8],
            cfg.SENSOR_LABELS['photodiode_10']: photodiodes_padded[9],
            cfg.SENSOR_LABELS['photodiode_11']: photodiodes_padded[10],
            cfg.SENSOR_LABELS['photodiode_12']: photodiodes_padded[11],
            cfg.SENSOR_LABELS['io_temp_1']: io_temps[0],
            cfg.SENSOR_LABELS['io_temp_2']: io_temps[1],
            cfg.SENSOR_LABELS['vial_temp_1']: vial_temps[0],
            cfg.SENSOR_LABELS['vial_temp_2']: vial_temps[1],
            cfg.SENSOR_LABELS['vial_temp_3']: vial_temps[2],
            cfg.SENSOR_LABELS['vial_temp_4']: vial_temps[3],
            cfg.SENSOR_LABELS['ambient_temp']: ambient_temp,
            cfg.SENSOR_LABELS['peltier_current']: peltier_current
        }

        bioreactor.writer.writerow(data_row)
        bioreactor.out_file.flush()
        
        # OD calculation and plotting
        if len(photodiodes) >= 8:  # Ensure we have enough readings
            # Calculate OD for 135° sensors (channels 0-3)
            for i in range(4):
                if i < len(photodiodes):
                    od_val_135 = calculate_od(photodiodes[i], OD_CALIBRATION_135, OD_OFFSET)
                    od_135[i].append(od_val_135)
                else:
                    od_135[i].append(float('nan'))
            
            # Calculate OD for 180° sensors (channels 4-7)
            for i in range(4):
                if i + 4 < len(photodiodes):
                    od_val_180 = calculate_od(photodiodes[i + 4], OD_CALIBRATION_180, OD_OFFSET)
                    od_180[i].append(od_val_180)
                else:
                    od_180[i].append(float('nan'))
            
            # Update OD plot
            od_times.append(elapsed)
            
            # Update 135° lines
            for i, line in enumerate(od_lines_135):
                line.set_data(od_times, od_135[i])
            
            # Update 180° lines
            for i, line in enumerate(od_lines_180):
                line.set_data(od_times, od_180[i])
            
            od_ax.relim()
            od_ax.autoscale_view()
            # Save OD plot
            od_fig.savefig('od_plot.png', dpi=150, bbox_inches='tight')
            
            logging.info(f"OD measurements at {elapsed}s: 135°={[f'{x:.3f}' for x in od_135[-1] if not np.isnan(x)]}, 180°={[f'{x:.3f}' for x in od_180[-1] if not np.isnan(x)]}")

    # Define jobs with their intervals and durations
    jobs = [
        (temp_job, DT, True),                    # Temperature control every DT seconds
        (flow_job, FLOW_DT, DURATION),           # Flow control every FLOW_DT seconds
        (od_and_data_job, DATA_DT, True)         # Combined OD measurement and data logging every DATA_DT seconds
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
