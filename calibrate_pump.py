import time
import csv
import numpy as np
import serial
from ticlib import TicUSB
from math import floor

import re

# Configuration
STEPS_MIN = 30 *1000             # steps/sec minimum
STEPS_MAX = 400 * 1000           # steps/sec maximum
NUM_POINTS = 21            # number of test points
DURATION = 60.0            # seconds per test

DENSITY_OF_WATER = 1.0     # g/ml
FLASK_CAPACITY = 1000.0      # ml
STEP_MODE = 3             # step mode
STEPS_PER_PULSE = 0.5 ** STEP_MODE     # steps/pulse
ML_PER_STEP_ESTIMATE = 0.001        # ml/step (estimated empirically, make sure this is greater than actual value)
SERIAL_PORT_SCALE = "/dev/ttyUSB0"
BAUDRATE = 9600
TOLERANCE = 0.001         # g tolerance for stability
READ_TIMEOUT = 1          # serial read timeout

def parse_weight(s: str) -> float:
    """
    Extracts a floating-point weight from the ASCII response string.
    """
    m = re.search(r'[-+]?\d*\.\d+|\d+', s)
    return float(m.group()) if m else None


def read_stable_weight():
    """
    Polls for weight until two consecutive readings agree within TOLERANCE.
    Returns weight_g.
    """
    # Open serial for scale
    ser = serial.Serial(port=SERIAL_PORT_SCALE,
                              baudrate=BAUDRATE,
                              bytesize=serial.EIGHTBITS,
                              parity=serial.PARITY_NONE,
                              stopbits=serial.STOPBITS_ONE,
                              timeout=READ_TIMEOUT)
    
    prev = None
    while True:
        ser.write(b'w')
        raw = ser.read(18)
        try:
            text = raw.decode('ascii', errors='ignore')
        except:
            continue
        w = parse_weight(text)
        if w is None:
            continue
        if prev is not None and abs(w - prev) < TOLERANCE:
            ser.close()
            del ser
            return w
        prev = w
        time.sleep(3.0)

def calibrate_single_pump(pump_serial, direction):
    steps_rates = np.linspace(STEPS_MIN, STEPS_MAX, NUM_POINTS)
    steps_rates = np.unique(steps_rates.astype(int))
    print(f"{steps_rates=}")


    csv_filename = f"pump_{pump_serial}_{direction}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['steps_rate', 'duration', 'delta_mass', 'ml_rate'])
        for steps_rate in steps_rates:

            # Tare and initial weight
            mass0 = read_stable_weight()
            
            pump = TicUSB(serial_number=pump_serial)  # auto-detect port
            pump.halt_and_set_position(0)
            pump.energize()
            pump.exit_safe_start()
            pump.set_step_mode(3)
            pump.set_current_limit(32)
    
            # Start pump and maintain velocity
            vel = steps_rate / STEPS_PER_PULSE if direction == 'forward' else -steps_rate / STEPS_PER_PULSE
            vel = int(floor(vel))
            real_steps_rate = abs(vel) * STEPS_PER_PULSE
            
            t_start = time.time()
            while time.time() - t_start < DURATION:
                pump.set_target_velocity(vel)
            pump.set_target_velocity(0)
            real_duration = time.time() - t_start
            
            pump.deenergize()
            pump.enter_safe_start()
            del pump
            
            # Final weight
            mass1 = read_stable_weight()

            # Compute actual flow
            delta_mass = mass1 - mass0
            ml_rate = abs(delta_mass) / real_duration / DENSITY_OF_WATER

            writer.writerow([real_steps_rate, real_duration, delta_mass, ml_rate])
            print(f"Rate {real_steps_rate} steps/s -> actual {ml_rate:.4f} ml/s")

pump_serials = ['00473498']

def main():
    for pump_serial in pump_serials:
        for direction in ['forward', 'reverse']:
            print(f"\n=== Calibration for pump serial: {pump_serial}, direction: {direction} ===")
            calibrate_single_pump(pump_serial, direction)
            print(f"Calibration for pump serial: {pump_serial}, direction: {direction} complete.\n")

if __name__ == '__main__':
    main()
