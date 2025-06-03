import time
import csv
import numpy as np
import serial
from ticlib import TicUSB
from math import floor

import re

# Configuration
STEPS_MIN = 500*1000 / 8             # steps/sec minimum
STEPS_MAX = 1000*1000 / 8           # steps/sec maximum
NUM_POINTS = 2            # number of test points
DURATION = 20.0            # seconds per test

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
    Returns (weight_g, timestamp).
    """
    # Open serial for scale
    ser = serial.Serial(port=SERIAL_PORT_SCALE,
                              baudrate=BAUDRATE,
                              bytesize=serial.EIGHTBITS,
                              parity=serial.PARITY_NONE,
                              stopbits=serial.STOPBITS_ONE,
                              timeout=READ_TIMEOUT)
    # prev = None
    counter = 0
    while True:
        if counter>10:
            break
        ser.write(b'w')
        raw = ser.read(18)
        print(raw,'\n')
        # try:
        
            # text = raw.decode('ascii', errors='ignore')
        # except:
            # continue
        # w = parse_weight(text)
        # if w is None:
            # continue
        # now = time.time()
        # if prev is not None and abs(w - prev) < TOLERANCE:
            # return w, now
        # prev = w
        # time.sleep(0.1)
        counter+=1
    ser.close()
    del ser
    return 0, 0

def calibrate_single_pump(pump_serial, direction):
    steps_rates = np.logspace(np.log10(STEPS_MIN), np.log10(STEPS_MAX), NUM_POINTS)
    steps_rates = np.unique(steps_rates.astype(int))
    print(f"{steps_rates=}")


    csv_filename = f"pump_{pump_serial}_{direction}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['steps_rate', 'duration', 'delta_mass', 'ml_rate'])
        for steps_rate in steps_rates:

            # Tare and initial weight
            # scale_ser.write(b't')
            mass0, t0 = read_stable_weight()
            
            pump = TicUSB(serial_number=pump_serial)  # auto-detect port
            pump.halt_and_set_position(0)
            pump.energize()
            pump.exit_safe_start()
            pump.set_step_mode(3)
            pump.set_current_limit(32)
            print(f"{pump.get_planning_mode()=}")
    
            
            # Start pump and maintain velocity
            vel = steps_rate / STEPS_PER_PULSE if direction == 'forward' else -steps_rate / STEPS_PER_PULSE
            vel = int(floor(vel))
            print(f"{vel=}")
            print(type(vel))
            true_steps_rate = abs(vel) * STEPS_PER_PULSE
            t_start = time.time()
            
            print(f"{pump.get_planning_mode()=}")
            counter = 0
            while time.time() - t_start < DURATION:
                pump.set_target_velocity(vel)
                # print(vel)
                # print(f"{pump.get_max_speed()=}")
                # print(f"{pump.get_max_acceleration()=}")
                # print(f"{pump.get_current_velocity()=}")
                # # print(f"{pump.get_status_flags()}")
                # counter += 1
                # if counter >= 10:
                    # break
            pump.set_target_velocity(0)
            
            pump.deenergize()
            pump.enter_safe_start()
            del pump
            # Final weight
            mass1, t1 = read_stable_weight()

            # Compute actual flow
            # delta_mass = mass1 - mass0
            # ml_rate = delta_mass / DURATION / DENSITY_OF_WATER

            # writer.writerow([true_steps_rate, DURATION, delta_mass, ml_rate])
            # print(f"Rate {true_steps_rate} steps/s -> actual {ml_rate:.4f} ml/s")

    # Cleanup


pump_serials = ['00473498']

def main():
    for pump_serial in pump_serials:
        for direction in ['forward', 'reverse']:
            print(f"\n=== Calibration for pump serial: {pump_serial}, direction: {direction} ===")
            input("Please empty the flask and press Enter to begin this calibration run...")
            calibrate_single_pump(pump_serial, direction)
            print(f"Calibration for pump serial: {pump_serial}, direction: {direction} complete.\n")

if __name__ == '__main__':
    main()
