import time
import csv
import numpy as np
import serial
from ticlib import TicUSB
import re
from config import Config

# Configuration
STEPS_MIN = 100            # steps/sec minimum
STEPS_MAX = 100000         # steps/sec maximum
NUM_POINTS = 10            # number of test points
DURATION = 20.0            # seconds per test

DENSITY_OF_WATER = 1.0     # g/ml
FLASK_CAPACITY = 50.0      # ml
STEP_MODE = 3             # step mode
STEPS_PER_PULSE = 0.5 ** STEP_MODE     # steps/pulse
ML_PER_STEP_ESTIMATE = 0.001        # ml/step (estimated empirically, make sure this is greater than actual value)
SERIAL_PORT_SCALE = "/dev/serial0"
BAUDRATE = 9600
TOLERANCE = 0.001         # g tolerance for stability
READ_TIMEOUT = 1          # serial read timeout

def parse_weight(s: str) -> float:
    """
    Extracts a floating-point weight from the ASCII response string.
    """
    m = re.search(r'[-+]?\d*\.\d+|\d+', s)
    return float(m.group()) if m else None


def read_stable_weight(ser: serial.Serial):
    """
    Polls for weight until two consecutive readings agree within TOLERANCE.
    Returns (weight_g, timestamp).
    """
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
        now = time.time()
        if prev is not None and abs(w - prev) < TOLERANCE:
            return w, now
        prev = w
        time.sleep(0.1)

def calibrate_single_pump(pump_serial, direction):
    steps_rates = np.logspace(np.log10(STEPS_MIN), np.log10(STEPS_MAX), NUM_POINTS)
    steps_rates = np.unique(steps_rates.astype(int))

    # Open serial for scale
    scale_ser = serial.Serial(port=SERIAL_PORT_SCALE,
                              baudrate=BAUDRATE,
                              bytesize=serial.EIGHTBITS,
                              parity=serial.PARITY_NONE,
                              stopbits=serial.STOPBITS_ONE,
                              timeout=READ_TIMEOUT)

    # Initialize pump via TTL serial
    pump = TicUSB(serial_number=pump_serial)  # auto-detect port
    pump.energize()
    pump.exit_safe_start()
    pump.set_step_mode(3)
    pump.set_current_limit(32)

    csv_filename = f"pump_{pump_serial}_{direction}.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['steps_rate', 'duration', 'delta_mass', 'ml_rate'])

        projected_ml = 0.0
        for steps_rate in steps_rates:
            projected_ml += steps_rate * DURATION * ML_PER_STEP_ESTIMATE
            if projected_ml > FLASK_CAPACITY:
                input(f"Projected {projected_ml:.1f} ml exceeds flask capacity ({FLASK_CAPACITY} ml). Empty & press Enter to continue.")
                scale_ser.write(b't')
                read_stable_weight(scale_ser)
                projected_ml = steps_rate * DURATION * ML_PER_STEP_ESTIMATE

            # Tare and initial weight
            scale_ser.write(b't')
            mass0, t0 = read_stable_weight(scale_ser)

            # Start pump and maintain velocity
            vel = steps_rate / STEPS_PER_PULSE if direction == 'forward' else -steps_rate / STEPS_PER_PULSE
            t_start = time.monotonic()
            while time.monotonic() - t_start < DURATION:
                pump.set_target_velocity(vel)
            pump.set_target_velocity(0)

            # Final weight
            mass1, t1 = read_stable_weight(scale_ser)

            # Compute actual flow
            delta_mass = mass1 - mass0
            ml_rate = delta_mass / DURATION / DENSITY_OF_WATER

            writer.writerow([steps_rate, DURATION, delta_mass, ml_rate])
            print(f"Rate {steps_rate} steps/s -> actual {ml_rate:.4f} ml/s")

    # Cleanup
    pump.deenergize()
    pump.enter_safe_start()
    scale_ser.close()

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
