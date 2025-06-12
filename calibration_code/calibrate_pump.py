"""
Example usage:
    python calibrate_pump.py                # Calibrate all pumps forward
    python calibrate_pump.py --reverse      # Calibrate all pumps in reverse
    python calibrate_pump.py A              # Calibrate both A_in and A_out forward
    python calibrate_pump.py B --reverse    # Calibrate both B_in and B_out in reverse
    python calibrate_pump.py A_in B C_out   # Calibrate A_in, B_in, B_out, and C_out forward (in order)
    python calibrate_pump.py A_in B --both  # Calibrate A_in, B_in, B_out both directions

If no pump or letter is given, all pumps are calibrated. Default direction is forward.
"""
import time
import csv
import numpy as np
import serial
from ticlib import TicUSB
from math import floor
import re
import sys
from config import Config as cfg
from datetime import datetime

# Configuration
STEPS_MIN = 50 *1000             # steps/sec minimum
STEPS_MAX = 250 * 1000           # steps/sec maximum
NUM_POINTS = 41            # number of test points
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

def calibrate_single_pump(pump_serial, direction, repeats=3, pump_key=None):
    steps_rates = np.linspace(STEPS_MIN, STEPS_MAX, NUM_POINTS)
    steps_rates = np.unique(steps_rates.astype(int))

    from datetime import datetime
    date_str = datetime.now().strftime('%Y%m%d')
    if pump_key is None:
        pump_key = pump_serial
    # Save CSV in calibration_results directory
    csv_filename = f"calibration_results/{date_str}_calibration_{pump_key}_{direction}_results.csv"
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['steps_rate', 'duration', 'delta_mass', 'ml_rate'])
        for _ in range(repeats):
            np.random.shuffle(steps_rates)  # Randomize the order
            print(f"{steps_rates=}")
            for steps_rate in steps_rates:

                # Initial weight
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

def parse_args():
    args = sys.argv[1:]
    directions = []
    pump_tokens = []
    direction_flag = 'forward'
    for arg in args:
        if arg == '--forward':
            direction_flag = 'forward'
        elif arg == '--reverse':
            direction_flag = 'reverse'
        elif arg == '--both':
            direction_flag = 'both'
        elif arg.startswith('--'):
            print(f"Unknown flag: {arg}")
            sys.exit(1)
        else:
            pump_tokens.append(arg)
    return pump_tokens, direction_flag

# Helper to expand user tokens to pump keys
PUMP_KEYS = list(cfg.PUMPS.keys())
PUMP_LETTERS = ['A', 'B', 'C', 'D']

def expand_pump_tokens(tokens):
    # If no tokens, return all pumps
    if not tokens:
        return PUMP_KEYS.copy()
    result = []
    for token in tokens:
        t = token.upper()
        if t in PUMP_LETTERS:
            # Add both in and out for this letter
            result.append(f'{t}_in')
            result.append(f'{t}_out')
        elif t.endswith('_IN') or t.endswith('_OUT'):
            # Add specific pump if valid
            key = t.capitalize() if t[1] == '_' else t[0].upper() + t[1:].lower()
            if key in cfg.PUMPS:
                result.append(key)
            else:
                print(f"Unknown pump: {token}")
                sys.exit(1)
        else:
            print(f"Unknown argument: {token}")
            sys.exit(1)
    return result

def main():
    pump_tokens, direction_flag = parse_args()
    pump_keys = expand_pump_tokens(pump_tokens)
    # Remove duplicates but preserve order
    seen = set()
    ordered_pump_keys = []
    for k in pump_keys:
        if k not in seen:
            ordered_pump_keys.append(k)
            seen.add(k)
    directions = ['forward'] if direction_flag == 'forward' else ['reverse'] if direction_flag == 'reverse' else ['forward', 'reverse']
    for direction in directions:
        for pump_key in ordered_pump_keys:
            pump_serial = cfg.PUMPS[pump_key]['serial']
            print(f"\n=== Calibration for {pump_key} (serial: {pump_serial}), direction: {direction} ===")
            calibrate_single_pump(pump_serial, direction, repeats=3, pump_key=pump_key)
            print(f"Calibration for {pump_key} (serial: {pump_serial}), direction: {direction} complete.\n")

if __name__ == '__main__':
    main()
