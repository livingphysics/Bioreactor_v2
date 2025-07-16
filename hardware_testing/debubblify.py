"""
Example usage:
    python debubblify.py                # Run all pumps forward
    python debubblify.py --reverse      # Run all pumps in reverse
    python debubblify.py A              # Run vial A pump forward
    python debubblify.py B --reverse    # Run vial B pump in reverse
    python debubblify.py D --forward    # Run vial D pump forward

If no letter is given, all pumps are run. Default direction is forward.
"""
import time
import sys
from ticlib import TicUSB
from math import floor
from src.config import Config as cfg

# --- Argument parsing ---
def parse_args():
    letter = None
    direction = 'forward'
    args = sys.argv[1:]
    for arg in args:
        if arg.startswith('--'):
            if arg == '--forward':
                direction = 'forward'
            elif arg == '--reverse':
                direction = 'reverse'
            else:
                print(f"Unknown flag: {arg}")
                sys.exit(1)
        elif len(arg) == 1 and arg.upper() in 'ABCD':
            letter = arg.upper()
        else:
            print(f"Unknown argument: {arg}")
            sys.exit(1)
    return letter, direction

letter, direction = parse_args()

# --- Determine pump serials ---
def get_pump_serials(letter, direction):
    pumps = []
    if letter is None:
        # All pumps, all directions
        for pump_cfg in cfg.PUMPS.values():
            if pump_cfg['direction'] == direction:
                pumps.append(pump_cfg['serial'])
    else:
        # Only pumps for the given letter
        in_key = f'{letter}_in'
        out_key = f'{letter}_out'
        if direction == 'forward':
            pumps.append(cfg.PUMPS[in_key]['serial'])
        elif direction == 'reverse':
            pumps.append(cfg.PUMPS[out_key]['serial'])
    return pumps

pump_serials = get_pump_serials(letter, direction)
if not pump_serials:
    print(f"No pumps found for letter={letter}, direction={direction}")
    sys.exit(1)

STEP_MODE = 3  # Step mode (as in calibration)
STEPS_PER_PULSE = 0.5 ** STEP_MODE  # steps/pulse
TARGET_STEPS_PER_SEC = 2_000_000  # Target steps/sec

# Calculate velocity (pulses/sec)
velocity = int(floor(TARGET_STEPS_PER_SEC / STEPS_PER_PULSE))

# Initialize pumps
def setup_pump(serial):
    pump = TicUSB(serial_number=serial)
    pump.energize()
    pump.exit_safe_start()
    pump.set_step_mode(STEP_MODE)
    pump.set_current_limit(32)
    return pump

pumps = []
for serial in pump_serials:
    try:
        pump = setup_pump(serial)
        pumps.append(pump)
        print(f"Pump {serial} initialized.")
    except Exception as e:
        print(f"Error initializing pump {serial}: {e}")

print(f"Running all pumps forward at {TARGET_STEPS_PER_SEC} steps/sec (velocity={velocity})")

try:
    while True:
        for pump in pumps:
            try:
                pump.set_target_velocity(velocity)
            except Exception as e:
                print(f"Error setting velocity for pump: {e}")
        time.sleep(0.05)  # 50ms loop
except KeyboardInterrupt:
    print("Keyboard interrupt received. Stopping pumps.")
finally:
    for pump in pumps:
        try:
            pump.set_target_velocity(0)
            pump.deenergize()
            pump.enter_safe_start()
        except Exception as e:
            print(f"Error stopping pump: {e}")
    print("All pumps stopped and deenergized.")
