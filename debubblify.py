import time
from ticlib import TicUSB
from math import floor

# List of pump serials (edit as needed)
pump_serials = ['00473498', '00473504', '00473510', '00473552', '00473517', '00473408', '00473497', '00473491']  # Add more serials as needed

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
