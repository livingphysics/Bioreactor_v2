"""
pump_control.py

A module for controlling individual pumps with specific parameters.

Example usage:
    python pump_control.py A in forward 1000    # Run pump A_in forward at 1000 steps/s
    python pump_control.py B out reverse 2000   # Run pump B_out reverse at 2000 steps/s
    python pump_control.py C in forward 500     # Run pump C_in forward at 500 steps/s
    python pump_control.py D out forward 1500   # Run pump D_out forward at 1500 steps/s
"""

import time
import sys
from ticlib import TicUSB
from math import floor
from src.config import Config as cfg

# --- Argument parsing ---
def parse_args():
    if len(sys.argv) != 5:
        print("Usage: python pump_control.py <letter> <pump_type> <direction> <rate>")
        print("  letter: A, B, C, or D")
        print("  pump_type: 'in' or 'out'")
        print("  direction: 'forward' or 'reverse'")
        print("  rate: steps per second (integer)")
        sys.exit(1)
    
    letter = sys.argv[1].upper()
    pump_type = sys.argv[2].lower()
    direction = sys.argv[3].lower()
    rate = int(sys.argv[4])
    
    # Validate inputs
    if letter not in ['A', 'B', 'C', 'D']:
        print(f"Error: Letter must be A, B, C, or D. Got: {letter}")
        sys.exit(1)
    
    if pump_type not in ['in', 'out']:
        print(f"Error: Pump type must be 'in' or 'out'. Got: {pump_type}")
        sys.exit(1)
    
    if direction not in ['forward', 'reverse']:
        print(f"Error: Direction must be 'forward' or 'reverse'. Got: {direction}")
        sys.exit(1)
    
    if rate <= 0:
        print(f"Error: Rate must be positive. Got: {rate}")
        sys.exit(1)
    
    return letter, pump_type, direction, rate

# --- Get pump serial ---
def get_pump_serial(letter, pump_type):
    """Get the serial number for the specified pump."""
    pump_key = f'{letter}_{pump_type}'
    if pump_key not in cfg.PUMPS:
        raise ValueError(f"Pump {pump_key} not found in configuration")
    return cfg.PUMPS[pump_key]['serial']

# --- Pump setup ---
def setup_pump(serial):
    """Initialize and configure a pump."""
    pump = TicUSB(serial_number=serial)
    pump.energize()
    pump.exit_safe_start()
    pump.set_step_mode(3)  # Step mode (as in calibration)
    pump.set_current_limit(32)
    return pump

# --- Main pump control function ---
def run_pump(letter, pump_type, direction, rate_steps_per_sec, duration_minutes=None, stop_event=None):
    """
    Run a specific pump at a given rate and direction.
    
    Args:
        letter: Pump letter (A, B, C, D)
        pump_type: 'in' or 'out'
        direction: 'forward' or 'reverse'
        rate_steps_per_sec: Pump rate in steps per second
        duration_minutes: Duration to run in minutes (None for indefinite)
        stop_event: Threading.Event to signal stop (alternative to duration)
    
    Returns:
        TicUSB: The pump object that was initialized
    """
    # Get pump serial
    pump_serial = get_pump_serial(letter, pump_type)
    
    # Calculate velocity (pulses/sec)
    steps_per_pulse = 0.5 ** 3  # Step mode 3
    velocity = int(floor(rate_steps_per_sec / steps_per_pulse))
    
    # Set velocity sign based on direction
    # Note: The actual direction depends on the pump's configured direction in cfg.PUMPS
    # and the velocity sign. We'll use positive velocity for now and let the config handle direction.
    if direction == 'reverse':
        velocity = -velocity
    
    # Initialize pump
    try:
        pump = setup_pump(pump_serial)
        print(f"Pump {letter}_{pump_type} (serial {pump_serial}) initialized.")
    except Exception as e:
        raise RuntimeError(f"Error initializing pump {letter}_{pump_type}: {e}")
    
    print(f"Running pump {letter}_{pump_type} {direction} at {rate_steps_per_sec} steps/sec (velocity={velocity})")
    
    # Run pump
    start_time = time.time()
    try:
        while True:
            # Check if we should stop
            if duration_minutes and (time.time() - start_time) > (duration_minutes * 60):
                break
            if stop_event and stop_event.is_set():
                break
                
            try:
                pump.set_target_velocity(velocity)
            except Exception as e:
                print(f"Error setting velocity for pump: {e}")
                break
                
            time.sleep(0.05)  # 50ms loop
            
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping pump.")
    finally:
        try:
            pump.set_target_velocity(0)
            pump.deenergize()
            pump.enter_safe_start()
            print(f"Pump {letter}_{pump_type} stopped and deenergized.")
        except Exception as e:
            print(f"Error stopping pump: {e}")
    
    return pump

# --- Command line interface ---
if __name__ == "__main__":
    letter, pump_type, direction, rate = parse_args()
    
    try:
        run_pump(letter, pump_type, direction, rate)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 