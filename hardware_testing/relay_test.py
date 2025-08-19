import time
import logging
from src.bioreactor import Bioreactor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_relays():
    """Test the relay functionality of the bioreactor."""
    
    print("Testing relay functionality...")
    print("This will cycle through all relays, turning them on and off.")
    print("Press Ctrl+C to stop the test.")
    
    try:
        with Bioreactor() as bioreactor:
            # Check if relays are initialized
            if not bioreactor._initialized.get('relays'):
                print("ERROR: Relays not initialized!")
                return
            
            print(f"Available relays: {list(bioreactor.relays.keys())}")
            
            # Test individual relay control
            print("\n=== Testing Individual Relay Control ===")
            for relay_name in bioreactor.relays.keys():
                print(f"Turning ON {relay_name}")
                bioreactor.change_relay(relay_name, True)
                time.sleep(1)
                
                state = bioreactor.get_relay_state(relay_name)
                print(f"  {relay_name} state: {'ON' if state else 'OFF'}")
                
                print(f"Turning OFF {relay_name}")
                bioreactor.change_relay(relay_name, False)
                time.sleep(1)
                
                state = bioreactor.get_relay_state(relay_name)
                print(f"  {relay_name} state: {'ON' if state else 'OFF'}")
                print()
            
            # Test all relays simultaneously
            print("=== Testing All Relays Simultaneously ===")
            print("Turning all relays ON")
            bioreactor.change_all_relays(True)
            time.sleep(2)
            
            states = bioreactor.get_all_relay_states()
            print("All relay states:", states)
            
            print("Turning all relays OFF")
            bioreactor.change_all_relays(False)
            time.sleep(2)
            
            states = bioreactor.get_all_relay_states()
            print("All relay states:", states)
            
            # Continuous cycling test
            print("\n=== Continuous Cycling Test ===")
            print("Cycling all relays every 2 seconds...")
            cycle_count = 0
            while cycle_count < 10:  # Run for 10 cycles
                # Turn all relays on
                bioreactor.change_all_relays(True)
                print(f"Cycle {cycle_count + 1}: All relays ON")
                time.sleep(2)
                
                # Turn all relays off
                bioreactor.change_all_relays(False)
                print(f"Cycle {cycle_count + 1}: All relays OFF")
                time.sleep(2)
                
                cycle_count += 1
            
            print("Relay test completed successfully!")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"Error during relay test: {e}")

if __name__ == '__main__':
    test_relays()
