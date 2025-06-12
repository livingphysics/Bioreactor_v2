from PWM_Motor import PWM_Motor, initialize_gpio, cleanup_gpio
import time

try:
        initialize_gpio()
        peltier = PWM_Motor(24, 25, 3000)
        for i in [100]:
                peltier.run(i, forward=True)
                time.sleep(600)
        # peltier.run(50, forward=True)  # Run at 50% power, forward direction
        # time.sleep(5)  # Run for 5 seconds
        # peltier.run(50, forward=False)  # Run at 50% power, forward direction
        # time.sleep(5)  # Run for 5 seconds
        peltier.stop()
finally:
        peltier.cleanup()
        cleanup_gpio()
