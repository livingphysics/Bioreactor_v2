# File: pwm_motor.py

import RPi.GPIO as GPIO
import time

class PWM_Motor:
    def __init__(self, pwm_pin: int, dir_pin: int, freq: int):
        self.pwm_pin = pwm_pin
        self.dir_pin = dir_pin
        self.freq = freq
        
        GPIO.setup(pwm_pin, GPIO.OUT)
        GPIO.setup(dir_pin, GPIO.OUT)
        
        self.pwm = GPIO.PWM(pwm_pin, freq)
        self.pwm.start(0)
        
    def run(self, power: int, forward: bool):
        """Basic forward & backward motor functionality"""
        GPIO.output(self.dir_pin, GPIO.HIGH if forward else GPIO.LOW)
        
        # power arg is now 0-100 (duty cycle percentage)
        self.pwm.ChangeDutyCycle(power)
        
    def linear(self, duration: float, start_power: float, end_power: float, forward: bool):
        """Linear increase in motor output over specified time duration"""
        if not forward and end_power > start_power:
            raise ValueError("End power cannot be greater than start power when moving backward")
            
        dt = 0.01
        steps = int(duration / dt)
        
        self.run(start_power, forward)
        
        for i in range(steps):
            power = start_power + (i / steps) * (end_power - start_power)
            self.pwm.ChangeDutyCycle(power)
            time.sleep(dt)
        
    def stop(self):
        self.pwm.ChangeDutyCycle(0)
        
    def cleanup(self):
        self.pwm.stop()

def initialize_gpio():
    GPIO.setmode(GPIO.BCM)
    # Add any other global GPIO setup here if needed

def cleanup_gpio():
    GPIO.cleanup()

# Example usage
if __name__ == "__main__":
    try:
        initialize_gpio()
        peltier = PWM_Motor(24, 25, 1000)
        peltier.run(100, forward=True)  # Run at 50% power, forward direction
        time.sleep(5)  # Run for 5 seconds
        peltier.run(100, forward=False)  # Run at 50% power, forward direction
        time.sleep(5)  # Run for 5 seconds
        peltier.stop()
    finally:
        peltier.cleanup()
        cleanup_gpio()
