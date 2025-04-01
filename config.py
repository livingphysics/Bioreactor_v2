class Config:
    # GPIO pins
    PELTIER_PIN1 = 24
    PELTIER_PIN2 = 25
    PELTIER_PWM_FREQ = 1000

    # Temperature sensor configuration
    TEMP_SENSOR_INPUTS = [0, 1, 2, 3]
    TEMP_SENSOR_GAINS = [3.15, 3.15, 1, 1]
