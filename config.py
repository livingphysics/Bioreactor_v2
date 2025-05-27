class Config:
    # GPIO pins
    PELTIER_PWM_PIN = 24
    PELTIER_DIR_PIN = 25
    PELTIER_PWM_FREQ = 1000

    # BCM Pin Mapping
    BCM_MAP: dict[int, int] = {
        7: 4, 11: 17, 12: 18, 13: 27, 15: 22, 16: 23, 18: 24, 22: 25,
        29: 5, 31: 6, 32: 12, 33: 13, 35: 19, 36: 16, 37: 26, 38: 20, 40: 21
    }
    
    # LED Configuration
    LED_PIN: int = 37
    LED_MODE: str = 'bcm'

    # Stirrer Configuration
    STIRRER_PIN: int = 35
    STIRRER_SPEED: int = 1000
    DUTY_CYCLE: int = 15

    # Ring Light Configuration
    RING_LIGHT_COUNT: int = 32
    RING_LIGHT_BRIGHTNESS: float = 0.2

    # ADC Configuration
    # TODO: Have 8 different lists: ADC_1_REF, ADC_1_135, ADC_1_180, ADC_1_IO_TEMP, ADC_2_REF, ADC_2_135, ADC_2_180, ADC_2_IO_TEMP
    ADC_1_ADDRESS: int = 0x48
    ADC_1_REF_VOLTAGE: float = 4.2
    ADC_1_PHOTODIODE_CHANNELS: list[int] = [0, 1, 2, 3, 4, 5, 6, 7]
    ADC_1_IO_TEMP_CHANNELS: list[int] = []
    ADC_2_ADDRESS: int = 0x49
    ADC_2_REF_VOLTAGE: float = 4.2
    ADC_2_PHOTODIODE_CHANNELS: list[int] = [0, 1, 2, 3]
    ADC_2_IO_TEMP_CHANNELS: list[int] = [4, 5]

    # Temperature Sensor Arrays
    VIAL_TEMP_SENSOR_ORDER: list[int] = [2, 0, 3, 1]

    # Logging Configuration
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(levelname)s - %(message)s'

    # Pump Configuration
    PUMPS: dict[str, dict[str, Union[str, float]]] = {
        'tube_1_in':  {'port': '/dev/ttyACM0', 'gradient': 0.002, 'intercept': 0.0},
        'tube_1_out': {'port': '/dev/ttyACM1', 'gradient': 0.002, 'intercept': 0.0},
        'tube_2_in':  {'port': '/dev/ttyACM2', 'gradient': 0.002, 'intercept': 0.0},
        'tube_2_out': {'port': '/dev/ttyACM3', 'gradient': 0.002, 'intercept': 0.0},
        'tube_3_in':  {'port': '/dev/ttyACM4', 'gradient': 0.002, 'intercept': 0.0},
        'tube_3_out': {'port': '/dev/ttyACM5', 'gradient': 0.002, 'intercept': 0.0},
        'tube_4_in':  {'port': '/dev/ttyACM6', 'gradient': 0.002, 'intercept': 0.0},
        'tube_4_out': {'port': '/dev/ttyACM7', 'gradient': 0.002, 'intercept': 0.0},
    }
