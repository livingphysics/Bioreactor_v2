from typing import Union

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
    PUMPS: dict[str, dict[str, Union[str, float, dict]]] = {
        'A_in':  {
            'serial': '00473498',
            'direction': 'forward',  # user-set: 'forward' or 'reverse'
            'forward': {'gradient': 0.002, 'intercept': 0.0},
        },
        'A_out': {
            'serial': '00473497',
            'direction': 'reverse',
            'forward': {'gradient': 0.002, 'intercept': 0.0},
        },
        'B_in':  {
            'serial': '00473504',
            'direction': 'forward',
            'forward': {'gradient': 0.002, 'intercept': 0.0},
        },
        'B_out': {
            'serial': '00473508',
            'direction': 'reverse',
            'forward': {'gradient': 0.002, 'intercept': 0.0},
        },
        'C_in':  {
            'serial': '00473510',
            'direction': 'forward',
            'forward': {'gradient': 0.002, 'intercept': 0.0},
        },
        'C_out': {
            'serial': '00473517',
            'direction': 'reverse',
            'forward': {'gradient': 0.002, 'intercept': 0.0},
        },
        'D_in':  {
            'serial': '00473491',
            'direction': 'forward',
            'forward': {'gradient': 0.002, 'intercept': 0.0},
        },
        'D_out': {
            'serial': '00473552',
            'direction': 'reverse',
            'forward': {'gradient': 0.002, 'intercept': 0.0},
        },
    }

    # Initialization Components
    INIT_COMPONENTS: dict[str, bool] = {
        'leds': True,
        'pumps': True,
        'ring_light': True,
        'optical_density': True,
        'temp': True,
        'peltier': True,
        'stirrer': True
    }

    LOG_FILE: str = 'bioreactor.log'

    SENSOR_LABELS: dict = {
        'photodiode_1': 'vial_A_180_degree',
        'photodiode_2': 'vial_A_135_degree',
        'photodiode_3': 'vial_B_180_degree',
        'photodiode_4': 'vial_B_135_degree',
        'photodiode_5': 'vial_C_180_degree',
        'photodiode_6': 'vial_C_135_degree',
        'photodiode_7': 'vial_D_180_degree',
        'photodiode_8': 'vial_D_135_degree',
        'photodiode_9': 'vial_A_reference',
        'photodiode_10': 'vial_B_reference',
        'photodiode_11': 'vial_C_reference',
        'photodiode_12': 'vial_D_reference',
        'io_temp_1': 'io_temp_in',
        'io_temp_2': 'io_temp_out',
        'vial_temp_1': 'vial_A_temp',
        'vial_temp_2': 'vial_B_temp',
        'vial_temp_3': 'vial_C_temp',
        'vial_temp_4': 'vial_D_temp',
        'peltier_current': 'peltier_current',
    }
