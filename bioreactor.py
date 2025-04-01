import u3
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PWM_Motor import PWM_Motor, initialize_gpio, cleanup_gpio
import time
from adafruit_ina219 import INA219
import board
from config import Config

class Bioreactor():
    def __init__(self):
        initialize_gpio()
        self.peltier = PWM_Motor(Config.PELTIER_PIN1, Config.PELTIER_PIN2, Config.PELTIER_PWM_FREQ)
        self.i2c_bus = board.I2C()
        self.ina219 = INA219(self.i2c_bus)
        self.inputs = Config.TEMP_SENSOR_INPUTS
        self.gain = Config.TEMP_SENSOR_GAINS
        self.d = u3.U3()
    
    def get_curr(self):
        try:
            return self.ina219.current / 1000
        except:
            return float('nan')
    
    def get_temp(self):
        try:
            return [(self.d.getAIN(sens)/self.gain[i]*100-32)*5/9 for i, sens in enumerate(self.inputs)]
        except:
            return [float('nan')] * len(self.inputs)

