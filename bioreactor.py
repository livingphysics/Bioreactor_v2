import u3
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PWM_Motor import PWM_Motor, initialize_gpio, cleanup_gpio
import time
from adafruit_ina219 import INA219
import board
from config import Config as cfg
import RPi.GPIO as IO
import logging

from typing import List, Tuple, Optional, Union
import busio
import adafruit_ads7830.ads7830 as ADC
import numpy as np
from ds18b20 import DS18B20
from contextlib import contextmanager
import neopixel
from ticlib import TicUSB

logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL),
    format=cfg.LOG_FORMAT
)


class Bioreactor():
    """Class to manage all sensors and operations for the bioreactor"""
    
    def __init__(self) -> None:
        """Initialize all sensors and store them as instance attributes"""
        try:
            self.init_i2c()
            self.init_leds()
            self.init_stirrer()
            self.init_ring_light()
            self.init_optical_density()
            self.init_temp()
            self.init_peltier()
            self.init_pumps()
        except OSError as e:
            logging.error(f"Hardware initialization error: {e}")
            raise
        except Exception as e:
            logging.error(f"Some (probably non-hardware) error during initialization: {e}")
            raise

    def init_i2c(self) -> None:
        """Initialize the I2C bus"""
        self.i2c = busio.I2C(board.SCL, board.SDA)
    
    def init_leds(self) -> None:
        """Initialize the LEDs"""
        # Board mode must be the same for leds and peltier
        self.board_mode = cfg.LED_MODE.upper()
        self.led_pin = cfg.LED_PIN
        if self.board_mode == 'BOARD':
            IO.setmode(IO.BOARD)
        elif self.board_mode == 'BCM':
            self.led_pin = cfg.BCM_MAP[self.led_pin]
            IO.setmode(IO.BCM)
        else:
            raise ValueError("Invalid board mode: use 'BCM' or 'BOARD'")
        IO.setup(self.led_pin, IO.OUT)
        IO.output(self.led_pin, 0)
    
    def init_stirrer(self) -> None:
        """Initialize the stirrer"""
        IO.setup(cfg.STIRRER_PIN, IO.OUT)
        self.stirrer = IO.PWM(cfg.STIRRER_PIN, cfg.STIRRER_SPEED)
        self.stirrer.start(0)
        self.stirrer.ChangeDutyCycle(cfg.DUTY_CYCLE)
    
    def init_ring_light(self) -> None:
        """Initialize the ring light"""
        self.ring_light = neopixel.NeoPixel(board.D10, cfg.RING_LIGHT_COUNT, brightness=cfg.RING_LIGHT_BRIGHTNESS, auto_write=False)
        self.change_ring_light((0,0,0))
    
    def init_optical_density(self) -> None:
        """Initialize the optical density sensors and input/output temp sensors"""
        # ADS7830 setup
        self.adc_1: ADC.ADS7830 = ADC.ADS7830(self.i2c, address=cfg.ADC_1_ADDRESS)
        self.REF_1: float = cfg.ADC_1_REF_VOLTAGE
        self.adc_2: ADC.ADS7830 = ADC.ADS7830(self.i2c, address=cfg.ADC_2_ADDRESS)
        self.REF_2: float = cfg.ADC_2_REF_VOLTAGE
    
    def init_temp(self) -> None:
        """Initialize the external temperature sensors"""
        self.vial_temp_sensors: np.ndarray = np.array(DS18B20.get_all_sensors())[cfg.VIAL_TEMP_SENSOR_ORDER]
    
    def init_peltier(self) -> None:
        """Initialize the peltier"""
        self.peltier_curr_sensor = INA219(self.i2c)

        IO.setup(cfg.PELTIER_PWM_PIN, IO.OUT)
        IO.setup(cfg.PELTIER_DIR_PIN, IO.OUT)
        
        self.pwm = IO.PWM(cfg.PELTIER_PWM_PIN, cfg.PELTIER_PWM_FREQ)
        self.pwm.start(0)
    
    def init_pumps(self) -> None:
        """
        Initialize TicUSB controllers for each pump defined in `Config.PUMPS`.
        Store calibration (gradient, intercept) for conversion.
        """
        self.pumps: dict[str, TicUSB] = {}
        self.calibration: dict[str, dict[str, float]] = {}

        for name, settings in cfg.PUMPS.items():
            port = settings['port']
            tic = TicUSB(port=port)
            tic.energize()
            tic.exit_safe_start()
            self.pumps[name] = tic
            # Store gradient & intercept for ml/sec = gradient*steps/sec + intercept
            self.calibration[name] = {
                'gradient': settings['gradient'],
                'intercept': settings['intercept']
            }

    def led_on(self) -> None:
        """Turn on the LED"""
        IO.output(self.led_pin, 1)

    def led_off(self) -> None:
        """Turn off the LED"""
        IO.output(self.led_pin, 0)
    
    def change_ring_light(self, color: Tuple[int, int, int], pixel: Optional[int] = None) -> None:
        """Change the color of the ring light"""
        if pixel is None:
            self.ring_light.fill(color)
        else:
            self.ring_light[pixel] = color
        self.ring_light.show()
    
    def change_peltier(self, power: int, forward: bool):
        """Change the peltier power and direction"""
        IO.output(cfg.PELTIER_DIR_PIN, IO.HIGH if forward else IO.LOW)
        
        # power arg is now 0-100 (duty cycle percentage)
        self.pwm.ChangeDutyCycle(power)

    def change_pumps_rate(self, pump_name: str, ml_per_sec: float) -> None:
        """
        Set a pump to achieve a target volumetric flow (ml/sec).

        Uses:
            ml_per_sec = gradient * (steps_per_sec) + intercept
        => steps_per_sec = (ml_per_sec - intercept) / gradient

        Args:
            pump_name: key in `self.pumps` (e.g. 'tube_1_in')
            ml_per_sec: Desired volumetric rate in ml/sec (>= 0)
        """
        if pump_name not in self.pumps:
            raise ValueError(f"No pump named '{pump_name}' configured")

        cal = self.calibration[pump_name]
        # Compute steps/sec from ml/sec
        steps_per_sec = int((ml_per_sec - cal['intercept']) / cal['gradient'])
        # Determine direction: inlet pumps forward, outlet pumps reverse
        forward = pump_name.endswith('_in')
        velocity = steps_per_sec if forward else -steps_per_sec

        try:
            self.pumps[pump_name].set_target_velocity(velocity)
        except Exception as e:
            logging.error(f"Error setting velocity for '{pump_name}': {e}")
            raise

    def finish(self) -> None:
        """Clean up resources"""
        IO.output(self.led_pin, 0)
        self.stirrer.stop(0)
        self.change_ring_light((0,0,0))
        self.pwm.ChangeDutyCycle(0)
        IO.cleanup()
        if hasattr(self, 'pump'):
            self.pump.deenergize()
            self.pump.enter_safe_start()

    def get_photodiodes(self) -> List[float]:
        """Get the photodiodes readings"""
        try:
            return [self.adc_1.read(i) * self.REF_1 / 65535.0 for i in cfg.ADC_1_PHOTODIODE_CHANNELS] + [self.adc_2.read(i) * self.REF_2 / 65535.0 for i in cfg.ADC_2_PHOTODIODE_CHANNELS]
        except OSError as e:
            logging.error(f"Hardware error reading photodiodes: {e}")
            return [float('nan')] * 12
        except Exception as e:
            logging.error(f"Unexpected error reading photodiodes: {e}")
            return [float('nan')] * 12
    
    def get_io_temp(self) -> List[float]:
        """Get the input/output temperature readings"""
        try:
            return [self.adc_1.read(i) * self.REF_1 / 65535.0 for i in cfg.ADC_1_IO_TEMP_CHANNELS] + [self.adc_2.read(i) * self.REF_2 / 65535.0 for i in cfg.ADC_2_IO_TEMP_CHANNELS]
        except OSError as e:
            logging.error(f"Hardware error reading input/output temperature: {e}")
            return [float('nan')] * 2
        except Exception as e:
            logging.error(f"Unexpected error reading input/output temperature: {e}")
            return [float('nan')] * 2
    
    def get_vial_temp(self) -> List[float]:
        """Get the temperature readings"""
        try:
            return [vial_temp_sensor.get_temperature() for vial_temp_sensor in self.vial_temp_sensors]
        except OSError as e:
            logging.error(f"Hardware error reading temperature: {e}")
            return [float('nan')] * len(self.vial_temp_sensors)
        except Exception as e:
            logging.error(f"Unexpected error reading temperature: {e}")
            return [float('nan')] * len(self.vial_temp_sensors)
    
    def get_peltier_curr(self) -> float:
        """Get the current reading"""
        try:
            return self.peltier_curr_sensor.current / 1000
        except:
            return float('nan')
    
    def balanced_flow(self, pump_name: str, ml_per_sec: float) -> None:
        """
        For a given pump, set its flow and automatically set the converse pump
        to the same volumetric rate in the opposite direction.

        E.g. if pump_name is 'tube_1_in', the converse is 'tube_1_out'.

        Args:
            pump_name: e.g. 'tube_1_in' or 'tube_1_out'
            ml_per_sec: Desired flow rate in ml/sec (>= 0)
        """
        # Find converse pump by swapping suffix
        if pump_name.endswith('_in'):
            converse = pump_name[:-3] + 'out'
        elif pump_name.endswith('_out'):
            converse = pump_name[:-4] + 'in'
        else:
            raise ValueError("Pump name must end with '_in' or '_out'")

        # Set both pumps
        self.change_pumps_rate(pump_name, ml_per_sec)
        self.change_pumps_rate(converse, ml_per_sec)
    
    def pid_temp_controller(
        self,
        setpoint: float,
        current_temp: Optional[float] = None,
        kp: float = 10.0,
        ki: float = 1.0,
        kd: float = 0.0,
        dt: float = 1.0
    ) -> None:
        """
        PID loop to maintain reactor temperature at `setpoint`.

        Args:
            setpoint: Desired temperature (°C)
            current_temp: Measured temp (°C). If None, reads from first vial sensor.
            kp, ki, kd: PID gains (defaults from temperatureController.py).
            dt: Time elapsed since last call (s, default 1s).
        """
        # Read temperature if not provided
        if current_temp is None:
            temps = self.get_vial_temp()
            current_temp = temps[0]

        # PID calculations
        error = setpoint - current_temp
        self._temp_integral += error * dt
        derivative = (error - self._temp_last_error) / dt if dt > 0 else 0.0
        output = kp * error + ki * self._temp_integral + kd * derivative

        # Clamp output to [0,100] for PWM duty cycle
        duty = max(0, min(100, int(abs(output))))
        # Determine direction: True=heat (forward), False=cool (reverse)
        forward = output >= 0
        self.change_peltier(duty, forward)

        # Store for next iteration
        self._temp_last_error = error

    def chemostat_mode(
        self,
        pump_name: str,
        flow_rate_ml_s: float,
        temp_setpoint: float,
        kp: float = 10.0,
        ki: float = 1.0,
        kd: float = 0.0,
        dt: float = 1.0
    ) -> None:
        """
        Run the reactor in chemostat mode:
        - Balanced flow on the specified pump.
        - PID temperature control (defaults from temperatureController.py).

        Args:
            pump_name: e.g. 'tube_1_in' or 'tube_1_out'
            flow_rate_ml_s: Inflow/outflow rate (ml/sec)
            temp_setpoint: Desired temperature (°C)
            kp, ki, kd: PID gains (defaults 10,1,0)
            dt: Time step for PID loop (s, default 1s).
        """
        # Maintain flow
        self.balanced_flow(pump_name, flow_rate_ml_s)
        # Maintain temperature
        self.pid_temp_controller(
            setpoint=temp_setpoint,
            kp=kp,
            ki=ki,
            kd=kd,
            dt=dt
        )

    @contextmanager
    def led_context(self, settle_time: float = 1.0):
        """Context manager for LED control"""
        try:
            # Turn IR LEDs on and wait for signal to settle
            self.led_on()
            time.sleep(settle_time)
            yield
        finally:
            # Turn IR LEDs off
            self.led_off()
    
    def __enter__(self):
        """Enter the context manager"""
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager"""
        self.finish()
        return False
