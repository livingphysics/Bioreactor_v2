import serial
import re

SERIAL_PORT_SCALE = "/dev/serial/by-id/usb-Prolific_Technology_Inc._USB-Serial_Controller_BBAHb135B02-if00-port0"

def try_baudrates(port, candidates):
    for rate in candidates:
        ser = serial.Serial(port, baudrate=rate, timeout=0.5)
        ser.write(b's')
        resp = ser.read(18).decode('ascii', errors='ignore')
        ser.close()
        if re.search(r'\d+\.\d+', resp):
            print(f"Detected working baudrate: {rate}")
            return rate
    raise RuntimeError("No valid baudrate found")

BAUDRATE = try_baudrates(SERIAL_PORT_SCALE,
    [300, 1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200]
)

print(BAUDRATE)
