import RPi.GPIO as IO
import time

IO.setmode(IO.BOARD)
LED_Pins = [21, 29, 31, 33]
for LED_Pin in LED_Pins:
	IO.setup(LED_Pin,IO.OUT)

try:
	while True:
		for LED_Pin in LED_Pins:
			IO.output(LED_Pin,1)
		time.sleep(3)
		for LED_Pin in LED_Pins:
			IO.output(LED_Pin,0)
		time.sleep(3)
		
except KeyboardInterrupt:
	for LED_Pin in LED_Pins:
		IO.output(LED_Pin,0)
	IO.cleanup()
	pass
	
