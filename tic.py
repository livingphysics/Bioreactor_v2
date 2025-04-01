from ticlib import TicUSB
from time import sleep

tic = TicUSB()
 
tic.halt_and_set_position(0)
tic.energize()
tic.exit_safe_start()
print(tic.get_step_mode())
positions = [30, 60, 90, 120, 0]

for position in positions:
		tic.set_target_position(position)
		while tic.get_current_position() != tic.get_target_position():
			sleep(0.1)
tic.deenergize()
tic.enter_safe_start()
