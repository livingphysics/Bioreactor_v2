from ticlib import TicUSB
import time
serials = ["00473498"]

tic_list = [TicUSB(serial_number=serial) for serial in serials]
print((tic_list))
for tic in tic_list:
	tic.halt_and_set_position(0)
	tic.energize()
	tic.exit_safe_start()
	tic.set_step_mode(3)
	tic.set_current_limit(32)
	print(tic.get_step_mode())

print(f"{tic.get_planning_mode()=}")

t_start = time.time()
duration = 10
vel = 1000000
print(type(vel))
while time.time() - t_start < duration:
	for tic in tic_list:
		tic.set_target_velocity(vel)
		print(f"{tic.get_planning_mode()=}")

for tic in tic_list:
	tic.set_target_velocity(0)				
	tic.deenergize()
	tic.enter_safe_start()
