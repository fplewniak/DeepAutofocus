from pycromanager import Core

core = Core()

print(core)

z_stage = core.get_focus_device()
print(z_stage)
adapter_name = core.get_device_library(z_stage)
print(adapter_name)

core.set_position(0.0)
print(core.get_position())
core.set_position(1.2)
print(core.get_position())
core.wait_for_device(z_stage)
print(core.get_position())
