# This script provides a free run of vehicle in an environment 
# defined in *.json file.
# Reference - https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/manual_control.py

"""
Control commands.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    ESC          : quit
"""


from .utils import *


class Run():

    def __init__(self) -> None:
        config_validator()