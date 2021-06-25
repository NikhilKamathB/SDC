# This script provides a free run of vehicle in an environment 
# defined in *.json file. Enter the listing of various sensors
# and its positioning in the environment, the town in which the
# simulation must run, spawing of objects and other condtions in
# the *.json file.
# Reference - https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/manual_control.py

"""
Json file body.

host (str)       : '127.0.0.1'
port (int)       : 2000
resolution (str) : '1280x720'
"""

"""
Control commands.

    W            : throttle
    S            : brake
    A/D          : steer left/right
    Q            : toggle reverse
    Space        : hand-brake
    ESC          : quit
"""

# Finding the Carla module.

import os
import sys
import glob
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except Exception as e:
    print("An error occurred! This may be due to errors apropos to Carla egg file setup.")


# Importing necessary modules.

import carla
import pygame
from pygame.locals import *
from .utils import *


class Run():

    def __init__(self, json_path=None, output_dir=None) -> None:
        assert json_path is not None or output_dir is not None; "Path to the configuration file / output directory must be provided."
        self.json_path = json_path
        self.output_dir = output_dir
    
    def game_loop(self, args=None) -> None:
        pass

    def main(self) -> None:
        self.args = config_validator(json_path=self.json_path)
        try:
            self.game_loop(args=self.args)
        except KeyboardInterrupt:
            print("\nOperation Cancelled. Keyboard interrupt!")