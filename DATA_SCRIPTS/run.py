# This script provides a free run of vehicle in an environment 
# defined in *.json file. Enter the listing of various sensors
# and its positioning in the environment, the town in which the
# simulation must run, spawing of objects and other condtions in
# the *.json file.
# Reference - https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/manual_control.py

"""
Json file body.

host (str)              : '127.0.0.1'   # required
port (int)              : 2000   # required
worker_threads (int)    : 0   # required
network_timeout (float) : 10.0   # required
width (int)             : 1280   # required
height (int)            : 720   # required
rolename(str)           : 'hero'   # required
vehicle_filter (str)    : 'vehicle.*'   # required
gamma (float)           : 2.2   # required



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
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except Exception as e:
    print("An error occurred! This may be due to errors apropos to Carla egg file setup.")


# Importing necessary modules.

import re
import carla
import random
import pygame
from pygame.locals import *
from .utils import *

class HelpText:

    def __init__(self, font, width, height) -> None:
        self.font = font
        lines = __doc__.split('\n')
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.line_space = 18
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n*self.line_space))
            self._render = False
        self.surface.set_alpha(220)
    
    def toggle(self) -> None:
        self._render = not self._render

    def render(self, display) -> None:
        if self._render:
            display.blit(self.surface, self.pos)


class HUD:

    def __init__(self, width=None, height=None) -> None:
        self.width, self.height = width, height
        self.dim = (self.width, self.height)
        font_name = "courier" if os.name=="nt'"else "mono"
        fonts = [f for f in pygame.font.get_fonts() if font_name in f]
        mono = "ubuntumono" if "ubuntumono" in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        # font = pygame.font.Font(pygame.font.get_default_font(), 20)
        self._font_mono = pygame.font.Font(mono, 12 if os.name=="nt" else 14)
        self.help_text = HelpText(font=pygame.font.Font(mono, 16), width=self.width, height=self.height)
        self.server_fps, self.frame, self.simulation_time = 0, 0, 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
    
    def on_world_tick(self, timestamp) -> None:
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds


class World:

    def __init__(self, world=None, hud=None, args=None) -> None:
        self.world, self.hud, self.args = world, hud, args
        self.rolename = self.args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as e:
            print(f"Runtime error occurred: {e}")
            sys.exit(1)
        self.player, self.camera_manager = None, None
        self._weather_presets = self.find_weather_presets()
        self._weather_index = 0
        self._actor_filter = self.args.vehicle_filter
        self._gamma = self.args.gamma
        self.recording_enabled = False
        self.restart()
        self.world.on_tick(self.hud.on_world_tick)

    def find_weather_presets(self):
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]
    
    def restart(self):
        # Define speed intities if necessary.
        self.player_max_speed, self.player_max_speed_fast = 1.589, 3.713
        # Get a blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        # Set the blueprint's attribute.
        blueprint.set_attribute("role_name", self.rolename)
        if blueprint.has_attribute("color"):
            color = random.choice(blueprint.get_attribute("color").recommended_values)
            blueprint.set_attribute("color",color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_tranform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll, spawn_point.rotation.ptich = 0.0, 0.0
            self.destroy()
        while self.player is None:
            if not self.map.get_spawn_points():
                print("There are not spawn points available in the map.")
                sys.exit(1)
            spawn_point = random.choice(self.map.get_spawn_points())
        self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Setup sensors.


class Run():

    '''
    This class enables the vehicle's free drive and data capture.
    '''

    def __init__(self, json_path=None, output_dir=None) -> None:
        assert json_path is not None or output_dir is not None; "Path to the configuration file / output directory must be provided."
        self.json_path = json_path
        self.output_dir = output_dir
    
    def game_loop(self, args=None) -> None:
        # Initialize pygame and fonts.
        pygame.init()
        pygame.font.init()
        world = None
        try:
            # Initialize client, hud, world and controller and run game.
            client = carla.Client(host=args.host, port=args.port, worker_threads=args.worker_threads)
            client.set_timeout(seconds=args.network_timeout)
            display = pygame.display.set_mode(size=(args.width, args.height), flags=pygame.HWSURFACE | pygame.DOUBLEBUF)
            hud = HUD(width=args.width, height=args.height)
            world = World(world=client.get_world(), hud=hud, args=args)
            # controller = # TODO
        except Exception as e :
            print("An Exception occurred!")
            print(e)
        finally:
            # Destroy the world and quit.
            if world is not None:
                world.destroy()
            pygame.quit()

    def main(self) -> None:
        self.args = config_validator(json_path=self.json_path)
        try:
            self.game_loop(args=self.args)
        except KeyboardInterrupt:
            print("\nOperation Cancelled. Keyboard interrupt!")