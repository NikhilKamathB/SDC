# This script provides a free run of vehicle in an environment 
# defined in *.json file. Enter the listing of various sensors
# and its positioning in the environment, the town in which the
# simulation must run, spawing of objects and other condtions in
# the *.json file.
# Reference - https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/manual_control.py

"""
Json file body.

host (str)              : "127.0.0.1"   # required
port (int)              : 2000   # required
worker_threads (int)    : 0   # required
network_timeout (float) : 10.0   # required
width (int)             : 1280   # required
height (int)            : 720   # required
role_name(str)          : "hero"   # required
vehicle_filter (str)    : "vehicle.*"   # required
autopilot (int)         : 0   # required
gamma (float)           : 2.2   # required



Control commands.

    w            : throttle
    s            : brake
    a/d          : steer left/right
    q            : toggle reverse
    Space        : hand-brake
    h            : help
    ESC          : quit
"""


# Finding the Carla module.

import os
import sys
import glob
from typing import List

from pygame import key, surface
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except Exception as e:
    print("An error occurred! This may be due to errors apropos to Carla egg file setup.")


# Importing necessary modules.

import re
import math
import carla
import random
import pygame
from datetime import datetime
from pygame.locals import *
from .utils import *


class HelpText:

    def __init__(self, font=None, width=None, height=None) -> None:
        self.font = font
        lines = __doc__.split('\n')
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
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

    def render(self, display=None) -> None:
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
        self.help = HelpText(font=pygame.font.Font(mono, 16), width=self.width, height=self.height)
        self.server_fps, self.frame, self.simulation_time = 0, 0, 0
        self._show_info = True        
        self._info_text = []
        self._server_clock = pygame.time.Clock()
    
    def on_world_tick(self, timestamp=None) -> None:
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds
    
    def get_actor_display_name(self, player=None, truncate=20) -> str:
        name = ' '.join(player.type_id.replace('_', '.').title().split('.')[1:])
        return (name[: truncate-1] + u'\u2026') if len(name) > truncate else name
    
    def toggle_info(self):
        self._show_info = not self._show_info
    
    def tick(self, world=None, clock=None) -> None:
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        # compass = world.imu_sensor.compass
        # heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        # heading += 'S' if 90.5 < compass < 269.5 else ''
        # heading += 'E' if 0.5 < compass < 179.5 else ''
        # heading += 'W' if 180.5 < compass < 359.5 else ''
        # colhist = world.collision_sensor.get_collision_history()
        # collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        # max_col = max(1.0, max(collision))
        # collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % self.get_actor_display_name(player=world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            # u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            # 'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            # 'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            # 'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            ''
        ]
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)
            ]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)
            ]
        self._info_text += [
            '',
            # 'Collsion:',
            # collision,
            # '',
            'Number of vehicles: % 8d' % len(vehicles)
        ]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x-t.location.x)**2 + (l.y-t.location.y)**2 + (l.z-t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 200.0:
                    break
                vehicle_type = self.get_actor_display_name(player=vehicle, truncate=20)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def render(self, display=None) -> None:
        info_surface = pygame.Surface((220, self.dim[1]))
        info_surface.set_alpha(100)
        display.blit(info_surface, (0, 0))
        v_offset, bar_h_offset, bar_width = 4, 100, 106
        for item in self._info_text:
            if v_offset + 18 > self.dim[1]:
                break
            if isinstance(item, list):
                if len(item) > 1:
                    points = [(x+8, v_offset+8+(1.0-y)*30) for x, y in enumerate(item)]
                    pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                item = None
                v_offset += 18
            elif isinstance(item, tuple):
                if isinstance(item[1], bool):
                    rect = pygame.Rect((bar_h_offset, v_offset+8), (6, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                else:
                    rect_border = pygame.Rect((bar_h_offset, v_offset+8), (bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                    f = (item[1]-item[2]) / (item[3]-item[2])
                    if item[2] < 0.0:
                        rect = pygame.Rect((bar_h_offset+f*(bar_width-6), v_offset+8), (6, 6))
                    else:
                        rect = pygame.Rect((bar_h_offset, v_offset+8), (f*bar_width, 6))
                    pygame.draw.rect(display, (255, 255, 255), rect)
                item = item[0]
            if item:
                surface = self._font_mono.render(item, True, (255, 255, 255))
                display.blit(surface, (8, v_offset))
            v_offset += 18
        self.help.render(display)
    

class World:

    def __init__(self, world=None, hud=None, args=None) -> None:
        self.world, self.hud, self.args = world, hud, args
        self.rolename = self.args['role_name']
        try:
            self.map = self.world.get_map()
        except RuntimeError as e:
            print(f"Runtime error occurred: {e}")
            sys.exit(1)
        self.player, self.camera_manager = None, None
        self._weather_presets = self.find_weather_presets()
        self._weather_index = 0
        self._actor_filter = self.args['vehicle_filter']
        self._gamma = self.args['gamma']
        self.recording_enabled = False
        self.restart()
        self.world.on_tick(self.hud.on_world_tick)

    def find_weather_presets(self) -> List:
        rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
        name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
        presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
        return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]
    
    def tick(self, clock=None) -> None:
        self.hud.tick(self, clock)

    def render(self, display=None) -> None:
        self.hud.render(display=display)
    
    def restart(self) -> None:
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
        # TODO : Setup sensors.


class KeyboardControl:

    def __init__(self, world=None, autopilot=None) -> None:
        self._autopilot_enabled = autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
    
    def parse_events(self, client, world, clock) -> bool:
        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
    
    def _is_quit_shortcut(key=None) -> bool:
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


class Run():

    '''
    This class enables the vehicle's free drive and data capture.
    '''

    def __init__(self, args=None) -> None:
        assert args is not None; "Args argument must be provided."
        self.json_path = args.configuration
        self.output_dir = args.output_dir
    
    def game_loop(self, args=None) -> None:
        # Initialize pygame and fonts.
        pygame.init()
        pygame.font.init()
        world = None
        try:
            # Initialize client, hud, world and controller and run game.
            client = carla.Client(host=args['host'], port=args['port'], worker_threads=args['worker_threads'])
            client.set_timeout(seconds=args['network_timeout'])
            display = pygame.display.set_mode(size=(args['width'], args['height']), flags=pygame.HWSURFACE | pygame.DOUBLEBUF)
            hud = HUD(width=args['width'], height=args['height'])
            world = World(world=client.get_world(), hud=hud, args=args)
            controller = KeyboardControl(world=world, autopilot=args['autopilot'])
            clock = pygame.time.Clock()
            while True:
                clock.tick_busy_loop(60)
                if controller.parse_events(client=client, world=world, clock=clock):
                    return
                world.tick(clock=clock)
                world.render(display)
                pygame.display.flip()
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