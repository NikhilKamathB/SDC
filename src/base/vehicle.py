##########################################################################################################
# Define a vehicle module here. This module will be used to define the vehicle class and its methods.
##########################################################################################################

import os
import carla
import random
import logging
from enum import Enum
from typing import Union
from src.model.enum import VehicleDoor, VehicleLightState, Gen2VehicleType


logger = logging.getLogger(__name__)


class Vehicle:

    """
    Define a vehicle that would get spawned in the environment.
    """

    __MAX_RETRY__ = 10
    __LOG_PREFIX__ = "Vehicle"

    def __init__(self, 
                 world: carla.World, 
                 blueprint_id: str = Gen2VehicleType.LINCOLN_MKZ_2020.value, 
                 role_name: str = "vehicle", 
                 location: carla.Location = None,
                 rotation: carla.Rotation = None):
        """
        Initialize the vehicle with the blueprint id.
        Input parameters:
            - world: the carla world where the vehicle would be spawned.
            - blueprint_id: the blueprint id of the vehicle.
            - role_name: the role name of the vehicle.
            - location: the location where the vehicle would be spawned.
            - rotation: the rotation of the vehicle.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the vehicle with blueprint id {blueprint_id}")
        self.world = world
        self.blueprint_id = blueprint_id
        self.role_name = role_name
        self.location = location
        self.rotation = rotation
        self._build()
    
    def _set_blueprint(self) -> None:
        """
        Set the blueprint of the vehicle.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Setting the blueprint of the vehicle with blueprint_id {self.blueprint_id}")
        try:
            self.vehicle_bp = self.world.get_blueprint_library().find(self.blueprint_id)
            self.vehicle_bp.set_attribute("role_name", self.role_name)
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while setting the blueprint of the vehicle with blueprint_id {self.blueprint_id}")
            raise e
    
    def _spawn_vehicle(self) -> carla.Vehicle:
        """
        Spawn the vehicle in the environment.
        Output:
            - carla.Vehicle: the spawned vehicle.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Spawning the vehicle in the environment")
        try:
            if self.location is not None:
                if self.rotation is None:
                    self.rotation = carla.Rotation(0, 0, 0)
                transform = carla.Transform(self.location, self.rotation)
                vehicle = self.world.spawn_actor(self.vehicle_bp, transform)
            else:
                tries = 0
                spawn_points = self.world.get_map().get_spawn_points()
                while tries < self.__MAX_RETRY__:
                    spawn_point = random.choice(spawn_points)
                    try:
                        vehicle = self.world.spawn_actor(self.vehicle_bp, spawn_point)
                        break
                    except RuntimeError:
                        tries += 1
                        continue
                if vehicle is None:
                    raise Exception(f"{self.__LOG_PREFIX__}: Could not spawn the vehicle in the environment after {self.__MAX_RETRY__} retries")
            return vehicle
        except RuntimeError as e:
            logger.warning(f"{self.__LOG_PREFIX__}: Could not spawn the vehicle with blueprint_id {self.blueprint_id} in the environment for the given location {self.location} and rotation {self.rotation}")
            return None
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while spawning the vehicle with blueprint_id {self.blueprint_id} in the environment")
            raise e
    
    def _build(self) -> None:
        """
        Build the vehicle.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Building the vehicle with blueprint_id {self.blueprint_id}")
        self._set_blueprint()
        self.vehicle = self._spawn_vehicle()
    
    def _pick_random(self, EnumClass: Enum) -> carla.VehicleDoor:
        """
        Pick a random door from the vehicle.
        Input parameters:
            - EnumClass: the class of the enum.
        Output:
            - carla.VehicleDoor: the random door.
        """
        return random.choice(list(EnumClass)).value
        
    def close_door(self, door_idx: carla.VehicleDoor = VehicleDoor.All.value, is_random: bool = False) -> bool:
        """
        Close the door of the vehicle.
        Input parameters:
            - door_idx: the index of the door to be closed.
            - is_random: whether to close a random door or not.
        Output:
            - bool: whether the door was closed or not.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Closing the door of the vehicle with id {self.vehicle.id} | {self.blueprint_id}")
        if is_random:
            door_idx = self._pick_random_door(VehicleDoor)
        try:
            self.vehicle.close_door(carla.VehicleDoor(door_idx))
            return True
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while closing the door of the vehicle with id {self.vehicle.id} | {self.blueprint_id} | {door_idx}")
            return False
    
    def open_door(self, door_idx: carla.VehicleDoor = VehicleDoor.All.value, is_random: bool = False) -> bool:
        """
        Open the door of the vehicle.
        Input parameters:
            - door_idx: the index of the door to be opened.
            - is_random: whether to open a random door or not.
        Output:
            - bool: whether the door was opened or not.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Opening the door of the vehicle with id {self.vehicle.id} | {self.blueprint_id}")
        if is_random:
            door_idx = self._pick_random_door(VehicleDoor)
        try:
            self.vehicle.open_door(carla.VehicleDoor(door_idx))
            return True
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while opening the door of the vehicle with id {self.vehicle.id} | {self.blueprint_id} | {door_idx}")
            return False
    
    def set_light_state(self, light_state: carla.VehicleLightState = VehicleLightState.NoneLight.value, is_random: bool = False) -> None:
        """
        Set the light state of the vehicle.
        Input parameters:
            - light_state: the light state to be set.
        Output:
            - None
        """
        logger.info(f"{self.__LOG_PREFIX__}: Setting the light state of the vehicle with id {self.vehicle.id} | {self.blueprint_id}")
        if is_random:
            light_state = self._pick_random_door(VehicleLightState)
        try:
            self.vehicle.set_light_state(carla.VehicleLightState(light_state))
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while setting the light state of the vehicle with id {self.vehicle.id} | {self.blueprint_id} | {light_state}")
            return False
    
    def get_light_state(self) -> Union[carla.VehicleLightState, None]:
        """
        Get the light state of the vehicle.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the light state of the vehicle with id {self.vehicle.id} | {self.blueprint_id}")
        try:
            return self.vehicle.get_light_state()
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while getting the light state of the vehicle with id {self.vehicle.id} | {self.blueprint_id}")
            return None
    
    def set_autopilot(self) -> bool:
        """
        Set the autopilot of the vehicle.
        Output:
            - bool: whether the autopilot was set or not.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Setting the autopilot of the vehicle with id {self.vehicle.id} | {self.blueprint_id}")
        try:
            self.vehicle.set_autopilot(True)
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while setting the autopilot of the vehicle with id {self.vehicle.id} | {self.blueprint_id}")
            return False
    
    def unset_autopilot(self) -> bool:
        """
        Unset the autopilot of the vehicle.
        Output:
            - bool: whether the autopilot was unset or not.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Unsetting the autopilot of the vehicle with id {self.vehicle.id} | {self.blueprint_id}")
        try:
            self.vehicle.set_autopilot(False)
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while unsetting the autopilot of the vehicle with id {self.vehicle.id} | {self.blueprint_id}")
            return False
    
    def destroy(self) -> None:
        """
        Destroy the vehicle from the environment.
        Output:
            - None
        """
        logger.info(f"{self.__LOG_PREFIX__}: Destroying the vehicle with id {self.vehicle.id}, {self.blueprint_id}, from the environment")
        self.vehicle.destroy()
    
    def __str__(self) -> str:
        """
        Return the string representation of the vehicle object.
        """
        if self.vehicle is None:
            return f"Vehicle: {self.blueprint_id} | could not be spawned"
        return f"Vehicle: {self.vehicle.id} | {self.blueprint_id} | {self.vehicle.get_location()} | {self.vehicle.get_transform().rotation}"