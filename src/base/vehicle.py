##########################################################################################################
# Define a vehicle module here. This module will be used to define the vehicle class and its methods.
##########################################################################################################

import carla
import random
import logging

from typing import Union
from src.base.mixin import ActorMixin
from src.model.enum import VehicleDoor, VehicleLightState, Gen2VehicleType


logger = logging.getLogger(__name__)


class Vehicle(ActorMixin):

    """
    Define a vehicle that would get spawned in the environment.
    """
    
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
        super().__init__(world, blueprint_id, role_name, location, rotation,
                         spawn_on_road=True, spawn_on_side=False)
        self._build()
        
    def close_door(self, door_id: carla.VehicleDoor = VehicleDoor.All.value, is_random: bool = False) -> bool:
        """
        Close the door of the vehicle.
        Input parameters:
            - door_id: the index of the door to be closed.
            - is_random: whether to close a random door or not.
        Output:
            - bool: whether the door was closed or not.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Closing the door of the vehicle with id {self.actor.id} | {self.blueprint_id}")
        if is_random:
            door_id = self._pick_random_from_enum(VehicleDoor)
        try:
            self.actor.close_door(carla.VehicleDoor(door_id))
            return True
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while closing the door of the vehicle with id {self.actor.id} | {self.blueprint_id} | {door_id} | {e}")
            return False
    
    def open_door(self, door_id: carla.VehicleDoor = VehicleDoor.All.value, is_random: bool = False) -> bool:
        """
        Open the door of the vehicle.
        Input parameters:
            - door_id: the index of the door to be opened.
            - is_random: whether to open a random door or not.
        Output:
            - bool: whether the door was opened or not.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Opening the door of the vehicle with id {self.actor.id} | {self.blueprint_id}")
        if is_random:
            door_id = self._pick_random_from_enum(VehicleDoor)
        try:
            self.actor.open_door(carla.VehicleDoor(door_id))
            return True
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while opening the door of the vehicle with id {self.actor.id} | {self.blueprint_id} | {door_id} | {e}")
            return False
    
    def set_light_state(self, light_state: carla.VehicleLightState = VehicleLightState.NoneLight.value, is_random: bool = False) -> bool:
        """
        Set the light state of the vehicle.
        Input parameters:
            - light_state: the light state to be set.
        Output:
            - bool: whether the light state was set or not.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Setting the light state of the vehicle with id {self.actor.id} | {self.blueprint_id}")
        if is_random:
            light_state = self._pick_random_from_enum(VehicleLightState)
        try:
            self.actor.set_light_state(carla.VehicleLightState(light_state))
            return True
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while setting the light state of the vehicle with id {self.actor.id} | {self.blueprint_id} | {light_state} | {e}")
            return False
    
    def get_light_state(self) -> Union[carla.VehicleLightState, None]:
        """
        Get the light state of the vehicle.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the light state of the vehicle with id {self.actor.id} | {self.blueprint_id}")
        try:
            return self.actor.get_light_state()
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while getting the light state of the vehicle with id {self.actor.id} | {self.blueprint_id} | {e}")
            return None
    
    def set_autopilot(self) -> bool:
        """
        Set the autopilot of the vehicle.
        Output:
            - bool: whether the autopilot was set or not.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Setting the autopilot of the vehicle with id {self.actor.id} | {self.blueprint_id}")
        try:
            self.actor.set_autopilot(True)
            return True
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while setting the autopilot of the vehicle with id {self.actor.id} | {self.blueprint_id} | {e}")
            return False
    
    def unset_autopilot(self) -> bool:
        """
        Unset the autopilot of the vehicle.
        Output:
            - bool: whether the autopilot was unset or not.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Unsetting the autopilot of the vehicle with id {self.actor.id} | {self.blueprint_id}")
        try:
            self.actor.set_autopilot(False)
            return True
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while unsetting the autopilot of the vehicle with id {self.actor.id} | {self.blueprint_id} | {e}")
            return False