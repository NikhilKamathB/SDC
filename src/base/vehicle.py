##########################################################################################################
# Define a vehicle module here. This module will be used to define the vehicle class and its methods.
##########################################################################################################

import carla
import random
import logging

from typing import Union, List
from src.base.mixin import ActorMixin
from src.model.enum import VehicleDoor, VehicleLightState, Gen2VehicleType
from src.base.sensor import CameraRGB, CameraDepth, CameraSemanticSegmentation, CameraInstanceSegmentation


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
                 rotation: carla.Rotation = None,
                 **kwargs) -> None:
        """
        Initialize the vehicle with the blueprint id.
        Input parameters:
            - world: the carla world where the vehicle would be spawned.
            - blueprint_id: the blueprint id of the vehicle.
            - role_name: the role name of the vehicle.
            - location: the location where the vehicle would be spawned.
            - rotation: the rotation of the vehicle.
            - kwargs: additional keyword arguments.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the vehicle with blueprint id {blueprint_id}")
        super().__init__(world, blueprint_id, role_name, location, rotation,
                         spawn_on_road=True, spawn_on_side=False)
        self._build(**kwargs)
        self._is_ego = kwargs.get("is_ego", False)
        self._rgb_cameras = []
        self._depth_cameras = []
        self._semantic_segmentation_cameras = []
        self._instance_segmentation_cameras = []
    
    def is_ego(self) -> bool:
        """
        Check if this vehicle is an ego vehicle.
        Returns: bool
        """
        return self._is_ego
    
    def add_rgb_camera(self, rgb_camera: Union[CameraRGB, List[CameraRGB]]) -> None:
        """
        Add an RGB camera to the vehicle.
        Input parameters:
            - rgb_camera: a list or an instance of an RGB camera to be added.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Adding reference of an RGB camera to the vehicle with id {self.actor.id} | {self.blueprint_id}")
        if isinstance(rgb_camera, list):
            self._rgb_cameras.extend(rgb_camera)
        else:
            self._rgb_cameras.append(rgb_camera)
    
    def add_depth_camera(self, depth_camera: Union[CameraDepth, List[CameraDepth]]) -> None:
        """
        Add a depth camera to the vehicle.
        Input parameters:
            - depth_camera: a list or an instance of a depth camera to be added.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Adding reference of a depth camera to the vehicle with id {self.actor.id} | {self.blueprint_id}")
        if isinstance(depth_camera, list):
            self._depth_cameras.extend(depth_camera)
        else:
            self._depth_cameras.append(depth_camera)
    
    def add_semantic_segmentation_camera(self, semantic_segmentation_camera: Union[CameraSemanticSegmentation, List[CameraSemanticSegmentation]]) -> None:
        """
        Add a semantic segmentation camera to the vehicle.
        Input parameters:
            - semantic_segmentation_camera: a list or an instance of a semantic segmentation camera to be added.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Adding reference of a semantic segmentation camera to the vehicle with id {self.actor.id} | {self.blueprint_id}")
        if isinstance(semantic_segmentation_camera, list):
            self._semantic_segmentation_cameras.extend(semantic_segmentation_camera)
        else:
            self._semantic_segmentation_cameras.append(semantic_segmentation_camera)
    
    def add_instance_segmentation_camera(self, instance_segmentation_camera: Union[CameraInstanceSegmentation, List[CameraInstanceSegmentation]]) -> None:
        """
        Add an instance segmentation camera to the vehicle.
        Input parameters:
            - instance_segmentation_camera: a list or an instance of an instance segmentation camera to be added.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Adding reference of an instance segmentation camera to the vehicle with id {self.actor.id} | {self.blueprint_id}")
        if isinstance(instance_segmentation_camera, list):
            self._instance_segmentation_cameras.extend(instance_segmentation_camera)
        else:
            self._instance_segmentation_cameras.append(instance_segmentation_camera)
    
    def get_rgb_cameras(self) -> List[CameraRGB]:
        """
        Get the RGB cameras attached to the vehicle.
        Returns: List[CameraRGB]
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the RGB cameras attached to the vehicle with id {self.actor.id} | {self.blueprint_id}")
        return self._rgb_cameras

    def get_depth_cameras(self) -> List[CameraDepth]:
        """
        Get the depth cameras attached to the vehicle.
        Returns: List[CameraDepth]
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the depth cameras attached to the vehicle with id {self.actor.id} | {self.blueprint_id}")
        return self._depth_cameras

    def get_semantic_segmentation_cameras(self) -> List[CameraSemanticSegmentation]:
        """
        Get the semantic segmentation cameras attached to the vehicle.
        Returns: List[CameraSemanticSegmentation]
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the semantic segmentation cameras attached to the vehicle with id {self.actor.id} | {self.blueprint_id}")
        return self._semantic_segmentation_cameras
    
    def get_instance_segmentation_cameras(self) -> List[CameraInstanceSegmentation]:
        """
        Get the instance segmentation cameras attached to the vehicle.
        Returns: List[CameraInstanceSegmentation]
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the instance segmentation cameras attached to the vehicle with id {self.actor.id} | {self.blueprint_id}")
        return self._instance_segmentation_cameras
    
    def close_door(self, door_id: carla.VehicleDoor = VehicleDoor.ALL.value, is_random: bool = False) -> bool:
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
    
    def open_door(self, door_id: carla.VehicleDoor = VehicleDoor.ALL.value, is_random: bool = False) -> bool:
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
    
    def set_light_state(self, light_state: carla.VehicleLightState = VehicleLightState.NONE.value, is_random: bool = False) -> bool:
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
    
    def set_autopilot(self, tm_port: int = 8000) -> bool:
        """
        Set the autopilot of the vehicle.
        Input parameters:
            - tm_port: the port number of the traffic manager.
        Output:
            - bool: whether the autopilot was set or not.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Setting the autopilot of the vehicle with id {self.actor.id} | {self.blueprint_id}")
        try:
            self.actor.set_autopilot(True, tm_port=tm_port)
            return True
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while setting the autopilot of the vehicle with id {self.actor.id} | {self.blueprint_id} | {e}")
            return False
    
    def unset_autopilot(self, tm_port: int = 8000) -> bool:
        """
        Unset the autopilot of the vehicle.
        Input parameters:
            - tm_port: the port number of the traffic manager.
        Output:
            - bool: whether the autopilot was unset or not.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Unsetting the autopilot of the vehicle with id {self.actor.id} | {self.blueprint_id}")
        try:
            self.actor.set_autopilot(False, port=tm_port)
            return True
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while unsetting the autopilot of the vehicle with id {self.actor.id} | {self.blueprint_id} | {e}")
            return False