##########################################################################################################
# Define a RGB camera module here. This module will be used to define the camera class and its methods.
##########################################################################################################

import carla
import random
import logging
from src.base.mixin import ActorMixin
from src.model.enum import WalkerType


logger = logging.getLogger(__name__)


class Camera_RGB(ActorMixin):

    """
    Define a RGB camera that would get spawned in the environment.
    """

    __LOG_PREFIX__ = "Camera_RGB"

    def __init__(
                self,
                world: carla.World,
                blueprint_id: str = WalkerType.pedestrian_1_1.value,
                role_name: str = "camera_rgb",
                location: carla.Location = None,
                rotation: carla.Rotation = None) -> None:
        """
        Initialize the RGB camera with the blueprint id.
        Input parameters:
            - world: the carla world where the RGB camera would be spawned.
            - blueprint_id: the blueprint id of the RGB camera.
            - role_name: the role name of the RGB camera.
            - location: the location where the RGB camera would be spawned, relative to the parent actor.
            - rotation: the rotation of the RGB camera, relative to the parent actor.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the RGB camera with blueprint id {blueprint_id}")
        super().__init__(world, blueprint_id, role_name, spawn_on_road=False, spawn_on_side=False)
        pass
