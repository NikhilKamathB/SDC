##########################################################################################################
# Define a sensor module here. This module will be used to define sensor classes and its methods.
##########################################################################################################

import carla
import logging
from src.base.mixin import ActorMixin


logger = logging.getLogger(__name__)


class CameraRGB(ActorMixin):

    """
    Define a RGB camera that would get spawned in the environment.
    """

    __LOG_PREFIX__ = "Camera_RGB"

    def __init__(
                self,
                world: carla.World,
                parent: carla.Actor = None,
                role_name: str = "camera_rgb",
                location: carla.Location = None,
                rotation: carla.Rotation = None,
                **kwargs
                ) -> None:
        """
        Initialize the RGB camera with the blueprint id.
        Input parameters:
            - world: the carla world where the RGB camera would be spawned.
            - parent: the parent actor of the RGB camera.
            - blueprint_id: the blueprint id of the RGB camera.
            - role_name: the role name of the RGB camera.
            - location: the location where the RGB camera would be spawned, relative to the parent actor.
            - rotation: the rotation of the RGB camera, relative to the parent actor.
            - kwargs: additional keyword arguments.
        """
        self.blueprint_id = "sensor.camera.rgb"
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the RGB camera with blueprint id {self.blueprint_id}")
        super().__init__(world, self.blueprint_id, role_name, location, rotation, parent=parent)
        self._build(**kwargs)


class CameraDepth(ActorMixin):

    """
    Define a depth camera that would get spawned in the environment.
    """

    __LOG_PREFIX__ = "Camera_Depth"

    def __init__(
        self,
        world: carla.World,
        parent: carla.Actor = None,
        role_name: str = "camera_depth",
        location: carla.Location = None,
        rotation: carla.Rotation = None,
        **kwargs
    ) -> None:
        """
        Initialize the Depth camera with the blueprint id.
        Input parameters:
            - world: the carla world where the depth camera would be spawned.
            - parent: the parent actor of the depth camera.
            - blueprint_id: the blueprint id of the depth camera.
            - role_name: the role name of the depth camera.
            - location: the location where the depth camera would be spawned, relative to the parent actor.
            - rotation: the rotation of the depth camera, relative to the parent actor.
            - kwargs: additional keyword arguments.
        """
        self.blueprint_id = "sensor.camera.depth"
        logger.info(
            f"{self.__LOG_PREFIX__}: Initializing the depth camera with blueprint id {self.blueprint_id}")
        super().__init__(world, self.blueprint_id, role_name, location, rotation, parent=parent)
        self._build(**kwargs)