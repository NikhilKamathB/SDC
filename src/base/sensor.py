##########################################################################################################
# Define a sensor module here. This module will be used to define sensor classes and its methods.
##########################################################################################################

import os
import carla
import logging
from src.base.mixin import ActorMixin


logger = logging.getLogger(__name__)


class SensorMixin(ActorMixin):

    """
    Define a mixin for the sensor class.
    """
    
    def setup_registry(self, output_directory: str = "./data/raw", rfps: int = 15) -> None:
        """
        Setup the registry for the sensor.
        Input parameters:
            - output_directory: the output directory where the sensor data would be stored.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Setting up the registry for the sensor")
        self.rfps = rfps
        self.output_directory = os.path.join(output_directory, self.init_time, self.parent.type_id, self.role_name)
        os.makedirs(self.output_directory, exist_ok=True)
    
    def callback(self, sensor_data: carla.Image, **kwargs) -> None:
        """
        Define the callback function for the sensor.
        Note: `convertor` must be list of tuples with length 2, where the first element is the convertor type name and the second element is the convertor.
        """
        if sensor_data.frame % self.rfps != 0:
            return
        if isinstance(sensor_data, carla.Image):
            for convertor in kwargs.get("convertor", []):
                convertor_type, convertor = convertor
                sensor_data.save_to_disk(
                    os.path.join(
                        self.output_directory, f"{sensor_data.frame}_{sensor_data.timestamp}_{convertor_type}.png"),
                    convertor
                )
    
    def _build(self, **kwargs) -> None:
        """
        Build the sensor.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Building the sensor with role name: {self.role_name} and blueprint id: {self.blueprint_id}")
        super()._build(**kwargs)
        self._add_listener(**kwargs)


class CameraRGB(SensorMixin):

    """
    Define a RGB camera that would get spawned in the environment.
    """

    __LOG_PREFIX__ = "CameraRGB"

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
        self.output_directory = "./data/raw"
        self.rfps = 30
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the RGB camera with blueprint id {self.blueprint_id}")
        super().__init__(world, self.blueprint_id, role_name, location, rotation, parent=parent)
        self._build(**kwargs)
    
    def _add_listener(self, **kwargs) -> None:
        """
        Add the callback listener for the RGB camera.
        """
        if kwargs.get("add_listener", True):
            logger.info(f"{self.__LOG_PREFIX__}: Adding the callback listener for the RGB camera")
            self.actor.listen(lambda sensor_data: self.callback(sensor_data, convertor=[
                ("rgb", carla.ColorConverter.Raw),
            ]))


class CameraDepth(SensorMixin):

    """
    Define a depth camera that would get spawned in the environment.
    """

    __LOG_PREFIX__ = "CameraDepth"

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
        self.output_directory = "./data/raw"
        self.rfps = 30
        logger.info(
            f"{self.__LOG_PREFIX__}: Initializing the depth camera with blueprint id {self.blueprint_id}")
        super().__init__(world, self.blueprint_id, role_name, location, rotation, parent=parent)
        self._build(**kwargs)
    
    def _add_listener(self, **kwargs) -> None:
        """
        Add the callback listener for the depth camera.
        """
        if kwargs.get("add_listener", True):
            logger.info(
                f"{self.__LOG_PREFIX__}: Adding the callback listener for the depth camera")
            self.actor.listen(lambda sensor_data: self.callback(
                sensor_data, convertor=[
                    ("depth", carla.ColorConverter.Depth),
                    ("logarithmic_depth", carla.ColorConverter.LogarithmicDepth),
                ]))