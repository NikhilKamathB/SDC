##########################################################################################################
# Store all the pydantic models here
##########################################################################################################

import carla
import random
from pydantic import BaseModel
from typing import Optional, List, Union
from src.model.enum import Gen2VehicleType, WalkerType
from src.base.sensor import (
    CameraRGB as CameraRGBActor,
    CameraDepth as CameraDepthActor,
    CameraSemanticSegmentation as CameraSemanticSegmentationActor,
    CameraInstanceSegmentation as CameraInstanceSegmentationActor
)


class Location(BaseModel):

    """
    Define the location model here.
    """
    x: float
    y: float
    z: float

    def to_carla_location(self) -> carla.Location:
        """
        Convert the location to a carla location.
        Return: carla.Location
        """
        return carla.Location(x=self.x, y=self.y, z=self.z)


class Rotation(BaseModel):

    """
    Define the rotation model here.
    """
    pitch: float  # Y-axis
    yaw: float  # Z-axis
    roll: float  # X-axis

    def to_carla_rotation(self) -> carla.Rotation:
        """
        Convert the rotation to a carla rotation.
        Return: carla.Rotation
        """
        return carla.Rotation(pitch=self.pitch, yaw=self.yaw, roll=self.roll)


class _BaseActor(BaseModel):

    """
    Define the base actor model here.
    """
    blueprint_id: Optional[str] = None
    role_name: Optional[str] = "actor"
    location: Optional[Location] = None
    rotation: Optional[Rotation] = None

    def to_carla_location(self) -> Union[carla.Location, None]:
        """
        Convert the location to a carla location.
        Return: carla.Location
        """
        return self.location.to_carla_location() if self.location else None

    def to_carla_rotation(self) -> Union[carla.Rotation, None]:
        """
        Convert the rotation to a carla rotation.
        Return: carla.Rotation
        """
        return self.rotation.to_carla_rotation() if self.rotation else None


class Weather(BaseModel):

    """
    Define the weather model here.
    """
    cloudiness: Optional[float] = 0.0
    precipitation: Optional[float] = 0.0
    precipitation_deposits: Optional[float] = 0.0
    wind_intensity: Optional[float] = 0.0
    sun_azimuth_angle: Optional[float] = 0.0
    sun_altitude_angle: Optional[float] = 90.0
    fog_density: Optional[float] = 0.0
    fog_distance: Optional[float] = 0.0
    wetness: Optional[float] = 0.0
    fog_falloff: Optional[float] = 0.0
    scattering_intensity: Optional[float] = 0.0
    mie_scattering_scale: Optional[float] = 0.0
    rayleigh_scattering_scale: Optional[float] = 0.0331
    dust_storm: Optional[float] = 0.0


class World(BaseModel):

    """
    Define the world model here.
    """
    weather: Optional[Weather] = Weather()


class _BaseCameraSensor(_BaseActor):

    """
    Define the base sensor model here.
    """
    fov: Optional[float] = 90.0
    image_size_x: Optional[int] = 800
    image_size_y: Optional[int] = 600
    sensor_tick: Optional[float] = 0.0
    bloom_intensity: Optional[float] = 0.675
    fstop: Optional[float] = 1.4
    iso: Optional[float] = 100.0
    gamma: Optional[float] = 2.2
    lens_flare_intensity: Optional[float] = 0.1
    # Custom
    rfps: Optional[int] = 30
    output_directory: Optional[str] = "./data/raw"


class _BaseCameraLens(_BaseActor):

    """
    Define the base camera lens model here.
    """
    lens_circle_falloff: Optional[float] = 5.0
    lens_circle_multiplier: Optional[float] = 0.0
    lens_k: Optional[float] = -1.0
    lens_kcube: Optional[float] = 0.0
    lens_x_size: Optional[float] = 0.08
    lens_y_size: Optional[float] = 0.08


class CameraRGB(_BaseCameraSensor, _BaseCameraLens):

    """
    Define a camera that would get spawned in the environment.
    """
    blueprint_id: Optional[str] = "sensor.camera.rgb"
    enable_postprocess_effects: Optional[bool] = True
    focal_distance: Optional[float] = 1000.0


class CameraDepth(_BaseCameraSensor, _BaseCameraLens):

    """
    Define a depth camera that would get spawned in the environment.
    """
    blueprint_id: Optional[str] = "sensor.camera.depth"


class CameraSemanticSegmentation(_BaseCameraSensor, _BaseCameraLens):

    """
    Define a semantic segmentation camera that would get spawned in the environment.
    """
    blueprint_id: Optional[str] = "sensor.camera.semantic_segmentation"


class CameraInstanceSegmentation(_BaseCameraSensor, _BaseCameraLens):

    """
    Define a instance segmentation camera that would get spawned in the environment.
    """
    blueprint_id: Optional[str] = "sensor.camera.instance_segmentation"


class Sensor(BaseModel):

    """
    Define the sensor model here.
    """
    camera_rgb: Optional[List[CameraRGB]] = None
    camera_depth: Optional[List[CameraDepth]] = None
    camera_semantic_segmentation: Optional[List[CameraSemanticSegmentation]] = None
    camera_instance_segmentation: Optional[List[CameraInstanceSegmentation]] = None

    def create_camera_rgb_objects(self, world: carla.World, parent: carla.Actor = None) -> List[carla.Actor]:
        """
        Create the RGB camera objects.
        Input parameters:
            - world: the carla world where the cameras would be spawned.
            - parent: the parent actor of the cameras.
        Return: List[carla.Actor]
        """
        return [
            CameraRGBActor(
                world,
                parent=parent,
                location=data.location.to_carla_location(),
                rotation=data.rotation.to_carla_rotation(),
                **{k: v for k, v in data.model_dump().items() if k not in ['location', 'rotation']}
            )
            for data in self.camera_rgb
        ] if self.camera_rgb else []

    def create_camera_depth_objects(self, world: carla.World, parent: carla.Actor = None) -> List[carla.Actor]:
        """
        Create the depth camera objects.
        Input parameters:
            - world: the carla world where the cameras would be spawned.
            - parent: the parent actor of the cameras.
        Return: List[carla.Actor]
        """
        return [
            CameraDepthActor(
                world,
                parent=parent,
                location=data.location.to_carla_location(),
                rotation=data.rotation.to_carla_rotation(),
                **{k: v for k, v in data.model_dump().items() if k not in ['location', 'rotation']}
            )
            for data in self.camera_depth
        ] if self.camera_depth else []

    def create_camera_semantic_segmentation_objects(self, world: carla.World, parent: carla.Actor = None) -> List[carla.Actor]:
        """
        Create the semantic segmentation camera objects.
        Input parameters:
            - world: the carla world where the cameras would be spawned.
            - parent: the parent actor of the cameras.
        Return: List[carla.Actor]
        """
        return [
            CameraSemanticSegmentationActor(
                world,
                parent=parent,
                location=data.location.to_carla_location(),
                rotation=data.rotation.to_carla_rotation(),
                **{k: v for k, v in data.model_dump().items() if k not in ['location', 'rotation']}
            )
            for data in self.camera_semantic_segmentation
        ] if self.camera_semantic_segmentation else []

    def create_camera_instance_segmentation_objects(self, world: carla.World, parent: carla.Actor = None) -> List[carla.Actor]:
        """
        Create the instance segmentation camera objects.
        Input parameters:
            - world: the carla world where the cameras would be spawned.
            - parent: the parent actor of the cameras.
        Return: List[carla.Actor]
        """
        return [
            CameraInstanceSegmentationActor(
                world,
                parent=parent,
                location=data.location.to_carla_location(),
                rotation=data.rotation.to_carla_rotation(),
                **{k: v for k, v in data.model_dump().items() if k not in ['location', 'rotation']}
            )
            for data in self.camera_instance_segmentation
        ] if self.camera_instance_segmentation else []


class Vehicle(_BaseActor):

    """
    Define a vehicle that would get spawned in the environment.
    """
    blueprint_id: Optional[str] = random.choice(list(Gen2VehicleType)).value
    role_name: Optional[str] = "vehicle"
    sensors: Optional[Sensor] = None


class Walker(_BaseActor):

    """
    Define a walker that would get spawned in the environment.
    """
    blueprint_id: Optional[str] = random.choice(list(WalkerType)).value
    role_name: Optional[str] = "walker"
    is_invincible: Optional[bool] = False
    attach_ai: Optional[bool] = True
    run_probability: Optional[float] = 0.5
