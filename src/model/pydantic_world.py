##########################################################################################################
# Store all the pydantic models here
##########################################################################################################

import carla
from pydantic import BaseModel
from typing import Optional, List
from src.model.enum import Gen2VehicleType, WalkerType


class _BaseActor(BaseModel):

    """
    Define the base actor model here.
    """
    blueprint_id: Optional[str] = None
    role_name: Optional[str] = "actor"
    location: Optional[carla.Location] = None
    rotation: Optional[carla.Rotation] = None


class Vehicle(_BaseActor):

    """
    Define a vehicle that would get spawned in the environment.
    """
    blueprint_id: Optional[str] = Gen2VehicleType.LINCOLN_MKZ_2020.value
    role_name: Optional[str] = "vehicle"


class Walker(_BaseActor):
    
    """
    Define a walker that would get spawned in the environment.
    """
    blueprint_id: Optional[str] = WalkerType.pedestrian_1_1.value
    role_name: Optional[str] = "walker"
    attach_ai: Optional[bool] = True
    invincible: Optional[bool] = False
    run_probability: Optional[float] = 0.5


class Camera_RGB(_BaseActor):

    """
    Define a camera that would get spawned in the environment.
    """
    blueprint_id: Optional[str] = "sensor.camera.rgb"
    role_name: Optional[str] = "camera_rgb"
    fov: Optional[float] = 90.0
    image_size_x: Optional[int] = 800
    image_size_y: Optional[int] = 600
    bloom_intensity: Optional[float] = 0.675
    fstop: Optional[float] = 1.4
    iso: Optional[float] = 100.0
    gamma: Optional[float] = 2.2
    lens_flare_intensity: Optional[float] = 0.1
    sensor_tick: Optional[float] = 0.0
    shutter_speed: Optional[float] = 200.0
    enable_postprocess_effects: Optional[bool] = True


class Camera_Depth(_BaseActor):

    """
    Define a depth camera that would get spawned in the environment.
    """
    blueprint_id: Optional[str] = "sensor.camera.depth"
    role_name: Optional[str] = "camera_depth"
    image_size_x: Optional[int] = 800
    image_size_y: Optional[int] = 600
    fov: Optional[float] = 90.0
    sensor_tick: Optional[float] = 0.0
    lens_circle_falloff: Optional[float] = 5.0
    lens_circle_multiplier: Optional[float] = 0.0
    lens_k: Optional[float] = -1.0
    lens_kcube: Optional[float] = 0.0
    lens_x_size: Optional[float] = 0.08
    lens_y_size: Optional[float] = 0.08


class Sensor(BaseModel):

    """
    Define the sensor model here.
    """
    camera_rgb: Optional[List[Camera_RGB]] = None
    camera_depth: Optional[List[Camera_Depth]] = None


class World(BaseModel):

    """
    Define the Carla world here.
    """
    vehicles: List[Vehicle] = None
    walker: List[Walker] = None