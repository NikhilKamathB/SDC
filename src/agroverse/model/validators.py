##########################################################################################################
# Store all the pydantic models specific to the agroverse here
##########################################################################################################

import numpy as np
from typing import Optional, List
from typing_extensions import Annotated
from pydantic import BaseModel, Field, ConfigDict
from pydantic.functional_validators import AfterValidator
from src.agroverse.constants import CAMERA_TYPE_NAME

def validate_sensor_location(location: "Location") -> "Location":
    """
    Validate the location of the sensor.
    Args:
        location (Location): Location of the sensor.
    Returns:
        Location: Location of the sensor.
    """
    assert 0 <= location.x <= 1, "Location x should be between 0 and 1."
    assert 0 <= location.y <= 1, "Location y should be between 0 and 1."
    assert 0 <= location.z <= 1, "Location z should be between 0 and 1."
    return location

def validate_camera_fov(fov: float) -> float:
    """
    Validate the field of view of the camera.
    Args:
        fov (float): Field of view of the camera.
    Returns:
        float: Field of view of the camera.
    """
    assert 0 <= fov <= 180, "Field of view should be between 0 and 180."
    return fov

class Location(BaseModel):

    """
    Define the location model here.
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class Rotation(BaseModel):

    """
    Define the rotation model here in degrees.
    """
    yaw: float = 0.0  # Z-axis
    pitch: float = 0.0  # Y-axis
    roll: float = 0.0  # X-axis
    
    def get_rotation_radians(self) -> List[float]:
        """
        Get the rotation in radians.
        Returns:
            List[float]: Rotation in radians - yaw, pitch, roll.
        """
        return [np.deg2rad(self.yaw), np.deg2rad(self.pitch), np.deg2rad(self.roll)]


class _BaseSensor(BaseModel):

    """
    Define the base sensor model for agroverse here.
    """
    rotation: Rotation
    location: Annotated[Location, AfterValidator(validate_sensor_location)]


class CameraSensor(_BaseSensor):

    """
    Define the camera sensor model for agroverse here.
    """
    fov: Optional[Annotated[float, AfterValidator(validate_camera_fov)]] = 60.0
    image_size_x: Optional[int] = 800
    image_size_y: Optional[int] = 600

    def get_fov_radians(self) -> float:
        """
        Get the field of view in radians.
        Returns:
            float: Field of view in radians.
        """
        return np.deg2rad(self.fov)


class Sensor(BaseModel):

    """
    Define the sensor model for agroverse here.
    """
    cameras: Optional[List[CameraSensor]] = []

    def validate_location(self) -> None:
        """
        Validate the location of the sensors.
        """
        for camera in self.cameras:
            if not 0 <= camera.location.x <= 1:
                raise ValueError("Location x should be between 0 and 1.")
            if not 0 <= camera.location.y <= 1:
                raise ValueError("Location y should be between 0 and 1.")
            if not 0 <= camera.location.z <= 1:
                raise ValueError("Location z should be between 0 and 1.")


class Vehicle(BaseModel):

    """
    Define the vehicle model for agroverse here.
    """
    av_version: Optional[int] = 2
    sensors: Optional[Sensor] = None


class InternalSensorMountResponse(BaseModel):

    """
    Used by internal coding to generate response for sensor mount.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    bounds: np.ndarray

    def get_mid_bounds(self) -> np.ndarray:
        """
        Get the mid bounds of the sensor.
        Returns:
            np.ndarray: Mid bounds of the sensor.
        """
        return np.mean(self.bounds, axis=0)


class InternalCameraMountResponse(InternalSensorMountResponse):

    type: str = Field(CAMERA_TYPE_NAME, literal=True)
    camera: CameraSensor