##########################################################################################################
# Store all the pydantic models here
##########################################################################################################

import carla
from typing import Optional
from pydantic import BaseModel
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


class World(BaseModel):

    """
    Define the Carla world here.
    """
    vehicle: Vehicle = Vehicle()
    walker: Walker = Walker()