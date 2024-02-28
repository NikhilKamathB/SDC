##########################################################################################################
# Store all the pydantic models here
##########################################################################################################

from typing import Optional
from pydantic import BaseModel
from src.model.enum import Gen2VehicleType


class Vehicle(BaseModel):

    """
    Define a vehicle that would get spawned in the environment.
    """
    blueprint_id: Optional[str] = Gen2VehicleType.LINCOLN_MKZ_2020.value


class World(BaseModel):

    """
    Define the Carla world here.
    """
    vehicle: Vehicle = Vehicle()