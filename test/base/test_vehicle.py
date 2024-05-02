##########################################################################################################
# Define test suit for the Vehicle class
##########################################################################################################

import carla
import pytest
from typing import Callable
from test.utils import init_client
from src.base.vehicle import Vehicle
from src.model.enum import Gen1VehicleType, Gen2VehicleType, VehicleLightState

class TestVehicle:

    def run_test(self, world: carla.World, blueprint_id: str) -> None:
        """
        Run the test for the vehicle class
        Input: world: carla.World, blueprint_id: str
        """
        blueprint_id = blueprint_id.value
        vehicle = Vehicle(world, blueprint_id=blueprint_id)
        if vehicle.actor is None:
            pytest.skip(
                "Vehicle actor is not spawned, try with other blueprint/spawn points")
        vehicle.open_door()
        vehicle.close_door()
        vehicle.set_light_state(light_state=VehicleLightState.LOW_BEAM.value)
        _ = vehicle.get_light_state()
        vehicle.set_autopilot()
        vehicle.unset_autopilot()
        vehicle.destroy()

    @pytest.mark.parametrize("blueprint_id", Gen1VehicleType)
    def test_gen1_vehicle_spawn(self, init_client: Callable, blueprint_id: str):
        """
        Test the spawning of the generation 1 vehicle
        Input: init_client: fixture returned carla.World, blueprint_id: str
        """
        try:
            self.run_test(init_client, blueprint_id)
        except Exception as e:
            pytest.fail(str(e))
    
    @pytest.mark.parametrize("blueprint_id", Gen2VehicleType)
    def test_gen2_vehicle_spawn(self, init_client: Callable, blueprint_id: str):
        """
        Test the spawning of the generation 2 vehicle
        Input: init_client: fixture returned carla.World, blueprint_id: blueprint id
        """
        try:
            self.run_test(init_client, blueprint_id)
        except Exception as e:
            pytest.fail(str(e))