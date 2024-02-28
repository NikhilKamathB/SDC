##########################################################################################################
# Define test suit for the Vehicle class
##########################################################################################################

import os
import carla
import pytest
from src.base.vehicle import Vehicle
from src.model.enum import Gen1VehicleType, Gen2VehicleType, VehicleLightState


@pytest.fixture
def init_client() -> carla.World:
    """
    Initialize the client for the test
    Return: carla.World
    """
    hostname = os.getenv("HOSTNAME", "localhost")
    port = int(os.getenv("PORT", "2000"))
    client = carla.Client(hostname, port)
    client.set_timeout(2.0)
    world = client.get_world()
    return world


class TestVehicle:

    def run_test(self, world, blueprint_id) -> None:
        """
        Run the test for the vehicle class
        Input: world: carla.World, blueprint_id: carla.ActorBlueprint
        """
        blueprint_id = blueprint_id.value
        vehicle = Vehicle(world, blueprint_id=blueprint_id)
        vehicle.open_door()
        vehicle.close_door()
        vehicle.set_light_state(light_state=VehicleLightState.LowBeam.value)
        _ = vehicle.get_light_state()
        vehicle.set_autopilot()
        vehicle.unset_autopilot()
        vehicle.destroy()

    @pytest.mark.parametrize("blueprint_id", Gen1VehicleType)
    def test_gen1_vehicle_spawn(self, init_client, blueprint_id):
        """
        Test the spawning of the generation 1 vehicle
        Input: init_client: fixture returned carla.World, blueprint_id: blueprint id
        """
        try:
            self.run_test(init_client, blueprint_id)
        except Exception as e:
            pytest.fail(str(e))
    
    @pytest.mark.parametrize("blueprint_id", Gen2VehicleType)
    def test_gen2_vehicle_spawn(self, init_client, blueprint_id):
        """
        Test the spawning of the generation 2 vehicle
        Input: init_client: fixture returned carla.World, blueprint_id: blueprint id
        """
        try:
            self.run_test(init_client, blueprint_id)
        except Exception as e:
            pytest.fail(str(e))