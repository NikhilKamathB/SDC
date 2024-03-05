##########################################################################################################
# Define utilities for the tests
##########################################################################################################

import os
import carla
import pytest


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