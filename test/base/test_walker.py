##########################################################################################################
# Define test suit for the Walker class
##########################################################################################################

import carla
import pytest
from typing import Callable
from test.utils import init_client
from src.model.enum import WalkerType
from src.base.walker import Walker, WalkerAI


class TestWalker:
    
    @pytest.mark.parametrize("blueprint_id", WalkerType)
    def test_walker_spawn(self, init_client: Callable, blueprint_id: str):
        """
        Test the spawning of the walker
        Input: init_client: fixture returned carla.World, blueprint_id: str
        """
        try:
            world = init_client
            walker = Walker(world, blueprint_id= blueprint_id.value)
            if walker.actor is None:
                pytest.skip(
                    "Walker actor is not spawned, try with other blueprint/spawn points")
            walker.actor.destroy()
        except Exception as e:
            pytest.fail(str(e))


class TestWalkerAI:

    def run_agent(self, ai_walker: WalkerAI):
        """
        Run the test for the walker with AI
        Input: ai_walker: WalkerAI
        """
        ai_walker.set_speed()
        ai_walker.set_ai_walker_destination()
        ai_walker.start_ai()
        ai_walker.stop_ai()
        ai_walker.reset_ai()
        ai_walker.destroy()

    def test_ai_walker_spawn(self, init_client: Callable):
        """
        Test the spawning of the walker with AI
        Input: init_client: fixture returned carla.World
        """
        try:
            world = init_client
            walker = Walker(
                world, blueprint_id=WalkerType.PEDESTRIAN_1_3.value)
            if walker.actor is None:
                pytest.skip(
                    "Walker actor is not spawned, try with other blueprint/spawn points")
            ai_walker = WalkerAI(
                world, parent=walker.actor, speed=walker.speed)
            self.run_agent(ai_walker)
            walker.destroy()
        except Exception as e:
            pytest.fail(str(e))
    
    def test_ai_walker_with_transform_spawn(self, init_client: Callable):
        """
        Test the spawning of the walker with AI and transform
        Input: init_client: fixture returned carla.World
        """
        try:
            world = init_client
            walker = Walker(
                world, blueprint_id=WalkerType.PEDESTRIAN_1_3.value)
            if walker.actor is None:
                pytest.skip("Walker actor is not spawned, try with other blueprint/spawn points")
            location = carla.Location(x=0, y=0, z=0.5)
            rotation = carla.Rotation()
            ai_walker = WalkerAI(world, parent=walker.actor,
                                 location=location, rotation=rotation, speed=walker.speed)
            self.run_agent(ai_walker)
            walker.destroy()
        except Exception as e:
            pytest.fail(str(e))