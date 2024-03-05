##########################################################################################################
# Define test suit for the Walker class
##########################################################################################################

import carla
import pytest
from typing import Callable
from test.utils import init_client
from src.base.walker import Walker
from src.model.enum import WalkerType


class TestWalker:

    def run_test(self, world: carla.World, blueprint_id: str, attach_ai: bool = True) -> None:
        """
        Run the test for the walker class
        Input: world: carla.World, blueprint_id: str, attach_ai: bool = True
        """
        blueprint_id = blueprint_id.value
        walker = Walker(world, blueprint_id=blueprint_id, attach_ai=attach_ai)
        if attach_ai:
            walker.set_ai_walker_destination()
            walker.start_ai()
            walker.stop_ai()
            walker.reset_ai()
            walker.destroy()

    @pytest.mark.parametrize("blueprint_id", WalkerType)
    def test_walker_with_ai_controller_spawn(self, init_client: Callable, blueprint_id: str):
        """
        Test the spawning of the walker with the AI controller
        Input: init_client: fixture returned carla.World, blueprint_id: str
        """
        try:
            self.run_test(init_client, blueprint_id)
        except Exception as e:
            pytest.fail(str(e))
    
    @pytest.mark.parametrize("blueprint_id", WalkerType)
    def test_walker_spawn(self, init_client: Callable, blueprint_id: str):
        """
        Test the spawning of the walker
        Input: init_client: fixture returned carla.World, blueprint_id: str
        """
        try:
            self.run_test(init_client, blueprint_id, attach_ai=False)
        except Exception as e:
            pytest.fail(str(e))