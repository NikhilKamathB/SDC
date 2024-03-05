################################################################################################################
# Define a walker module here. This module will be used to define the walker/pedestrian class and its methods.
################################################################################################################

import carla
import random
import logging
from src.base.mixin import ActorMixin
from src.model.enum import WalkerType


logger = logging.getLogger(__name__)


class Walker(ActorMixin):

    """
    Define a walker that would get spawned in the environment.
    """

    __LOG_PREFIX__ = "Walker"

    def __init__(self, 
                 world: carla.World, 
                 blueprint_id: str = WalkerType.pedestrian_1_1.value,
                 role_name: str = "walker", 
                 location: carla.Location = None,
                 rotation: carla.Rotation = None,
                 attach_ai: bool = True,
                 invincible: bool = False,
                 run_probability: float = 0.5):
        """
        Initialize the walker with the blueprint id.
        Input parameters:
            - world: the carla world where the walker would be spawned.
            - blueprint_id: the blueprint id of the walker.
            - role_name: the role name of the walker.
            - location: the location where the walker would be spawned.
            - rotation: the rotation of the walker.
            - attach_ai: whether to attach the walker to the AI controller or not.
            - invincible: whether the walker is invincible or not.
            - run_probability: the probability of the walker running.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the walker with blueprint id {blueprint_id}")
        super().__init__(world, blueprint_id, role_name, location, rotation,
                         spawn_on_road=False, spawn_on_side=True)
        self.invincible = invincible
        self.attach_ai = attach_ai
        self.ai_walker = None
        self.run_probability = run_probability
        self._build()

    def _spawn_ai_walker(self) -> carla.Actor:
        """
        Spawn the walker with the AI controller.
        Output:
            - carla.Actor: the spawned walker.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Spawning the walker with AI controller")
        try:
            ai_controller_bp = self.world.get_blueprint_library().find("controller.ai.walker")
            ai_controller_bp.set_attribute("role_name", "ai_walker")
            ai_walker = self.world.spawn_actor(ai_controller_bp, carla.Transform(), attach_to=self.actor)
            return ai_walker
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while spawning the walker with AI controller")
            raise e
    
    def _set_speed(self) -> float:
        """
        Set the speed of the walker.
        Output:
            - float: the speed of the walker.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Setting the speed of the walker")
        try:
            if random.random() < self.run_probability:
                speed = self.actor_bp.get_attribute(
                    "speed").recommended_values[2]
            else:
                speed = self.actor_bp.get_attribute(
                    "speed").recommended_values[1]
            return float(speed)
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while setting the speed of the walker")
            raise e

    def _set_invincible(self) -> None:
        """
        Set the invincibility of the walker.
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Setting the invincibility of the walker")
        try:
            if self.invincible:
                self.actor_bp.set_attribute("is_invincible", "true")
            else:
                self.actor_bp.set_attribute("is_invincible", "false")
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: An error occurred while setting the invincibility of the walker")
            raise e
    
    def _build(self) -> None:
        """
        Build the walker.
        """
        super()._build()
        self._set_invincible()
        if self.attach_ai:
            self.ai_walker = self._spawn_ai_walker()
            speed = self._set_speed()
            self.ai_walker.set_max_speed(speed)
    
    def set_ai_walker_destination(self, location: carla.Location = None) -> None:
        """
        Set the destination for the ai walker.
        Input parameters:
            - location: the location where the walker would be spawned.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Setting the destination for the AI walker")
        try:
            if location is not None:
                spawn_point = self._get_spawn_point_from_location_and_rotation()
            else:
                spawn_point = self._get_random_spawn_point()
            self.ai_walker.go_to_location(spawn_point.location)
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while setting the destination for the AI walker")
            raise e
    
    def start_ai(self) -> None:
        """
        Start the ai walker.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Starting the ai walker")
        try:
            self.ai_walker.start()
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while starting the ai walker")
            raise e
    
    def stop_ai(self) -> None:
        """
        Stop the ai walker.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Stopping the ai walker")
        try:
            self.ai_walker.stop()
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while stopping the ai walker")
            raise e
    
    def reset_ai(self) -> None:
        """
        Reset the ai walker.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Resetting the ai walker")
        try:
            self.set_ai_walker_destination()
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while resetting the ai walker")
            raise e