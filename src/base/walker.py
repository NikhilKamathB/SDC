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
                 **kwargs) -> None:
        """
        Initialize the walker with the blueprint id.
        Input parameters:
            - world: the carla world where the walker would be spawned.
            - blueprint_id: the blueprint id of the walker.
            - role_name: the role name of the walker.
            - location: the location where the walker would be spawned.
            - rotation: the rotation of the walker.
            - kwargs: additional keyword arguments.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the walker with blueprint id {blueprint_id}")
        super().__init__(world, blueprint_id, role_name, location, rotation,
                         spawn_on_road=False, spawn_on_side=True)
        self.run_probability = kwargs.get("run_probability", 0.5)
        self.ai_walker = None
        self._build(**kwargs)
    
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
    
    def _build(self, **kwargs) -> None:
        """
        Build the walker.
        """
        super()._build(**kwargs)
        self.speed = self._set_speed()
        

class WalkerAI(ActorMixin):

    """
    Define an AI walker that would get spawned in the environment but attahced to a parent.
    """

    __LOG_PREFIX__ = "WalkerAI"

    def __init__(self,
                 world: carla.World,
                 parent: carla.Actor,
                 blueprint_id: str = "controller.ai.walker",
                 role_name: str = "ai_walker",
                 location: carla.Location = None,
                 rotation: carla.Rotation = None,
                 **kwargs
                 ) -> None:
        """
        Initialize the AI walker with the blueprint id.
        Input parameters:
            - world: the carla world where the AI walker would be spawned.
            - parent: the parent actor of the AI walker.
            - blueprint_id: the blueprint id of the AI walker.
            - role_name: the role name of the AI walker.
            - location: the location where the AI walker would be spawned, relative to the parent actor.
            - rotation: the rotation of the AI walker, relative to the parent actor.
            - kwargs: additional keyword arguments.
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Initializing the walker with blueprint id {blueprint_id}")
        super().__init__(world, blueprint_id, role_name, location, rotation,
                         spawn_on_road=False, spawn_on_side=True, parent=parent)
        self.speed = kwargs.get("speed", 1.8)
        self._build(**kwargs)
    
    def set_speed(self, speed: float = None) -> None:
        """
        Set the speed of the AI walker.
        """
        if speed is not None:
            self.actor.set_max_speed(speed)
        else:
            self.actor.set_max_speed(self.speed)
    
    def set_ai_walker_destination(self, location: carla.Location = None) -> None:
        """
        Set the destination for the ai walker.
        Input parameters:
            - location: the location where the walker would be spawned.
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Setting the destination for the AI walker")
        try:
            if location is not None:
                spawn_point = self._get_spawn_point_from_location_and_rotation()
            else:
                spawn_point = self._get_random_spawn_point()
            self.actor.go_to_location(spawn_point.location)
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: An error occurred while setting the destination for the AI walker")
            raise e

    def start_ai(self) -> None:
        """
        Start the ai walker.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Starting the ai walker")
        try:
            self.actor.start()
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: An error occurred while starting the ai walker")
            raise e

    def stop_ai(self) -> None:
        """
        Stop the ai walker.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Stopping the ai walker")
        try:
            self.actor.stop()
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: An error occurred while stopping the ai walker")
            raise e

    def reset_ai(self) -> None:
        """
        Reset the ai walker.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Resetting the ai walker")
        try:
            self.set_ai_walker_destination()
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: An error occurred while resetting the ai walker")
            raise e
