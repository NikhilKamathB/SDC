################################################################################################################
# Define a base mixins here. These mixins will be used to define the base classes for the project.
################################################################################################################

import carla
import random
import logging
from enum import Enum


logger = logging.getLogger(__name__)


class ActorMixin:
    
    """
    Define a mixin for the actor - vehicle, walker, sensors, props, etc. class.
    """

    __MAX_RETRY__ = 25
    __LOG_PREFIX__ = "ActorMixin"

    def __init__(self, 
                 world: carla.World, 
                 blueprint_id: str, 
                 role_name: str = "actor", 
                 location: carla.Location = None,
                 rotation: carla.Rotation = None,
                 spawn_on_road: bool = True,
                 spawn_on_side: bool = False,
                 parent: carla.Actor = None) -> None:
        """
        Initialize the actor with the blueprint id.
        Input parameters:
            - blueprint_id: the blueprint id of the actor.
            - world: the carla world where the actor would be spawned.
            - role_name: the role name of the actor.
            - location: the location where the actor would be spawned.
            - rotation: the rotation of the actor.
            - spawn_on_road: whether to spawn the actor on the road or not.
            - spawn_on_side: whether to spawn the actor on the side (other than road) or not.
            - parent: the parent actor of the actor.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the actor with blueprint id {blueprint_id}")
        self.world = world
        self.blueprint_id = blueprint_id
        self.role_name = role_name
        self.location = location
        self.rotation = rotation
        self.spawn_on_road = spawn_on_road
        self.spawn_on_side = spawn_on_side
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.parent = parent
    
    def _pick_random_from_enum(self, enum_class: Enum) -> str:
        """
        Pick a random door from the vehicle.
        Input parameters:
            - EnumClass: the class of the enum.
        Output:
            - str: the random value.
        """
        return random.choice(list(enum_class)).value
    
    def _set_blueprint(self, **kwargs) -> None:
        """
        Set the blueprint of the actor.
        Input parameters:
            - kwargs: the parameters to set the blueprint of the actor.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Setting the blueprint of the actor with blueprint_id {self.blueprint_id}")
        try:
            self.actor_bp = self.world.get_blueprint_library().find(self.blueprint_id)
            self.actor_bp.set_attribute("role_name", self.role_name)
            for key, value in kwargs.items():
                if self.actor_bp.has_attribute(key):
                    self.actor_bp.set_attribute(key, str(value))
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while setting the blueprint of the actor with blueprint_id {self.blueprint_id}")
            raise e
    
    def _generate_side_spawn_point(self) -> carla.Transform:
        """
        Generate a spawn point on the side (anypoint other than on the road).
        """
        spawn_point = carla.Transform()
        spawn_point.location = self.world.get_random_location_from_navigation()
        return spawn_point

    def _get_spawn_point_from_location_and_rotation(self) -> carla.Transform:
        """
        Get a spawn point from the given location and rotation members.
        """
        if self.rotation is None:
            self.rotation = carla.Rotation(0, 0, 0)
        return carla.Transform(self.location, self.rotation)

    def _get_random_spawn_point(self) -> carla.Transform:
        """
        Get a random spawn point for the actor.
        Output:
            - carla.Transform: the spawn point.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting a random spawn point for the actor with blueprint_id {self.blueprint_id}")
        if self.spawn_on_road:
            return random.choice(self.spawn_points)
        elif self.spawn_on_side:
            return self._generate_side_spawn_point()
        else:
            return random.choice(self.spawn_points + [self._generate_side_spawn_point()])
    
    def _spawn_actor(self) -> carla.Actor:
        """
        Spawn the actor in the environment.
        Output:
            - carla.Actor: the spawned actor.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Spawning the actor in the environment")
        try:
            if self.location is not None:
                if self.parent:
                    # location and rotation becomes relative to the parent actor
                    actor = self.world.spawn_actor(
                        self.actor_bp, self._get_spawn_point_from_location_and_rotation(), attach_to=self.parent)
                else:
                    actor = self.world.spawn_actor(
                        self.actor_bp, self._get_spawn_point_from_location_and_rotation())
            else:
                if self.parent is not None:
                    actor = self.world.spawn_actor(self.actor_bp, carla.Transform(), attach_to=self.parent)
                else:
                    tries = 0
                    while tries < self.__MAX_RETRY__:
                        spawn_point = self._get_random_spawn_point()
                        try:
                            actor = self.world.spawn_actor(self.actor_bp, spawn_point)
                            break
                        except RuntimeError:
                            tries += 1
                            continue
                    if actor is None:
                        raise RuntimeError(f"{self.__LOG_PREFIX__}: Could not spawn the actor in the environment after {self.__MAX_RETRY__} retries")
            return actor
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: An error occurred while spawning the actor with blueprint_id {self.blueprint_id} in the environment")
            return None
    
    def _build(self, **kwargs) -> None:
        """
        Build the actor.
        Input parameters:
            - kwargs: the parameters to build the actor.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Building the actor with blueprint_id {self.blueprint_id}")
        self._set_blueprint(**kwargs)
        self.actor = self._spawn_actor()

    def destroy(self) -> None:
        """
        Destroy the actor from the environment.
        Output:
            - None
        """
        logger.info(f"{self.__LOG_PREFIX__}: Destroying the actor with id {self.actor.id}, {self.blueprint_id}, from the environment")
        if self.actor.is_alive:
            self.actor.destroy()
    
    def __str__(self) -> str:
        """
        Return the string representation of the actor.
        """
        if self.actor is None:
            return f"{self.role_name.capitalize()}: {self.blueprint_id} | could not be spawned"
        return f"{self.role_name.capitalize()}: {self.actor.id} | {self.blueprint_id} | {self.actor.get_location()} | {self.actor.get_transform().rotation}"