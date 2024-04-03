################################################################################################################
# Define Carla client here. This would be the gateway to the Carla server.
################################################################################################################

import os
import carla
import random
import logging
from src.base.vehicle import Vehicle
from src.base.sensor import CameraRGB
from src.base.walker import Walker, WalkerAI
from src.model import (
    enum,
    validators as PydanticModel
)
from src.utils.utils import read_yaml_file as read_yaml


logger = logging.getLogger(__name__)


class CarlaClientCLI:

    """
    Define a class for the Carla client CLI that would trigger different events/actions/tasks.
    """

    __LOG_PREFIX__ = "CarlaClientCLI"

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the Carla client CLI.
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Initializing the Carla client CLI")
        self.hostname = kwargs.get("hostname", "localhost")
        self.port = kwargs.get("port", 2000)
        self.carla_client_timeout = kwargs.get("carla_client_timeout", 2.0)
        self.max_simulation_time = kwargs.get("max_simulation_time", 100000)
        self.synchronous = kwargs.get("synchronous", True)
        self.fixed_delta_seconds = kwargs.get("fixed_delta_seconds", 0.05)
        self.tm_port = kwargs.get("tm_port", 8000)
        self.tm_hybrid_physics_mode = kwargs.get("tm_hybrid_physics_mode", True)
        self.tm_hybrid_physics_radius = kwargs.get("tm_hybrid_physics_radius", 70.0)
        self.tm_global_distance_to_leading_vehicle = kwargs.get("tm_global_distance_to_leading_vehicle", 2.5)
        self.tm_seed = kwargs.get("tm_seed", 42)
        self.spectator_enabled= kwargs.get("spectator_enabled", True)
        self.spectator_attachment_mode = kwargs.get(
            "spectator_attachment_mode", enum.SpectatorAttachmentMode.Vehicle.value)
        self.spectator_location_offset = kwargs.get(
            "spectator_location_offset", [10.0, 0.0, 10.0])
        self.spectator_rotation = kwargs.get(
            "spectator_rotation", [-90.0, 0.0, 0.0])
        self.map = kwargs.get("map", "Town01")
        self.map_dir = kwargs.get("map_dir", "/Game/Carla/Maps")
        self.world_configuration = kwargs.get("world_configuration", "./data/config/town01_default.yaml")
        self.vehicles, self.sensors = [], []
        self.walkers = []
        self.specator = None
        self.specator_attahced_to = None
        self._init_client()

    def _init_client(self) -> None:
        """
        Initialize the client for the Carla server.
        """
        try:
            logger.info(
                f"{self.__LOG_PREFIX__}: Initializing the client for the Carla server")
            self.client = carla.Client(self.hostname, self.port)
            self.client.set_timeout(self.carla_client_timeout)
            self.world = self._init_world()
            self.traffic_manager = self._init_traffic_manager()
            self.simulation_start_time = self.world.get_snapshot().timestamp.elapsed_seconds
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: Error while initializing the client for the Carla server: {e}")
            raise e

    def _init_world(self) -> carla.World:
        """
        Set the world for the Carla server.
        Return: carla.World
        """
        try:
            logger.info(
                f"{self.__LOG_PREFIX__}: Setting the world for the Carla server | Map: {self.map}.")
            # Set the world at a higher level
            self.client.load_world(os.path.join(self.map_dir, self.map))
            world = self.client.get_world()
            settings = world.get_settings()
            settings.synchronous_mode = self.synchronous
            if self.synchronous:
                settings.fixed_delta_seconds = self.fixed_delta_seconds
            world.apply_settings(settings)
            # Set the world at a lower level
            raw_world = PydanticModel.World(**read_yaml(self.world_configuration))
            world.set_weather(self._init_weather(raw_world.weather.model_dump(mode="json")))
            return world
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: Error while setting the world for the Carla server: {e}")
            raise e
    
    def _init_weather(self, config: dict) -> carla.WeatherParameters:
        """
        Set the weather for the Carla server.
        Input parameters:
            - config: dict - the configuration for the weather.
        Return: carla.WeatherParameters
        """
        try:
            logger.info(
                f"{self.__LOG_PREFIX__}: Setting the weather for the Carla server")
            return carla.WeatherParameters(**config)
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: Error while setting the weather from the Carla server: {e}")
            raise e
    
    def _init_traffic_manager(self) -> carla.TrafficManager:
        """
        Set the traffic manager from the Carla server.
        Return: carla.TrafficManager
        """
        try:
            logger.info(
                f"{self.__LOG_PREFIX__}: Initiating the traffic manager for the Carla server")
            traffic_manager = self.client.get_trafficmanager(self.tm_port)
            traffic_manager.set_hybrid_physics_mode(self.tm_hybrid_physics_mode)
            traffic_manager.set_hybrid_physics_radius(self.tm_hybrid_physics_radius)
            traffic_manager.set_global_distance_to_leading_vehicle(self.tm_global_distance_to_leading_vehicle)
            traffic_manager.set_random_device_seed(self.tm_seed)
            traffic_manager.set_synchronous_mode(self.synchronous)
            return traffic_manager
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: Error while initiating the traffic manager from the Carla server: {e}")
            raise e

    def _spawn_vehicle(self, config: dict) -> bool:
        """
        Spawn an vehicle in the Carla environment.
        Input parameters:
            - config: dict - configuration for the vehicle.
        Return: bool - whether the vehicle and the sensors assciated with them were spawned or not.
        """
        try:
            # Get the vehicle configuration.
            raw_vehicle = PydanticModel.Vehicle(**config)
            # Spawn the vehicle in the Carla environment.
            vehicle = Vehicle(
                world=self.world,
                blueprint_id=raw_vehicle.blueprint_id,
                role_name=raw_vehicle.role_name,
                location=raw_vehicle.to_carla_location(),
                rotation=raw_vehicle.to_carla_rotation()
            )
            # Log vehicle spawned.
            logger.info(
                f"{self.__LOG_PREFIX__}: Vehicle spawned with details: {vehicle}")
            if vehicle is not None:
                self.vehicles.append(vehicle)
                # Spawn the RGB cameras for the vehicle.
                vehicle_rgb_cameras = raw_vehicle.sensors.create_camera_rgb_objects(
                    world=self.world,
                    parent=vehicle.actor,
                )
                # Spawn the depth cameras for the vehicle.
                vehicle_depth_cameras = raw_vehicle.sensors.create_camera_depth_objects(
                    world=self.world,
                    parent=vehicle.actor,
                )
                # Spawn the segmentation cameras for the vehicle.
                vehicle_semantic_segmentation_cameras = raw_vehicle.sensors.create_camera_semantic_segmentation_objects(
                    world=self.world,
                    parent=vehicle.actor,
                )
                # Spawn the instance segmentation cameras for the vehicle.
                vehicle_instance_segmentation_cameras = raw_vehicle.sensors.create_camera_instance_segmentation_objects(
                    world=self.world,
                    parent=vehicle.actor,
                )
                # Update the sensors list.
                self.sensors += vehicle_rgb_cameras + vehicle_depth_cameras + vehicle_semantic_segmentation_cameras + vehicle_instance_segmentation_cameras
                # Log the sensors spawned.
                for sensor in vehicle_rgb_cameras + vehicle_depth_cameras:
                    logger.info(
                        f"{self.__LOG_PREFIX__}: Sensor spawned with details: {sensor}")
            else:
                logger.info(f"{self.__LOG_PREFIX__}: Vehicle and its associated sensors could not be spawned")
            return True
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: Error while spawning the vehicle: {e}")
            return False

    def _spawn_vehicles(self, config_dir: str, max_vechiles: int) -> None:
        """
        Spawn vehicles in the Carla environment.
        Input parameters:
            - config_dir: str - the directory containing the configuration files for the vehicles.
            - max_vechiles: int - the maximum number of vehicles (other than ego) in the Carla environment.
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Spawning vehicles in the Carla environment")
        for idx, config in enumerate(os.listdir(config_dir)[:max_vechiles]):
            vehicle_config = read_yaml(os.path.join(config_dir, config))
            success = self._spawn_vehicle(vehicle_config)
            logger.info(
                f"{self.__LOG_PREFIX__}: Vehicle and its associated sensors - {idx} spawned with status {success}")

    def _spawn_walker(self, config: dict) -> bool:
        """
        Spawn a walker in the Carla environment.
        Input parameters:
            - config: dict - configuration for the walker.
        Return: bool - whether the walker was spawned or not.
        """
        try:
            # Get the walker configuration.
            raw_walker = PydanticModel.Walker(**config)
            # Spawn the walker in the Carla environment.
            walker = Walker(
                world=self.world,
                blueprint_id=raw_walker.blueprint_id,
                role_name=raw_walker.role_name,
                location=raw_walker.to_carla_location(),
                rotation=raw_walker.to_carla_rotation(),
                is_invincible=raw_walker.is_invincible,
                run_probability=raw_walker.run_probability
            )
            # Log walker spawned.
            logger.info(
                f"{self.__LOG_PREFIX__}: Walker spawned with details: {walker}")
            # Attach the AI walker to the parent if needed.
            walker_ai = None
            if raw_walker.attach_ai and walker:
                walker_ai = WalkerAI(
                    world=self.world,
                    parent=walker.actor,
                    role_name=raw_walker.role_name,
                    location=raw_walker.to_carla_location(),
                    rotation=raw_walker.to_carla_rotation()
                )
                # Log walker AI spawned.
                logger.info(
                    f"{self.__LOG_PREFIX__}: Walker AI spawned with details: {walker_ai}")
            if walker is not None:
                self.walkers.append(
                    (walker, walker_ai) # walker_ai can be None
                )
            return True
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: Error while spawning the walker: {e}")
            return False

    def _spawn_walkers(self, config_dir: str, max_pedestrians: int) -> None:
        """
        Spawn walkers in the Carla environment.
        Input parameters:
            - config_dir: str - the directory containing the configuration files for the pedestrians.
            - max_pedestrians: int - the maximum number of pedestrians in the Carla environment.
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Spawning walkers in the Carla environment")
        for idx, config in enumerate(os.listdir(config_dir)[:max_pedestrians]):
            walker_config = read_yaml(os.path.join(config_dir, config))
            success = self._spawn_walker(walker_config)
            logger.info(
                f"{self.__LOG_PREFIX__}: Walker - {idx} spawned with status {success}")
    
    def _assist_spectator(self, actor: carla.Actor) -> carla.Actor:
        """
        Assist the spectator movement in the Carla environment - a trick
        Input parameters:
            - actor: carla.Actor - the actor to which the dummy sensor and therefore the spectator is attached.
        Output: 
            - carla.Actor - the dummy sensor attached to the actor.
        """
        try:
            assert self.specator is not None, "Spectator not initialized"
            _dummy_sensor = CameraRGB(world=self.world, parent=actor, location=carla.Location(*self.spectator_location_offset), rotation=carla.Rotation(*self.spectator_rotation), add_listener=False)
            self.specator.set_transform(_dummy_sensor.get_transform())
            return _dummy_sensor
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: Error while assisting the spectator movement: {e}")
    
    def _spawn_spectator(self) -> None:
        """
        Spawn the spectator in the Carla environment.
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Spawning the spectator in the Carla environment")
        try:
            self.specator = self.world.get_spectator()
            if self.spectator_enabled and self.spectator_attachment_mode != enum.SpectatorAttachmentMode.Default.value:
                if self.spectator_attachment_mode == enum.SpectatorAttachmentMode.Vehicle.value:
                    _actor = random.choice(self.vehicles)
                elif self.spectator_attachment_mode == enum.SpectatorAttachmentMode.Pedestrian.value:
                    _actor = random.choice(self.walkers)[0] # Get the walker and not the controller
                else:
                    return None
                self.specator_attahced_to = self._assist_spectator(_actor.actor)
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: Error while spawning the spectator in the Carla environment: {e}")

    def configure_environemnt(self, *args, **kwargs) -> None:
        """
        Configure the environment for the Carla server.
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Configuring the environment for the Carla server")
        try:
            # Spawn the vehicles in the Carla environment.
            self._spawn_vehicles(kwargs.get(
                "vechile_config_dir"), kwargs.get("max_vechiles"))
            # Spawn the walkers in the Carla environment.
            self._spawn_walkers(kwargs.get(
                "pedestrian_config_dir"), kwargs.get("max_pedestrians"))
            # Spawn the spectator in the Carla environment.
            self._spawn_spectator()
            # Perform tick
            self.tick()
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: Error while configuring the environment: {e}")
            self.clear_environment()
            raise e

    def tick(self) -> None:
        """
        Tick the Carla server.
        """
        try:
            if self.synchronous:
                self.world.tick()
            else:
                self.world.wait_for_tick()
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: Error while ticking the Carla server: {e}")
            raise e

    def clear_environment(self) -> None:
        """
        Clear the environment for the Carla server.
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Clearing the environment for the Carla server")
        try:
            # Destroy the vehicles from the Carla environment.
            for vehicle in self.vehicles:
                vehicle.destroy()
            # Destroy the walkers and the associated controller from the Carla environment.
            for walker, walker_controller in self.walkers:
                walker.destroy()
                if walker_controller:
                    walker_controller.destroy()
            # Destroy the sensors from the Carla environment.
            for sensor in self.sensors:
                sensor.destroy()
            # Remove misc actors
            if self.specator_attahced_to:
                self.specator_attahced_to.destroy()
        except Exception as e:
            logger.error(
                f"{self.__LOG_PREFIX__}: Error while clearing the environment: {e}")
            raise e