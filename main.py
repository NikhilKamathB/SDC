################################################################################################################
# This will be the driver function. It contains the CLI bundle for executing various commands.
################################################################################################################

import os
import carla
import logging
import typer as T
from typing import Optional
from src import print_param_table, read_yaml
from src import PydanticModel, Walker, WalkerAI, Vehicle


__app__ = T.Typer()
__app__.prog_name = "CARLA CLIENT CLI"
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
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the Carla client CLI")
        self.hostname = kwargs.get("hostname", "localhost")
        self.port = kwargs.get("port", 2000)
        self.carla_client_timeout = kwargs.get("carla_client_timeout", 2.0)
        self.max_simulation_time = kwargs.get("max_simulation_time", 100000)
        self.vehicles, self.sensors = [], []
        self.walkers, self.ai_walkers = [], []
        self._init_client()

    def _init_client(self) -> None:
        """
        Initialize the client for the Carla server.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the client for the Carla server")
        self.client = carla.Client(self.hostname, self.port)
        self.client.set_timeout(self.carla_client_timeout)
        self.world = self._get_world()
    
    def _get_world(self) -> carla.World:
        """
        Get the world from the Carla server.
        Return: carla.World
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the world from the Carla server")
        return self.client.get_world()

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
            self.vehicles.append(vehicle)
            # Log vehicle spawned.
            logger.info(f"{self.__LOG_PREFIX__}: Vehicle spawned with details: {vehicle}")
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
            # Update the sensors list.
            self.sensors += vehicle_rgb_cameras + vehicle_depth_cameras
            # Log the sensors spawned.
            for sensor in vehicle_rgb_cameras + vehicle_depth_cameras:
                logger.info(f"{self.__LOG_PREFIX__}: Sensor spawned with details: {sensor}")
            return True
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: Error while spawning the vehicle: {e}")
            return False
    
    def _spawn_vehicles(self, config_dir: str, max_vechiles: int) -> None:
        """
        Spawn vehicles in the Carla environment.
        Input parameters:
            - config_dir: str - the directory containing the configuration files for the vehicles.
            - max_vechiles: int - the maximum number of vehicles (other than ego) in the Carla environment.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Spawning vehicles in the Carla environment")
        for idx, config in enumerate(os.listdir(config_dir)[:max_vechiles]):
            vehicle_config = read_yaml(os.path.join(config_dir, config))
            success = self._spawn_vehicle(vehicle_config)
            logger.info(f"{self.__LOG_PREFIX__}: Vehicle and its associated sensors - {idx} spawned with status {success}")

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
            self.walkers.append(walker)
            # Log walker spawned.
            logger.info(f"{self.__LOG_PREFIX__}: Walker spawned with details: {walker}")
            # Attach the AI walker to the parent if needed.
            if raw_walker.attach_ai:
                walker_ai = WalkerAI(
                    world=self.world,
                    parent=walker.actor,
                    role_name=raw_walker.role_name,
                    location=raw_walker.to_carla_location(),
                    rotation=raw_walker.to_carla_rotation()
                )
                self.ai_walkers.append(walker_ai)
                # Log walker AI spawned.
                logger.info(f"{self.__LOG_PREFIX__}: Walker AI spawned with details: {walker_ai}")
            return True
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: Error while spawning the walker: {e}")
            return False

    def _spawn_walkers(self, config_dir: str, max_pedestrians: int) -> None:
        """
        Spawn walkers in the Carla environment.
        Input parameters:
            - config_dir: str - the directory containing the configuration files for the pedestrians.
            - max_pedestrians: int - the maximum number of pedestrians in the Carla environment.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Spawning walkers in the Carla environment")
        for idx, config in enumerate(os.listdir(config_dir)[:max_pedestrians]):
            walker_config = read_yaml(os.path.join(config_dir, config))
            success = self._spawn_walker(walker_config)
            logger.info(
                f"{self.__LOG_PREFIX__}: Walker - {idx} spawned with status {success}")

    def configure_environemnt(self, *args, **kwargs) -> None:
        """
        Configure the environment for the Carla server.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Configuring the environment for the Carla server")
        try:
            # Spawn the vehicles in the Carla environment.
            self._spawn_vehicles(kwargs.get("vechile_config_dir"), kwargs.get("max_vechiles"))
            # Spawn the walkers in the Carla environment.
            self._spawn_walkers(kwargs.get("pedestrian_config_dir"), kwargs.get("max_pedestrians"))
            self.clear_environment()
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: Error while configuring the environment: {e}")
            self.clear_environment()
            raise e
    
    def clear_environment(self) -> None:
        """
        Clear the environment for the Carla server.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Clearing the environment for the Carla server")
        try:
            # Destroy the vehicles from the Carla environment.
            for vehicle in self.vehicles:
                vehicle.destroy()
            # Destroy the sensors from the Carla environment.
            for sensor in self.sensors:
                sensor.destroy()
            # Destroy the walkers from the Carla environment.
            for walker in self.walkers:
                walker.destroy()
            # Destroy the AI walkers from the Carla environment.
            for walker_ai in self.ai_walkers:
                walker_ai.destroy()
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: Error while clearing the environment: {e}")
            raise e


@__app__.command(name="generate_synthetic_data", help="This command generates synthetic data from various sensors in the Carla environment.")
def generate_synthetic_data(
    hostname: Optional[str] = T.Option("localhost", help="The hostname of the Carla server."),
    port: Optional[int] = T.Option(2000, help="The port on which the Carla server will be running."),
    carla_client_timeout: Optional[float] = T.Option(2.0, help="The connection timeout for the Carla client."),
    max_simulation_time: Optional[int] = T.Option(100000, help="The maximum time for which the simulation will run."),
    max_vechiles: Optional[int] = T.Option(25, help="The maximum number of vehicles (other than ego) in the Carla environment."),
    vechile_config_dir: Optional[str] = T.Option("./data/config/vehicles", help="The directory containing the configuration files for the vehicles."),
    max_pedestrians: Optional[int] = T.Option(100, help="The maximum number of pedestrians in the Carla environment."),
    pedestrian_config_dir: Optional[str] = T.Option("./data/config/pedestrians", help="The directory containing the configuration files for the pedestrians."),
):
    # Print the configuration of this function.
    print_param_table(
        parmas=locals(), title="Parameters for `generate_synthetic_data(...)`")
    # Instantiate the Carla client CLI.
    CarlaClientCLI(
        hostname=hostname,
        port=port,
        carla_client_timeout=carla_client_timeout,
        max_simulation_time=max_simulation_time,
    ).configure_environemnt(
        max_vechiles=max_vechiles,
        vechile_config_dir=vechile_config_dir,
        max_pedestrians=max_pedestrians,
        pedestrian_config_dir=pedestrian_config_dir
    )





if __name__ == "__main__":
    __app__()