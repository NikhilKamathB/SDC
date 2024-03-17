################################################################################################################
# This will be the driver function. It contains the CLI bundle for executing various commands.
################################################################################################################

import os
import logging
import typer as T
from dotenv import load_dotenv
from typing import Optional, List
from src import print_param_table, read_yaml, write_yaml
from src import (
    CarlaClientCLI, DataSynthesizer,
    generate_vehicle_config, generate_pedestrian_config
)


__app__ = T.Typer()
__app__.prog_name = "CARLA CLIENT CLI"
load_dotenv()
logger = logging.getLogger(__name__)


@__app__.command(name="generate_synthetic_data", help="This command generates synthetic data from various sensors in the Carla environment.")
def generate_synthetic_data(
    hostname: Optional[str] = T.Option(os.getenv("HOSTNAME"), help="The hostname of the Carla server."),
    port: Optional[int] = T.Option(os.getenv("PORT"), help="The port on which the Carla server will be running."),
    carla_client_timeout: Optional[float] = T.Option(10.0, help="The connection timeout for the Carla client."),
    synchronous: Optional[bool] = T.Option(
        True, help="Whether to run the simulation in synchronous mode or not."),
    tm_port: Optional[int] = T.Option(8000, help="The port on which the traffic manager will be running."),
    tm_hybrid_physics_mode: Optional[bool] = T.Option("True", help="Whether to run the traffic manager in hybrid physics mode or not."),
    tm_hybrid_physics_radius: Optional[float] = T.Option(70.0, help="The radius for the hybrid physics mode."),
    tm_global_distance_to_leading_vehicle: Optional[float] = T.Option(2.5, help="The global distance to the leading vehicle for the traffic manager."),
    tm_seed: Optional[int] = T.Option(42, help="The seed for the traffic manager."),
    rfps: Optional[int] = T.Option(15, help="Record frame for every `k` steps."),
    spectator_enabled: Optional[bool] = T.Option(True, help="Whether to enable the spectator for custom spawning or not."),
    spectator_attachment_mode: Optional[str] = T.Option("v", help="The mode of attachment for the spectator [d - default, v - vehicle, p - pedestrian]."),
    spectator_location_offset: Optional[List[float]] = T.Option([-7.0, 0.0, 5.0], help="The location offset for the spectator in [x, y, z] format. This is only applicable when the spectator is attached to the vehicle."),
    spectator_rotation: Optional[List[float]] = T.Option([-15.0, 0.0, 0.0], help="The rotation offset for the spectator in [pitch, yaw, roll] format. This is only applicable when the spectator is attached to the vehicle."),
    max_simulation_time: Optional[int] = T.Option(100000, help="The maximum time for which the simulation will run."),
    max_vechiles: Optional[int] = T.Option(50, help="The maximum number of vehicles (other than ego) in the Carla environment."),
    vechile_config_dir: Optional[str] = T.Option("./data/config/vehicles", help="The directory containing the configuration files for the vehicles."),
    max_pedestrians: Optional[int] = T.Option(100, help="The maximum number of pedestrians in the Carla environment."),
    pedestrian_config_dir: Optional[str] = T.Option("./data/config/pedestrians", help="The directory containing the configuration files for the pedestrians."),
    map: Optional[str] = T.Option("Town01", help="The map of the Carla environment. Your options are [Town01, Town01_Opt, Town02, Town02_Opt, Town03, Town03_Opt, Town04, Town04_Opt, Town05, Town05_Opt, Town10HD, Town10HD_Opt]."),
    map_dir: Optional[str] = T.Option("/Game/Carla/Maps", help="The directory where the maps are stored."),
    world_configuration: Optional[str] = T.Option("./data/config/town01_default.yaml", help="The configuration file for the Carla world that holds defintion to smaller details."),
    output_directory: Optional[str] = T.Option("./data/raw", help="The directory where the generated data will be stored."),
) -> None:
    """
    Generate synthetic data from the Carla environment.
    """
    # Print the configuration of this function.
    print_param_table(
        parmas=locals(), title="Parameters for `generate_synthetic_data(...)`")
    # Instantiate and configure the Carla client CLI.
    carla_cli = CarlaClientCLI(
        hostname=hostname,
        port=port,
        carla_client_timeout=carla_client_timeout,
        synchronous=synchronous,
        max_simulation_time=max_simulation_time,
        tm_port=tm_port,
        tm_hybrid_physics_mode=tm_hybrid_physics_mode,
        tm_hybrid_physics_radius=tm_hybrid_physics_radius,
        tm_global_distance_to_leading_vehicle=tm_global_distance_to_leading_vehicle,
        tm_seed=tm_seed, 
        spectator_enabled=spectator_enabled,
        spectator_attachment_mode=spectator_attachment_mode,
        spectator_location_offset=spectator_location_offset,
        spectator_rotation=spectator_rotation,
        map=map,
        map_dir=map_dir,
        world_configuration=world_configuration
    )
    carla_cli.configure_environemnt(
        max_vechiles=max_vechiles,
        vechile_config_dir=vechile_config_dir,
        max_pedestrians=max_pedestrians,
        pedestrian_config_dir=pedestrian_config_dir
    )
    # Instantiate the data synthesizer and run the simulation.
    data_synthesizer = DataSynthesizer(
        carla_client_cli=carla_cli,
        rfps=rfps,
        output_directory=output_directory
    )
    data_synthesizer.run()


@__app__.command(name="generate_configuration", help="This command generates configuration file for actors in the Carla environment.")
def generate_configuration(
    number_of_actors: Optional[int] = T.Option(50, help="The number of actors to be spawned in the Carla environment."),
    config_dir: Optional[str] = T.Option(
        "./data/config/vehicles", help="The directory where the configuration files will be stored."),
    reference_config_file: Optional[str] = T.Option(
        "./data/config/vehicles/vehicle0.yaml", help="The reference configuration file for the actor."),
    for_vehicle: Optional[bool] = T.Option(True, help="Whether to generate configuration for vehicles."),
    for_pedestrian: Optional[bool] = T.Option(False, help="Whether to generate configuration for pedestrians.")
) -> None:  
    """
    Generate configuration file for the vehicles in the Carla environment.
    """
    # Print the configuration of this function.
    print_param_table(
        parmas=locals(), title="Parameters for `generate_vehicle_configuration(...)`")
    assert (for_vehicle and for_pedestrian) == False, "Only one of `for_vehicle` and `for_pedestrian` can be True."
    try:
        # Read the reference configuration file.
        reference_configuration = read_yaml(reference_config_file)
        # Generate the configuration files for the actor.
        for i in range(1, number_of_actors + 1):
            logger.info(f"Generating configuration file for actor {i}...")
            if for_vehicle:
                new_actor_configuration = generate_vehicle_config(reference_configuration, i)
                file_path = os.path.join(config_dir, f"vehicle_{i}.yaml")
            elif for_pedestrian:
                new_actor_configuration = generate_pedestrian_config(reference_configuration, i)
                file_path = os.path.join(config_dir, f"pedestrian_{i}.yaml")
            write_yaml(file_path, new_actor_configuration)
    except Exception as e:
        logger.error(f"An error occurred while generating the configuration file: {e}")
        raise e


if __name__ == "__main__":
    __app__()