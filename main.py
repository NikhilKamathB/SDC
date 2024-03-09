################################################################################################################
# This will be the driver function. It contains the CLI bundle for executing various commands.
################################################################################################################

import os
import logging
import typer as T
from typing import Optional
from src import print_param_table, read_yaml, write_yaml
from src import (
    CarlaClientCLI, DataSynthesizer,
    generate_vehicle_config, generate_pedestrian_config
)


__app__ = T.Typer()
__app__.prog_name = "CARLA CLIENT CLI"
logger = logging.getLogger(__name__)


@__app__.command(name="generate_synthetic_data", help="This command generates synthetic data from various sensors in the Carla environment.")
def generate_synthetic_data(
    hostname: Optional[str] = T.Option("localhost", help="The hostname of the Carla server."),
    port: Optional[int] = T.Option(2000, help="The port on which the Carla server will be running."),
    carla_client_timeout: Optional[float] = T.Option(2.0, help="The connection timeout for the Carla client."),
    asynchronous: Optional[bool] = T.Option(
        True, help="Whether to run the simulation in asynchronous mode or not."),
    max_simulation_time: Optional[int] = T.Option(100000, help="The maximum time for which the simulation will run."),
    max_vechiles: Optional[int] = T.Option(50, help="The maximum number of vehicles (other than ego) in the Carla environment."),
    vechile_config_dir: Optional[str] = T.Option("./data/config/vehicles", help="The directory containing the configuration files for the vehicles."),
    max_pedestrians: Optional[int] = T.Option(100, help="The maximum number of pedestrians in the Carla environment."),
    pedestrian_config_dir: Optional[str] = T.Option("./data/config/pedestrians", help="The directory containing the configuration files for the pedestrians."),
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
        asynchronous=asynchronous,
        max_simulation_time=max_simulation_time,
    )
    carla_cli.configure_environemnt(
        max_vechiles=max_vechiles,
        vechile_config_dir=vechile_config_dir,
        max_pedestrians=max_pedestrians,
        pedestrian_config_dir=pedestrian_config_dir
    )
    # Instantiate the data synthesizer and run the simulation.
    data_synthesizer = DataSynthesizer(carla_client_cli=carla_cli)
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