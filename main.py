########################################################################################################################
# This will be the driver function. It contains the CLI bundle for executing various commands.
# To get to know more about the module run: `python -m main --help`
########################################################################################################################

import os
import glob
import logging
import typer as T
from dotenv import load_dotenv
from typing import Optional, List, TYPE_CHECKING
from src import AV2Forecasting, print_param_table
from utils import only_linux
if TYPE_CHECKING:
    from src import (
        read_yaml, write_yaml,
        CarlaClientCLI, DataSynthesizer, HighLevelMotionPlanner, SensorConvertorType,
        generate_vehicle_config, generate_pedestrian_config, write_txt_report_style_1
    )


__app__ = T.Typer()
__app__.prog_name = "CARLA CLIENT CLI"
load_dotenv()
logger = logging.getLogger(__name__)


# ---------------------------------------------  DATA  ------------------------------------------------------------------------
@__app__.command(name="generate_synthetic_data", help="This command generates synthetic data from various sensors in the Carla environment.")
@only_linux
def generate_synthetic_data(
    hostname: Optional[str] = T.Option(
        os.getenv("HOSTNAME", "localhost"), help="The hostname of the Carla server."),
    port: Optional[int] = T.Option(os.getenv(
        "PORT", 2000), help="The port on which the Carla server will be running."),
    carla_client_timeout: Optional[float] = T.Option(
        10.0, help="The connection timeout for the Carla client."),
    synchronous: Optional[bool] = T.Option(
        True, help="Whether to run the simulation in synchronous mode or not."),
    fixed_delta_seconds: Optional[float] = T.Option(
        0.05, help="The fixed delta seconds for the simulation - 0.05 ~ 20 FPS."),
    tm_enabled: Optional[bool] = T.Option(
        True, help="Whether to enable the traffic manager or not."),
    tm_port: Optional[int] = T.Option(
        8000, help="The port on which the traffic manager will be running."),
    tm_hybrid_physics_mode: Optional[bool] = T.Option(
        "True", help="Whether to run the traffic manager in hybrid physics mode or not."),
    tm_hybrid_physics_radius: Optional[float] = T.Option(
        70.0, help="The radius for the hybrid physics mode."),
    tm_global_distance_to_leading_vehicle: Optional[float] = T.Option(
        2.5, help="The global distance to the leading vehicle for the traffic manager."),
    tm_seed: Optional[int] = T.Option(
        42, help="The seed for the traffic manager."),
    tm_speed: Optional[float] = T.Option(
        60.0, help="The speed for actors controlled by the traffic manager."),
    tm_enable_autopilot_for_all_vehicles: Optional[bool] = T.Option(
        False, help="Whether to enable autopilot for all the vehicles - including ego vehicles - or not."),
    rfps: Optional[int] = T.Option(
        None, help="Record frame for every `k` steps - if provided, it will override the `rfps` in the sensor configuration."),
    spectator_enabled: Optional[bool] = T.Option(
        True, help="Whether to enable the spectator for custom spawning or not."),
    spectator_attachment_mode: Optional[str] = T.Option(
        'd', help="The mode of attachment for the spectator [d - default, e - ego vehicle, p - pedestrian, v - vehicle]."),
    spectator_location_offset: Optional[List[float]] = T.Option(
        [-7.0, 0.0, 5.0], help="The location offset for the spectator in [x, y, z] format. This is only applicable when the spectator is attached to the vehicle."),
    spectator_rotation: Optional[List[float]] = T.Option(
        [-15.0, 0.0, 0.0], help="The rotation offset for the spectator in [pitch, yaw, roll] format. This is only applicable when the spectator is attached to the vehicle."),
    max_simulation_time: Optional[int] = T.Option(
        100, help="The maximum time for which the simulation will run."),
    max_vechiles: Optional[int] = T.Option(
        50, help="The maximum number of vehicles (other than ego) in the Carla environment."),
    vechile_config_dir: Optional[str] = T.Option(
        "./data/config/vehicles", help="The directory containing the configuration files for the vehicles."),
    max_pedestrians: Optional[int] = T.Option(
        100, help="The maximum number of pedestrians in the Carla environment."),
    pedestrian_config_dir: Optional[str] = T.Option(
        "./data/config/pedestrians", help="The directory containing the configuration files for the pedestrians."),
    map: Optional[str] = T.Option(
        "Town01", help="The map of the Carla environment. Your options are [Town01, Town01_Opt, Town02, Town02_Opt, Town03, Town03_Opt, Town04, Town04_Opt, Town05, Town05_Opt, Town10HD, Town10HD_Opt]."),
    map_dir: Optional[str] = T.Option(
        "/Game/Carla/Maps", help="The directory where the maps are stored."),
    world_configuration: Optional[str] = T.Option(
        "./data/config/world0.yaml", help="The configuration file for the Carla world that holds defintion to smaller details."),
    output_directory: Optional[str] = T.Option(
        "./data/raw", help="The directory where the generated data will be stored."),
) -> None:
    """
    Generate synthetic data from the Carla environment.
    """
    # Print the configuration of this function.
    print_param_table(
        parmas=locals(), title="Parameters for `generate_synthetic_data(...)`")
    try:
        # Instantiate and configure the Carla client CLI.
        carla_cli = CarlaClientCLI(
            hostname=hostname,
            port=port,
            carla_client_timeout=carla_client_timeout,
            synchronous=synchronous,
            fixed_delta_seconds=fixed_delta_seconds,
            max_simulation_time=max_simulation_time,
            tm_enabled=tm_enabled,
            tm_port=tm_port,
            tm_hybrid_physics_mode=tm_hybrid_physics_mode,
            tm_hybrid_physics_radius=tm_hybrid_physics_radius,
            tm_global_distance_to_leading_vehicle=tm_global_distance_to_leading_vehicle,
            tm_seed=tm_seed,
            tm_speed=tm_speed,
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
            tm_enable_autopilot_for_all_vehicles=tm_enable_autopilot_for_all_vehicles,
            rfps=rfps,
            output_directory=output_directory
        )
        data_synthesizer.run()
    except Exception as e:
        logger.error(
            f"An error occurred while generating the synthetic data: {e}")
        raise e

@__app__.command(name="generate_configuration", help="This command generates configuration file for actors in the Carla environment.")
@only_linux
def generate_configuration(
    number_of_actors: Optional[int] = T.Option(
        50, help="The number of actors to be spawned in the Carla environment."),
    config_dir: Optional[str] = T.Option(
        "./data/config/vehicles", help="The directory where the configuration files will be stored."),
    reference_config_file: Optional[str] = T.Option(
        "./data/config/vehicles/vehicle0.yaml", help="The reference configuration file for the actor."),
    for_vehicle: Optional[bool] = T.Option(
        True, help="Whether to generate configuration for vehicles."),
    for_pedestrian: Optional[bool] = T.Option(
        False, help="Whether to generate configuration for pedestrians.")
) -> None:
    """
    Generate configuration file for the vehicles in the Carla environment.
    """
    # Print the configuration of this function.
    print_param_table(
        parmas=locals(), title="Parameters for `generate_configuration(...)`")
    assert (for_vehicle and for_pedestrian) == False, "Only one of `for_vehicle` and `for_pedestrian` can be True."
    try:
        logger.info("Generating configuration files for the actors...")
        # Read the reference configuration file.
        reference_configuration = read_yaml(reference_config_file)
        # Generate the configuration files for the actor.
        for i in range(1, number_of_actors + 1):
            logger.info(f"Generating configuration file for actor {i}...")
            if for_vehicle:
                new_actor_configuration = generate_vehicle_config(
                    reference_configuration, i)
                file_path = os.path.join(config_dir, f"vehicle_{i}.yaml")
            elif for_pedestrian:
                new_actor_configuration = generate_pedestrian_config(
                    reference_configuration, i)
                file_path = os.path.join(config_dir, f"pedestrian_{i}.yaml")
            write_yaml(file_path, new_actor_configuration)
    except Exception as e:
        logger.error(
            f"An error occurred while generating the configuration file: {e}")
        raise e


@__app__.command(name="generate_synthetic_data_report", help="This command generates report for all data generated synthetically.")
@only_linux
def generate_synthetic_data_report(
    data_dir: Optional[str] = T.Option(
        "./data/raw", help="The directory where the synthetic data is stored."),
    output_directory: Optional[str] = T.Option(
        "./data/interim", help="The directory where the reports will be stored."),
    prefix_tag: Optional[bool] = T.Option(
        False, help="Whether to prefix the data type as the tag within the report."),
    prefix_dir: Optional[bool] = T.Option(
        False, help="Whether to prefix the data file with the directory within the report."),
    need_file_name: Optional[bool] = T.Option(
        False, help="Whether to include the file name in the report.")
) -> None:
    """
    Generate report for the synthetic data generated.
    """
    # Print the configuration of this function.
    print_param_table(
        parmas=locals(), title="Parameters for `generate_synthetic_data_report(...)`")
    # Generate the report for the synthetic data.
    try:
        logger.info("Generating report for the synthetic data")
        os.makedirs(output_directory, exist_ok=True)
        for convertor_type in SensorConvertorType:
            name, value = convertor_type.name, convertor_type.value
            logger.info(f"Generating report for {name}")
            files = glob.glob(os.path.join(
                data_dir, "**", f"*_{value}.*"), recursive=True)
            # Handle logaritmic depth image reports when generating for depth images - remove the logaritmic depth image files - check pattern matching
            if value == SensorConvertorType.DEPTH.value:
                files = [
                    file for file in files if SensorConvertorType.LOGARITHMIC_DEPTH.value not in file]
            if files:
                output_file = os.path.join(output_directory, f"{value}.txt")
                write_txt_report_style_1(
                    files, output_file, value, prefix_tag, prefix_dir, need_file_name)
    except Exception as e:
        logger.error(
            f"An error occurred while generating the synthetic data report: {e}")
        raise e

# -----------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------  MOTION PLANNING  -------------------------------------------------------------
@__app__.command(name="generate_route", help="This command generates a graph of the given Carla Town map and uses it to find a path from the start to the goal.")
@only_linux
def generate_route(
    hostname: Optional[str] = T.Option(
        os.getenv("HOSTNAME", "localhost"), help="The hostname of the Carla server."),
    port: Optional[int] = T.Option(os.getenv(
        "PORT", 2000), help="The port on which the Carla server will be running."),
    carla_client_timeout: Optional[float] = T.Option(
        10.0, help="The connection timeout for the Carla client."),
    map: Optional[str] = T.Option(
        "Town10HD", help="The map of the Carla environment. Your options are [Town01, Town01_Opt, Town02, Town02_Opt, Town03, Town03_Opt, Town04, Town04_Opt, Town05, Town05_Opt, Town10HD, Town10HD_Opt]."),
    map_dir: Optional[str] = T.Option(
        "/Game/Carla/Maps", help="The directory where the maps are stored."),
    world_configuration: Optional[str]=T.Option(
        "./data/config/world0.yaml", help="The configuration file for the Carla world that holds defintion to smaller details."),
    distance_metric: Optional[str] = T.Option("euclidean", help="The distance metric to be used for the search algorithm. Your options are [euclidean, manhattan]."),
    search_algorithm: Optional[str] = T.Option("astar", help="The search algorithm to be used for finding the path. Your options are [bfs, dfs, ucs, astar]."),
    set_start_state: Optional[bool] = T.Option(True, help="Whether to manually set start state or not."),
    set_goal_state: Optional[bool] = T.Option(True, help="Whether to manually set goal state or not."),
    delimiter: Optional[str] = T.Option("__", help="The delimiter for the node representation."),
    figaspect: Optional[float] = T.Option(0.5, help="The aspect ratio of the figure."),
    verbose: Optional[bool] = T.Option(True, help="Whether to print the logs or not.")
) -> None:
    """
    Generate a graph of the given Carla Town map and use it to find a path from the start to the goal.
    To get to know more about the expected graph structure refer:
        https://github.com/NikhilKamathB/Algorithms/blob/main/search/include/search/environment.h
    """
    # Print the configuration of this function.
    print_param_table(
        parmas=locals(), title="Parameters for `generate_route(...)`")
    try:
        # Initiate and configure the Carla client CLI.
        carla_cli = CarlaClientCLI(
            hostname=hostname,
            port=port,
            carla_client_timeout=carla_client_timeout,
            map=map,
            map_dir=map_dir,
            world_configuration=world_configuration,
            tm_enabled=False
        )
        # Generate the map graph and find a path from the start to the goal.
        high_level_motion_planner = HighLevelMotionPlanner(
            carla_client_cli=carla_cli,
            distance_metric=distance_metric,
            search_algorithm=search_algorithm,
            set_start_state=set_start_state,
            set_goal_state=set_goal_state,
            node_name_delimiter=delimiter,
            figaspect=figaspect,
            verbose=verbose
        )
        high_level_motion_planner.run()
    except Exception as e:
        logger.error(
            f"An error occurred while generating the map graph: {e}")
        raise e

# -----------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------- AGROVERSE DATA  --------------------------------------------------------------
@__app__.command(name="visualize_agroverse_data", help="This command visualizes the Agroverse forecasting data.")
def visualize_agroverse_forecasting_data(
    input_directory: Optional[str] = T.Option(
        "./data/online/av2/train", help="The directory containing the Agroverse forecasting data instance."),
    output_directory: Optional[str] = T.Option(
        "./data/interim", help="The directory where the visualization will be stored."),
    scenario_id: Optional[str] = T.Option(
        "0000b0f9-99f9-4a1f-a231-5be9e4c523f7", help="The scenario id for the Agroverse forecasting data instance."),
    output_filename: Optional[str] = T.Option(
        None, help="The name of the output file."),
    raw: Optional[bool] = T.Option(
        True, help="Whether to visualize the data raw or not - using the av2 apis."),
    show_pedesrian_xing: Optional[bool] = T.Option(
        False, help="Whether to show pedestrian crossing or not."),
    codec: Optional[str] = T.Option(
        "mp4v", help="The codec for the video."),
    fps: Optional[int] = T.Option(
        10, help="The frames per second for the video.")
) -> None:
    # Print the configuration of this function.
    print_param_table(
        parmas=locals(), title="Parameters for `visualize_agroverse_forecasting_data(...)`")
    try:
        # Instantiate and configure the Agroverse forecasting dataset instance.
        av2_forecasting = AV2Forecasting(
            input_directory=input_directory,
            output_directory=output_directory,
            scenario_id=scenario_id,
            output_filename=output_filename,
            raw=raw,
            show_pedesrian_xing=show_pedesrian_xing,
            codec=codec,
            fps=fps
        )
        # Generate the scenario video for the given scenario id.
        _ = av2_forecasting.visualize()
    except Exception as e:
        logger.error(
            f"An error occurred while visualizing the Agroverse forecasting data: {e}")
        raise e
    
# -----------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------  MODULE TEST  ---------------------------------------------------------------
@__app__.command(name="hello_world", help="Hello World!")
def hello_world() -> None:
    print("Hello World!")

# -----------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    __app__()