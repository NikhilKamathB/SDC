################################################################################################################
# Script for generating any kind of synthetic data and storing it to a destined location.
################################################################################################################

import logging
from src.client import CarlaClientCLI

logger = logging.getLogger(__name__)


class DataSynthesizer:

    """
    Define the data synthesizer class here.
    """

    __LOG_PREFIX__ = "DataSynthesizer"

    def __init__(self, carla_client_cli: CarlaClientCLI, *args, **kwargs) -> None:
        """
        Initialize the data synthesizer.
        Input parameters:
            - carla_client_cli: the carla client command line interface that contians the carla world.
            - args and kwargs: additional arguments.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the data synthesizer")
        self.carla_client_cli = carla_client_cli
        self.rfps = kwargs.get("rfps", 15)
        self.output_directory = kwargs.get("output_directory", "./data/raw")
    
    def _set_vehicle_autopilot(self) -> None:
        """
        Set the vehicles to auto pilot mode.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Setting the vehicles to auto pilot mode with the traffic manager port {self.carla_client_cli.tm_port}")
        for vehicle in self.carla_client_cli.vehicles:
            vehicle.set_autopilot(tm_port=self.carla_client_cli.tm_port)
    
    def _set_sensor_registry(self) -> None:
        """
        Set the sensor registry.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Setting the sensor registry")
        for sensor in self.carla_client_cli.sensors:
            sensor.setup_registry(output_directory=self.output_directory, rfps=self.rfps)
    
    def _set_walker_ready(self) -> None:
        """
        Set the speed of the walkers.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Setting the walkers to ready")
        for walker, walker_controller in self.carla_client_cli.walkers:
            if walker_controller and walker: # Hard check
                walker_controller.set_ai_walker_destination()
                walker_controller.start_ai()
                walker_controller.set_speed(walker.speed)
    
    def _step_commit(self) -> None:
        """
        Perform custom step update here
        """
        logger.info(f"{self.__LOG_PREFIX__}: Performing the walker step")
        # Update walker
        for _, walker_controller in self.carla_client_cli.walkers:
            if walker_controller and walker_controller.destination and \
                walker_controller.actor.get_location().distance(walker_controller.destination) < 1.0:
                walker_controller.reset_ai()
                walker_controller.start_ai()
        # Update spectator
        if self.carla_client_cli.specator_attahced_to:
            self.carla_client_cli.specator.set_transform(
                self.carla_client_cli.specator_attahced_to.get_transform()
            )
    
    def _pre_commit(self) -> None:
        """
        Perform any pre-operations before actually ticking the simulator.
        """
        self._set_vehicle_autopilot()
        self._set_sensor_registry()
        self._set_walker_ready()
        self.carla_client_cli.tick()  # a tick to ensure client receives the recent information
        logger.info(f"{self.__LOG_PREFIX__}: Pre-commit operations completed in {self.carla_client_cli.world.get_snapshot().timestamp.elapsed_seconds - self.carla_client_cli.simulation_start_time} seconds")
        self.carla_client_cli.simulation_start_time = self.carla_client_cli.world.get_snapshot().timestamp.elapsed_seconds
    
    def run(self) -> None:
        """
        Run the data synthesizer.
        """
        try:
            self._pre_commit()
            while self.carla_client_cli.max_simulation_time > \
                    self.carla_client_cli.world.get_snapshot().timestamp.elapsed_seconds - self.carla_client_cli.simulation_start_time:
                snapshot = self.carla_client_cli.world.get_snapshot()
                logger.info(f"{self.__LOG_PREFIX__}: Running the data synthesizer | \
                            \n\tFrame elapsed: {snapshot.frame_count} | \
                                \n\tTime elapsed: {snapshot.timestamp.elapsed_seconds} | \
                                    \n\tDelta seconds since previous frame: {snapshot.timestamp.delta_seconds} | \
                                        \n\tPlatform timestamp: {snapshot.timestamp.platform_timestamp} | \
                                            \n\tTime remaining: {self.carla_client_cli.max_simulation_time - (snapshot.timestamp.elapsed_seconds - self.carla_client_cli.simulation_start_time)}")
                self._step_commit()
                self.carla_client_cli.tick()
        except KeyboardInterrupt:
            logger.warning(f"{self.__LOG_PREFIX__}: Keyboard interrupt occurred while running the data synthesizer")
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: Error occurred while running the data synthesizer | {e}")
            raise e
        finally:
            self.carla_client_cli.clear_environment()