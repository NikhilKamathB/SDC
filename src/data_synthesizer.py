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

    def __init__(self, carla_client_cli: CarlaClientCLI) -> None:
        """
        Initialize the data synthesizer.
        Input parameters:
            - carla_client_cli: the carla client command line interface that contians the carla world.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the data synthesizer")
        self.carla_client_cli = carla_client_cli
    
    def run(self) -> None:
        """
        Run the data synthesizer.
        """
        try:
            while self.carla_client_cli.max_simulation_time > \
                    self.carla_client_cli.world.get_snapshot().timestamp.elapsed_seconds - self.carla_client_cli.simulation_start_time:
                snapshot = self.carla_client_cli.world.get_snapshot()
                logger.info(f"{self.__LOG_PREFIX__}: Running the data synthesizer | \
                            \n\tFrame elapsed: {snapshot.frame_count} | \
                                \n\tTime elapsed: {snapshot.timestamp.elapsed_seconds} | \
                                    \n\tDelta seconds since previous frame: {snapshot.timestamp.delta_seconds} | \
                                        \n\tPlatform timestamp: {snapshot.timestamp.platform_timestamp} | \
                                            \n\tTime remaining: {self.carla_client_cli.max_simulation_time - (snapshot.timestamp.elapsed_seconds - self.carla_client_cli.simulation_start_time)}")
                self.carla_client_cli.world.tick()
        except KeyboardInterrupt:
            logger.warning(f"{self.__LOG_PREFIX__}: Keyboard interrupt occurred while running the data synthesizer")
        except Exception as e:
            logger.error(f"{self.__LOG_PREFIX__}: Error occurred while running the data synthesizer | {e}")
            raise e
        finally:
            self.carla_client_cli.clear_environment()