#########################################################################################################################
# Waymo Open-motion dataset | Forecasting : Define tasks/functionalities associated with Waymo Open-motion dataset here.
#########################################################################################################################

import logging
from workers.waymo_datasets import config
from workers.waymo_datasets.app import app


logger = logging.getLogger(__name__)


class WaymoForecasting:

    """
    Define tasks/functionalities associated with Waymo Open-motion dataset here.
    """

    __LOG_PREFIX__ = "WaymoForecasting"

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the WaymoForecasting instance.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing Waymo Forecasting.")
        self._input_directory = kwargs.get("input_directory", None)
        self._output_directory = kwargs.get("output_directory", None)
        self._scenario = kwargs.get("scenario", None)
        self._output_filename = kwargs.get("output_filename", None)
    
    def visualize(self):
        """
        Visualize the Waymo Open Motion Dataset.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Visualizing Waymo Open Motion Dataset.")
        result = app.send_task(
            "viz.waymo_open_motion_dataset", 
            exchange=config._exchange_name, 
            routing_key=config._routing_key,
            kwargs={
                "input_directory": self._input_directory,
                "output_directory": self._output_directory,
                "scenario": self._scenario,
                "output_filename": self._output_filename
            }
        )
        print(result.get())