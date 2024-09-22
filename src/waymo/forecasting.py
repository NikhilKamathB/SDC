#########################################################################################################################
# Waymo Open-motion dataset | Forecasting : Define tasks/functionalities associated with Waymo Open-motion dataset here.
#########################################################################################################################

import logging
from workers.waymo import config
from workers.waymo.app import app


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
        self._input_directory = kwargs.get("input_directory", "/data/online/waymo/waymo_open_dataset_motion_v_1_2_1/uncompressed/tf_example")
        self._output_directory = kwargs.get("output_directory", "/data/interim")
        self._scenario_id = kwargs.get("scenario_id", "training_tfexample.tfrecord-00000-of-01000")
        self._output_filename = kwargs.get("output_filename", None)
    
    def visualize(self):
        """
        Visualize the Waymo Open Motion Dataset.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Visualizing Waymo Open Motion Dataset.")
        result = app.send_task("test", exchange=config._exchange_name, routing_key=config._routing_key)
        print(result.get())