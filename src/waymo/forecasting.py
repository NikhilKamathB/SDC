#########################################################################################################################
# Waymo Open-motion dataset | Forecasting : Define tasks/functionalities associated with Waymo Open-motion dataset here.
#########################################################################################################################

import os
import logging
import asyncio
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
        self._generate_json = kwargs.get("generate_json", False)
        self._bulk = kwargs.get("bulk", False)
        self._apply_async = kwargs.get("apply_async", False)
        self._process_k = kwargs.get("process_k", 4)

    async def preprocess(self) -> str:
        """
        Preprocess the Waymo Open Motion Dataset.
        Returns:
            str: The path to the preprocessed Waymo Open Motion Dataset.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Preprocessing Waymo Open Motion Dataset.")
        kwargs = {
            "input_directory": self._input_directory,
            "output_directory": self._output_directory,
            "scenario": self._scenario,
            "output_filename": self._output_filename,
            "generate_json": self._generate_json
        }
        if not self._bulk:
            logger.info(f"{self.__LOG_PREFIX__}: Processing single file.")
            assert self._scenario is not None, "Scenario is required to process single file."
            scenario_proto_path = os.path.join(self._input_directory, self._scenario)
            kwargs["scenario_proto_path"] = scenario_proto_path
            result = app.send_task(
                "preprocess.waymo_open_motion_dataset_single_file", 
                exchange=config._exchange_name, 
                routing_key=config._routing_key,
                kwargs=kwargs
            )
        else:
            logger.info(f"{self.__LOG_PREFIX__}: Processing bulk files.")
            kwargs["apply_async"] = self._apply_async
            kwargs["process_k"] = self._process_k
            result = app.send_task(
                "preprocess.waymo_open_motion_dataset",
                exchange=config._exchange_name,
                routing_key=config._routing_key,
                kwargs=kwargs
            )
        return result.get()
    
    async def visualize(self) -> str:
        """
        Visualize the Waymo Open Motion Dataset.
        Returns:
            str: The path to the visualized Waymo Open Motion Dataset.
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
        return result.get()