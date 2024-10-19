#########################################################################################################################
# AV2 forecasting dataset | Forecasting : Define tasks/functionalities associated with AV2 forecasting dataset here.
#########################################################################################################################

import logging
from workers.av2_datasets import config
from workers.av2_datasets.app import app


logger = logging.getLogger(__name__)


class AV2Forecasting:

    """
    Define tasks/functionalities associated with AV2 forecasting dataset here.
    """

    __LOG_PREFIX__ = "AV2Forecasting"

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the AV2Forecasting instance.
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Initializing Agroverse Forecasting - av2 - dataset instance.")
        self.input_directory = kwargs.get("input_directory", None)
        self.output_directory = kwargs.get("output_directory", None)
        self.scenario_id = kwargs.get("scenario_id", None)
        self.output_filename = kwargs.get("output_filename", None)
        self.bev_fov_scale = max(0, kwargs.get("bev_fov_scale", 2))
        self.raw = kwargs.get("raw", True)
        self.av_configuration_path = kwargs.get("av_configuration_path", "./data/config/agroverse/vehicle0.yaml")
        self.show_pedestrian_xing = kwargs.get("show_pedestrian_xing", False)
        self.plot_occlusions = kwargs.get("plot_occlusions", True)
        self.codec = kwargs.get("codec", "mp4v")
        self.fps = kwargs.get("fps", 10)
        self.overwrite = kwargs.get("overwrite", False)

    async def visualize(self) -> str:
        """
        Visualize the AV2 forecasting dataset.
        Returns:
            str: The path to the visualized AV2 forecasting dataset.
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Visualizing AV2 forecasting dataset.")
        kwargs = {
            "input_directory": self.input_directory,
            "output_directory": self.output_directory,
            "scenario_id": self.scenario_id,
            "output_filename": self.output_filename,
            "bev_fov_scale": self.bev_fov_scale,
            "raw": self.raw,
            "av_configuration_path": self.av_configuration_path,
            "show_pedestrian_xing": self.show_pedestrian_xing,
            "plot_occlusions": self.plot_occlusions,
            "codec": self.codec,
            "fps": self.fps,
            "overwrite": self.overwrite
        }
        result = app.send_task(
            "viz.av2_motion_forecasting_dataset_single_file",
            exchange=config._exchange_name,
            routing_key=config._routing_key,
            kwargs=kwargs
        )
        return result.get()

    async def get_analytics(self, process_k: int = -1) -> str:
        """
        Get analytics of the AV2 forecasting dataset.
        Args:
            process_k (int): The number of scenarios to be processed, -1 for all.
        Returns:
            str: The path to the generated analytics file.
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Getting analytics of the AV2 forecasting dataset.")
        kwargs = {
            "input_directory": self.input_directory,
            "output_directory": self.output_directory,
            "output_filename": self.output_filename,
            "overwrite": self.overwrite,
            "process_k": process_k
        }
        result = app.send_task(
            "viz.av2_motion_forecasting_dataset_generate_analytics",
            exchange=config._exchange_name,
            routing_key=config._routing_key,
            kwargs=kwargs
        )
        return result.get()

    @classmethod
    async def query_max_occurrence(cls, *args, **kwargs) -> dict:
        """
        Query the maximum occurrence of the object type in the given CSV file.
        Returns:
            dict: The queried data.
        """
        logger.info(
            f"{cls.__LOG_PREFIX__}: Querying maximum occurrence of the object type in the given CSV file.")
        kwargs = {
            "csv_file": kwargs.get("csv_file", None),
            "object_type": kwargs.get("object_type", None),
            "top_k": kwargs.get("top_k", 10),
            "save": kwargs.get("save", False)
        }
        result = app.send_task(
            "viz.av2_motion_forecasting_dataset_query_max_occurrence",
            exchange=config._exchange_name,
            routing_key=config._routing_key,
            kwargs=kwargs
        )
        return result.get()
