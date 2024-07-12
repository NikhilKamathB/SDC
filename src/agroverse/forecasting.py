################################################################################################################
# Agroverse | Forecasting : Define tasks/functionalities associated with Agroverse Forecasting dataset here.
################################################################################################################

import os
import logging
from typing import Tuple
from pathlib import Path
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.viz.scenario_visualization import visualize_scenario


logger = logging.getLogger(__name__)


class AV2Forecasting:

    """
    Define tasks/functionalities associated with Agroverse Forecasting dataset instance here.
    """

    __LOG_PREFIX__ = "AV2Forecasting"

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize Agroverse Forecasting dataset instance.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing Agroverse Forecasting dataset instance.")
        self.input_directory = kwargs.get("input_directory", None)
        assert self.input_directory is not None, "Input directory not provided."
        self.output_directory = kwargs.get("output_directory", None)
        assert self.output_directory is not None, "Output directory not provided."
        self.scenario_id = kwargs.get("scenario_id", None)
        assert self.scenario_id is not None, "Scenario ID not provided."
        self.output_filename = kwargs.get("output_filename", None)
    
    def _make_output_directory(self) -> None:
        """
        Make output directory.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Making output directory if not exists.")
        os.makedirs(self.output_directory, exist_ok=True)
    
    def _get_input_file_names(self) -> Tuple[str, str]:
        """
        Get input file names - static map and scenario.
        As per the agroverse v2 dataset the input directory should contain two files:
        1. log_map_archive_<scenario_id>.json   # static map
        2. scenario_<scenario_id>.parquet    # scenario file
        Returns:
            Tuple[str, str]: Tuple containing static map and scenario file names.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting input file names.")
        static_map_file = f"log_map_archive_{self.scenario_id}.json"
        scenario_file = f"scenario_{self.scenario_id}.parquet"
        return static_map_file, scenario_file

    def generate_scenario_video(self) -> str:
        """
        Generate scenario video for the given scenario id.
        Returns:
            str: Path to the generated scenario video.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Generating scenario video for scenario id: {self.scenario_id}")
        static_map_file, scenario_file = self._get_input_file_names()
        scenario = scenario_serialization.load_argoverse_scenario_parquet(
            Path(os.path.join(self.input_directory, self.scenario_id, scenario_file))
        )
        static_map = ArgoverseStaticMap.from_json(
            Path(os.path.join(self.input_directory, self.scenario_id, static_map_file))
        )
        if self.output_filename is None:
            self.output_filename = f"{self.scenario_id}.mp4"
        self._make_output_directory()
        output_path = Path(os.path.join(self.output_directory, self.output_filename))
        visualize_scenario(scenario, static_map, output_path)
        logger.info(f"{self.__LOG_PREFIX__}: Scenario video generated at: {output_path}")
        return output_path