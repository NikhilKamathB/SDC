################################################################################################################
# Agroverse | Forecasting : Define tasks/functionalities associated with Agroverse Forecasting dataset here.
################################################################################################################

import os
import logging
from pathlib import Path
from typing import Tuple, Union
import matplotlib.pyplot as plt
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.viz.scenario_visualization import visualize_scenario
from src.agroverse.base import AV2Base


logger = logging.getLogger(__name__)


class AV2Forecasting(AV2Base):

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
        self.raw = kwargs.get("raw", True)
        self._show_pedesrian_xing = kwargs.get("show_pedesrian_xing", False)
        self._drivalble_area_alpha = kwargs.get("drivable_area_alpha", 0.5)
        self._drivable_area_color = kwargs.get("drivable_area_color", "#7A7A7A")
        self._lane_segment_style = kwargs.get("lane_segment_style", "-")
        self._lane_segment_linewidth = kwargs.get("lane_segment_linewidth", 1.0)
        self._lane_segment_alpha = kwargs.get("lane_segment_alpha", 0.5)
        self._lane_segment_color = kwargs.get("lane_segment_color", "#E0E0E0")
        self._pedestrian_crossing_style = kwargs.get("pedestrian_crossing_style", "-")
        self._pedestrian_crossing_linewidth = kwargs.get("pedestrian_crossing_linewidth", 1.0)
        self._pedestrian_crossing_alpha = kwargs.get("pedestrian_crossing_alpha", 0.5)
        self._pedestrian_crossing_color = kwargs.get("pedestrian_crossing_color", "#FF0000")
        self.static_map_file, self.scenario_file = self._get_input_file_names()
        self.scenario = scenario_serialization.load_argoverse_scenario_parquet(
            Path(os.path.join(self.input_directory, self.scenario_id, self.scenario_file))
        )
        self.static_map = ArgoverseStaticMap.from_json(
            Path(os.path.join(self.input_directory, self.scenario_id, self.static_map_file))
        )
    
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

    def _generate_scenario_video(self) -> str:
        """
        Generate scenario video for the given scenario id.
        Returns:
            str: Path to the generated scenario video.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Generating scenario video for scenario id: {self.scenario_id}")
        self._make_output_directory()
        output_path = Path(os.path.join(self.output_directory, self.output_filename))
        visualize_scenario(self.scenario, self.static_map, output_path)
        logger.info(f"{self.__LOG_PREFIX__}: Scenario video generated at: {output_path}")
        return output_path
    
    def _generate_indetail_scenario_video(self) -> str:
        """
        Generate detailed scenario video for the given scenario id.
        Returns:
            str: Path to the generated detailed scenario video.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Generating detailed scenario video for scenario id: {self.scenario_id}")
        _, ax = plt.subplots()
        self._visualize_map(
            show_pedesrian_xing=False,
            drivable_area_alpha=self._drivalble_area_alpha,
            drivable_area_color=self._drivable_area_color,
            lane_segment_style=self._lane_segment_style,
            lane_segment_linewidth=self._lane_segment_linewidth,
            lane_segment_alpha=self._lane_segment_alpha,
            lane_segment_color=self._lane_segment_color,
            pedestrian_crossing_style=self._pedestrian_crossing_style,
            pedestrian_crossing_linewidth=self._pedestrian_crossing_linewidth,
            pedestrian_crossing_alpha=self._pedestrian_crossing_alpha,
            pedestrian_crossing_color=self._pedestrian_crossing_color
        )
        plt.show()
        logger.info(f"{self.__LOG_PREFIX__}: Detailed scenario video generated.")
    
    def visualize(self) -> Union[str, None]:
        """
        Visualize motion forecasting data.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Visualizing motion forecasting data.")
        if self.output_filename is None:
            self.output_filename = f"{self.scenario_id}.mp4"
        if self.raw:
            return self._generate_scenario_video()
        return self._generate_indetail_scenario_video()