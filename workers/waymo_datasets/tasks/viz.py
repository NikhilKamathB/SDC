#################################################################################################################
# Vizualization tasks - Waymo specific visualization tasks.
#################################################################################################################

import os
import logging
from pathlib import Path
from celery import shared_task


logger = logging.getLogger(__name__)


class WaymoOpenMotionDatasetViz:

    """
    Define tasks/functionalities associated with Waymo Open Motion Dataset visualization here.
    """

    __LOG_PREFIX__ = "WaymoOpenMotionDatasetViz"

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the WaymoOpenMotionDatasetViz instance.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing Waymo Open Motion Dataset Viz.")
        self.input_directory = kwargs.get("input_directory", None)
        self.output_directory = kwargs.get("output_directory", None)
        assert self.input_directory is not None, "Input directory not provided."
        assert self.output_directory is not None, "Output directory not provided."
        self.scenario_id = kwargs.get("scenario_id", None)
        self.output_filename = kwargs.get("output_filename", None)
    
    def _init_visualization(self) -> None:
        """
        Initialize visualization.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing visualization.")
        assert self.scenario_id is not None, "Scenario ID not provided."
        self.output_directory = os.path.join(self.output_directory, self.scenario_id)

    def _get_output_file_path(self, output_filename: str) -> Path:
        """
        Get output file path.
        Args:
            output_filename (str): Output filename.
        Returns:
            Path: Path to the output file.
        """
        logger.debug(f"{self.__LOG_PREFIX__}: Getting output file path.")
        return Path(os.path.join(self.output_directory, output_filename))
    
    def _make_output_directory(self) -> None:
        """
        Make output directory.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Making output directory if not exists.")
        os.makedirs(self.output_directory, exist_ok=True)
    
    def visualize(self) -> str:
        """
        Visualize Waymo Open Motion Dataset.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Visualizing Waymo Open Motion Dataset.")
        self._init_visualization()
        self._make_output_directory()
        return "Hello, World! From Viz."


@shared_task(name="viz.waymo_open_motion_dataset")
def viz_waymo_open_dataset_tf_record(*args, **kwargs):
    """Visualize Waymo Open Dataset TFRecord."""
    return WaymoOpenMotionDatasetViz(*args, **kwargs).visualize()
    

@shared_task(name="test")
def test():
    """Test task."""
    return "Hello, World!"