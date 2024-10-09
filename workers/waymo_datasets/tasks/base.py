#################################################################################################################
# Waymo | Base class : Define base tasks/functionalities associated with Waymo dataset here.
#################################################################################################################

import os
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class WaymoBase:

    """
    Define base tasks/functionalities associated with Waymo dataset here.
    """

    __LOG_PREFIX__ = "WaymoBase"

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
    
    def _get_scenario_id_from_scenario(self, scenario: str) -> str:
        """
        Get scenario ID from scenario.
        training_tfexample.tfrecord-00000-of-01000 -> training_tfexample__tfrecord-00000-of-01000
        Input:
            scenario (str): Scenario.
        Returns:
            str: Scenario ID.
        """
        return scenario.replace('.', self.file_delimiter)
        