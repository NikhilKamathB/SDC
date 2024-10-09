#################################################################################################################
# Preprocessing tasks - Waymo specific preprocessing tasks.
# Reference:
#     1. https://github.com/sshaoshuai/MTR/blob/master/mtr/datasets/waymo/data_preprocess.py
#################################################################################################################

import os
import logging
import tensorflow as tf
from celery import shared_task
from typing import List, Dict, Any
from waymo_open_dataset.protos import scenario_pb2
from tasks.base import WaymoBase


logger = logging.getLogger(__name__)


class WaymoOpenMotionDatasetPreprocess(WaymoBase):

    """
    Define preprocessing tasks for Waymo Open Motion Dataset here.
    """

    __LOG_PREFIX__ = "WaymoOpenMotionDatasetPreprocess"

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the WaymoOpenMotionDatasetPreprocess instance.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing Waymo Open Motion Dataset Preprocess.")
        self.input_directory = kwargs.get("input_directory", None)
        self.output_directory = kwargs.get("output_directory", None)
        assert self.input_directory is not None, "Input directory not provided."
        assert self.output_directory is not None, "Output directory not provided."
        self.scenario = kwargs.get("scenario", None)
        self.output_filename = kwargs.get("output_filename", None)
        self.file_delimiter = kwargs.get("file_delimiter", "__")
        self.output_directory = os.path.join(self.output_directory, self._get_scenario_id_from_scenario(self.scenario))
        self._make_output_directory()
    
    @classmethod
    def process_scenario_proto(cls, scenario_proto_path: str) -> List[Dict[str, Any]]:
        """
        Process a scenario proto file and return a list of dictionaries containing the scenario information.
        Input:
            scenario_proto_path: str - path to the scenario proto file.
        Output:
            results: List[Dict[str, Any]] - list of dictionaries containing the scenario information.
        """
        dataset = tf.data.TFRecordDataset(scenario_proto_path, compression_type="")
        results = []
        for ctr, data in enumerate(dataset):
            try:
                info = {}
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(bytearray(data.numpy()))
                info["scenario_id"]: str = scenario.scenario_id
                info["timestamps_seconds"]: list = list(scenario.timestamps_seconds) # (91,)
                info["cuurent_time_index"]: int = scenario.current_time_index # 10
                info['sdc_track_index']: int = scenario.sdc_track_index
                info['objects_of_interest']: list = list(scenario.objects_of_interest)
                results.append(info)
            except Exception as e:
                logger.error(f"{cls.__LOG_PREFIX__}: Error processing scenario proto at {scenario_proto_path} - {ctr} with: {str(e)}")
                continue
        return results
    
    def process(self) -> List[Dict[str, Any]]:
        """
        Process the scenario proto files.
        Returns:
            results: List[Dict[str, Any]] - list of dictionaries containing the scenario information.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Processing scenario proto file - {self.scenario}")
        scenario_proto_path = os.path.join(self.input_directory, self.scenario)
        results = self.process_scenario_proto(scenario_proto_path)
        return results

@shared_task(name="preprocess.waymo_open_motion_dataset")
def preprocess_waymo_open_dataset_tf_record(*args, **kwargs):
    """Preprocess Waymo Open Dataset TFRecord."""
    return WaymoOpenMotionDatasetPreprocess(*args, **kwargs).process()
        