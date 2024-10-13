#################################################################################################################
# Preprocessing tasks - Waymo specific preprocessing tasks.
# Reference:
#     1. https://github.com/sshaoshuai/MTR/blob/master/mtr/datasets/waymo/data_preprocess.py
#################################################################################################################

import os
import logging
import numpy as np
import tensorflow as tf
from celery import shared_task
from typing import List, Dict, Any
from waymo_open_dataset.protos import scenario_pb2
from tasks.base import WaymoBase
from validators import ScenarioBaseModel, ScenarioInstanceBaseModel
from constants import OBJECT_TYPE, LANE_TYPE, ROAD_LINE_TYPE, POLYLINE_TYPE, ROAD_EDGE_TYPE, SIGNAL_STATE, A_MIN, A_MAX


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
        self.generate_json = kwargs.get("generate_json", False)
        self._make_output_directory()
    
    def _get_polyline_dir(self, polyline: np.ndarray) -> np.ndarray:
        """
        Get the direction of the polyline.
        Input:
            polyline: np.ndarray - (num_points, 3)
        Output:
            polyline_dir: np.ndarray - (num_points, 3)
        """
        polyline_pre = np.roll(polyline, shift=1, axis=0)
        polyline_pre[0] = polyline[0]
        diff = polyline - polyline_pre
        polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=A_MIN, a_max=A_MAX)
        return polyline_dir

    def _decode_tracks_scenario_proto(self, tracks: object) -> Dict[str, Any]:
        """
        Decode the tracks from the scenario proto.
        Input:
            tracks: object - type: google.protobuf.pyext._message.RepeatedCompositeContainer
        Output:
            track_info: Dict[str, Any] - dictionary containing the track information.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Decoding tracks from scenario proto.")
        track_info = {
            "object_id": [],
            "object_type": [],
            "trajs": []
        }
        for cur_data in tracks: # number of objects
            cur_traj = [np.array([x.center_x, x.center_y, x.center_z, x.length, x.width, x.height, x.heading, x.velocity_x, x.velocity_y, x.valid], dtype=np.float32) for x in cur_data.states]
            cur_traj = np.stack(cur_traj, axis=0) # (num_timestamp, 10)
            track_info["object_id"].append(cur_data.id)
            track_info["object_type"].append(OBJECT_TYPE[cur_data.object_type])
            track_info["trajs"].append(cur_traj)
        track_info["trajs"] = np.stack(track_info["trajs"], axis=0) # (num_objects, num_timestamp, 9)
        return track_info
    
    def _decode_map_features_from_proto(self, map_features: object) -> Dict[str, Any]:
        """
        Decode the map features from the scenario proto.
        Input:
            map_features: object - type: google.protobuf.pyext._message.RepeatedCompositeContainer
        Output:
            map_feature_info: Dict[str, Any] - dictionary containing the map feature information.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Decoding map features from scenario proto.")
        map_infos = {
            "lane": [],
            "road_line": [],
            "road_edge": [],
            "stop_sign": [],
            "crosswalk": [],
            "speed_bump": []
        }
        polylines = []
        point_cnt = 0
        for cur_data in map_features:
            cur_info = {"id": cur_data.id}
            # Lane
            if cur_data.lane.ByteSize() > 0:
                cur_info["speed_limit_mph"] = cur_data.lane.speed_limit_mph
                cur_info["type"] = LANE_TYPE[cur_data.lane.type]
                cur_info["interpolating"] = cur_data.lane.interpolating
                cur_info["entry_lanes"] = list(cur_data.lane.entry_lanes)
                cur_info["exit_lanes"] = list(cur_data.lane.exit_lanes)
                cur_info["left_boundary"] = [{
                        "start_index": x.lane_start_index,
                        "end_index": x.lane_end_index,
                        "feature_id": x.boundary_feature_id,
                        "boundary_type": ROAD_LINE_TYPE[x.boundary_type]
                    } for x in cur_data.lane.left_boundaries
                ]
                cur_info["right_boundary"] = [{
                        "start_index": x.lane_start_index,
                        "end_index": x.lane_end_index,
                        "feature_id": x.boundary_feature_id,
                        "boundary_type": ROAD_LINE_TYPE[x.boundary_type]
                    } for x in cur_data.lane.right_boundaries
                ]
                global_type = POLYLINE_TYPE[cur_info["type"]]
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.lane.polyline], axis=0)
                cur_polyline_dir = self._get_polyline_dir(cur_polyline[:, 0:3])
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
                map_infos["lane"].append(cur_info)
            # Stop Sign
            elif cur_data.stop_sign.ByteSize() > 0:
                cur_info["lane_ids"] = list(cur_data.stop_sign.lane)
                point = cur_data.stop_sign.position
                cur_info["position"] = np.array([point.x, point.y, point.z])
                global_type = POLYLINE_TYPE["TYPE_STOP_SIGN"]
                cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type]).reshape(1, 7)
                map_infos["stop_sign"].append(cur_info)
            # Speed Bump
            elif cur_data.speed_bump.ByteSize() > 0:
                global_type = POLYLINE_TYPE["TYPE_SPEED_BUMP"]
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.speed_bump.polygon], axis=0)
                cur_polyline_dir = self._get_polyline_dir(cur_polyline[:, 0:3])
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
                map_infos["speed_bump"].append(cur_info)
            # Crosswalk
            elif cur_data.crosswalk.ByteSize() > 0:
                global_type = POLYLINE_TYPE["TYPE_CROSSWALK"]
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.crosswalk.polygon], axis=0)
                cur_polyline_dir = self._get_polyline_dir(cur_polyline[:, 0:3])
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
                map_infos["crosswalk"].append(cur_info)
            # Road Edge
            elif cur_data.road_edge.ByteSize() > 0:
                cur_info["type"] = ROAD_EDGE_TYPE[cur_data.road_edge.type]
                global_type = POLYLINE_TYPE[cur_info["type"]]
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_edge.polyline], axis=0)
                cur_polyline_dir = self._get_polyline_dir(cur_polyline[:, 0:3])
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
                map_infos["road_edge"].append(cur_info)
            # Road Line
            elif cur_data.road_line.ByteSize() > 0:
                cur_info["type"] = ROAD_LINE_TYPE[cur_data.road_line.type]
                global_type = POLYLINE_TYPE[cur_info["type"]]
                cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type]) for point in cur_data.road_line.polyline], axis=0)
                cur_polyline_dir = self._get_polyline_dir(cur_polyline[:, 0:3])
                cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)
                map_infos["road_line"].append(cur_info)
            else:
                logger.warning(f"{self.__LOG_PREFIX__}: Unknown map feature - {cur_data}")
            polylines.append(cur_polyline)
            cur_info["polyline_index"] = (point_cnt, point_cnt + len(cur_polyline))
            point_cnt += len(cur_polyline)
        try:
            polylines = np.concatenate(polylines, axis=0).astype(np.float32)
        except:
            polylines = np.zeros((0, 7), dtype=np.float32)
            logger.warning(f"{self.__LOG_PREFIX__}: Found empty polylines.")
        map_infos["all_polylines"] = polylines
        return map_infos
    
    def _decode_dynamic_map_states_from_proto(self, dynamic_map_states: object) -> Dict[str, Any]:
        """
        Decode the dynamic map features from the scenario proto.
        Input:
            dynamic_map_states: object - type: google.protobuf.pyext._message.RepeatedCompositeContainer
        Output:
            dynamic_map_infos: Dict[str, Any] - dictionary containing the dynamic map feature information.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Decoding dynamic map features from scenario proto.")
        dynamic_map_infos = {
            "lane_id": [],
            "state": [],
            "stop_point": []
        }
        for cur_data in dynamic_map_states:  # (num_timestamp)
            lane_id, state, stop_point = [], [], []
            for cur_signal in cur_data.lane_states:  # (num_observed_signals)
                lane_id.append(cur_signal.lane)
                state.append(SIGNAL_STATE[cur_signal.state])
                stop_point.append([cur_signal.stop_point.x, cur_signal.stop_point.y, cur_signal.stop_point.z])
            dynamic_map_infos["lane_id"].append(np.array([lane_id]))
            dynamic_map_infos["state"].append(np.array([state]))
            dynamic_map_infos["stop_point"].append(np.array([stop_point]))
        return dynamic_map_infos

    def process_scenario_proto(self, scenario_proto_path: str) -> str:
        """
        Process a scenario proto file and return a list of dictionaries containing the scenario information.
        Input:
            scenario_proto_path: str - path to the scenario proto file.
        Output:
            output_directory: str - path to the output directory.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Processing scenario proto file - {scenario_proto_path}")
        dataset = tf.data.TFRecordDataset(scenario_proto_path, compression_type="")
        pydantic_scenario_infos = []
        for ctr, data in enumerate(dataset):
            try:
                info = {}
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(bytearray(data.numpy()))
                info["scenario_id"]: str = scenario.scenario_id
                info["timestamps_seconds"]: list = list(scenario.timestamps_seconds) # (91,)
                info["current_time_index"]: int = scenario.current_time_index # 10
                info["sdc_track_index"]: int = scenario.sdc_track_index
                info["objects_of_interest"]: list = list(scenario.objects_of_interest)
                info["tracks_to_predict"] = {
                    "track_index": [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
                    "difficulty": [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict]
                }
                track_infos = self._decode_tracks_scenario_proto(scenario.tracks)
                info["tracks_to_predict"]["object_type"] = [
                    track_infos["object_type"][idx] for idx in info["tracks_to_predict"]["track_index"]
                ]
                map_infos = self._decode_map_features_from_proto(scenario.map_features)
                dynamic_map_infos = self._decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)
                pydantic_info = {
                    "track_infos": track_infos,
                    "map_infos": map_infos,
                    "dynamic_map_infos": dynamic_map_infos
                }
                pydantic_info.update(info)
                pydantic_info = ScenarioInstanceBaseModel(**pydantic_info)
                pickle_path = self._get_output_file_path(f"{info['scenario_id']}.pkl")
                pydantic_info.save_pickle(pickle_path)
                if self.generate_json:
                    json_path = self._get_output_file_path(f"{info['scenario_id']}.json")
                    pydantic_info.save_json(json_path)
                pydantic_scenario_infos.append(pydantic_info)
            except Exception as e:
                logger.error(f"{self.__LOG_PREFIX__}: Error processing scenario proto at {scenario_proto_path} - {ctr} with: {str(e)}")
        pydantic_scenario = ScenarioBaseModel(scenarios=pydantic_scenario_infos)
        return self.output_directory
    
    def process(self) -> str:
        """
        Process the scenario proto files.
        Returns:
            output_directory: str - path to the output directory containing the processed scenario information.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Processing scenario proto file - {self.scenario}")
        scenario_proto_path = os.path.join(self.input_directory, self.scenario)
        output_directory = self.process_scenario_proto(scenario_proto_path)
        return output_directory

@shared_task(name="preprocess.waymo_open_motion_dataset")
def preprocess_waymo_open_dataset_tf_record(*args, **kwargs):
    """Preprocess Waymo Open Dataset TFRecord."""
    return WaymoOpenMotionDatasetPreprocess(*args, **kwargs).process()