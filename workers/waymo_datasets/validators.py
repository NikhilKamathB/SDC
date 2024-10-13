#############################################################################################################
# Define validators here
#############################################################################################################

import os
import json
import pickle
import numpy as np
from typing import List, Tuple, Optional
from pydantic import BaseModel, ConfigDict, field_validator


class ConfigBaseModel(BaseModel):

    """
    Base model for config.
    """

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist()
        }


class ScenarioTrackBaseModel(ConfigBaseModel):

    """
    Base model for scenario tracks.
    """

    object_id: List[int]
    object_type: List[str]
    trajs: List[np.ndarray]


class TracksToPredictBaseModel(ConfigBaseModel):

    """
    Base model for tracks to predict.
    """

    track_index: Optional[List[int]] = []
    difficulty: Optional[List[int]] = []
    object_type: Optional[List[str]] = []


class TrackBaseModel(ConfigBaseModel):

    """
    Base model for tracks.
    """

    object_id: Optional[List[int]] = []
    object_type: Optional[List[str]] = []
    trajs: Optional[np.ndarray] = None


class BaseMapFeatureBaseModel(ConfigBaseModel):

    """
    Base model for map features.
    """

    id: int
    type: Optional[int] = None
    polyline_index: Optional[Tuple[int, int]] = None


class LaneBoundaryBaseModel(ConfigBaseModel):

    """
    Base model for lane boundaries.
    """
    start_index: int
    end_index: int
    feature_id: int
    boundary_type: str


class LaneBaseModel(BaseMapFeatureBaseModel):

    """
    Base model for lanes.
    """
    speed_limit_mph: float
    interpolating: bool
    entry_lanes: List[int] = []
    exit_lanes: List[int] = []
    left_boundary: List[LaneBoundaryBaseModel] = []
    right_boundary: List[LaneBoundaryBaseModel] = []


class StopSignBaseModel(BaseMapFeatureBaseModel):

    """
    Base model for stop signs.
    """

    lane_ids: List[int]
    position: np.ndarray


class MapFeatureBaseModel(ConfigBaseModel):

    """
    Base model for map features.
    """

    lanes: List[LaneBaseModel] = []
    stop_signs: List[StopSignBaseModel] = []
    road_lines: List[BaseMapFeatureBaseModel] = []
    road_edges: List[BaseMapFeatureBaseModel] = []
    crosswalks: List[BaseMapFeatureBaseModel] = []
    speed_bumps: List[BaseMapFeatureBaseModel] = []
    all_polylines: np.ndarray = np.zeros((0, 7), dtype=np.float32)


class DynamicMapFeatureBaseModel(ConfigBaseModel):

    """
    Base model for dynamic map features.
    """

    lane_id: List[np.ndarray]
    state: List[np.ndarray]
    stop_point: List[np.ndarray]
    

class ScenarioInstanceBaseModel(ConfigBaseModel):

    """
    Base model for scenario.
    """
    scenario_id: str
    timestamps_seconds: List[float]
    current_time_index: int
    sdc_track_index: int
    objects_of_interest: Optional[List[int]] = []
    tracks_to_predict: Optional[TracksToPredictBaseModel] = None
    track_infos: Optional[TrackBaseModel] = None
    map_infos: Optional[MapFeatureBaseModel] = None
    dynamic_map_infos: Optional[DynamicMapFeatureBaseModel] = None

    def _clear_file(self, path: str) -> None:
        """
        Clear the file at the given path.
        """
        if os.path.exists(path): os.remove(path)

    def save_pickle(self, path: str) -> str:
        """
        Save the scenario instance to a pickle file.
        Args:
            path: str - path to save the pickle file.
        Returns:
            path: str - path to the saved pickle file.
        """
        self._clear_file(path)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        return path
    
    def save_json(self, path: str) -> str:
        """
        Save the scenario instance to a json file.
        Args:
            path: str - path to save the json file.
        Returns:
            path: str - path to the saved json file.
        """
        self._clear_file(path)
        with open(path, 'w') as f:
            json.dump(self.model_dump(mode="json"), f)
        return path


class ScenarioBaseModel(ConfigBaseModel):

    """
    Base model for scenarios.
    """

    scenarios: List[ScenarioInstanceBaseModel] = []