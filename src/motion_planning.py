################################################################################################################
# Script for performing motion planning (high and low level) must be written here.
################################################################################################################


import carla
import logging
import numpy as np
import pandas as pd
from typing import Tuple
from collections import defaultdict
from src.client import CarlaClientCLI
from src.utils.utils import plot_3d_matrix, plot_3d_roads


logger = logging.getLogger(__name__)


class HighLevelMotionPlanner:

    """
    Define the high level motion planner class here.
    """

    __LOG_PREFIX__ = "HighLevelMotionPlanner"

    def __init__(self, carla_client_cli: CarlaClientCLI, *args, **kwargs) -> None:
        """
        Initialize the high level motion planner.
        Input parameters:
            - carla_client_cli: the carla client command line interface that contians the carla world.
            - args and kwargs: additional arguments.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the high level motion planner")
        self.carla_client_cli = carla_client_cli
        self.figaspect = kwargs.get('figaspect', 0.5)
        self.verbose = kwargs.get('verbose', True)

    def _register_map_data(self) -> pd.DataFrame:
        """
        Register the map for the high level motion planner.
        Return: A pandas dataframe containing the topology information.
        """

        def _add_to_dict(w: carla.Waypoint) -> None:
            """
            Add the waypoint to the dictionary.
            """
            if w.id not in df_dict["id"]:
                df_dict["id"].append(w.id)
                df_dict["road_id"].append(w.road_id)
                df_dict["section_id"].append(w.section_id)
                df_dict["lane_id"].append(w.lane_id)
                df_dict["x"].append(w.transform.location.x)
                df_dict["y"].append(w.transform.location.y)
                df_dict["z"].append(w.transform.location.z)
                df_dict["roll"].append(w.transform.rotation.roll)
                df_dict["pitch"].append(w.transform.rotation.pitch)
                df_dict["yaw"].append(w.transform.rotation.yaw)
        
        logger.info(f"{self.__LOG_PREFIX__}: Registering the map for the high level motion planner")
        df_dict = defaultdict(list)
        for segment in self.carla_client_cli.map_topology:
            w1, w2 = segment
            _add_to_dict(w1)
            _add_to_dict(w2)
        return pd.DataFrame(df_dict)
    
    def _get_simplified_map_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the simplified map data - simplified topology with multiples lanes merged into one.
        Input parameters:
            - df: pd.DataFrame - the dataframe containing raw topology information.
        Return: A pandas dataframe containing the simplified topology information.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting the simplified map data for the high level motion planner")
        df["lane_sign"] = (df["lane_id"] > 0).astype(int)
        simplifiled_df = df.groupby(["road_id", "section_id", "lane_sign"]).agg(
            {
                'x': lambda x: np.mean(x),
                'y': lambda x: np.mean(x),
                'z': lambda x: np.mean(x),
                "id": lambda x: list(x)
            }
        )
        return simplifiled_df.reset_index()

    def _init_graph(self, df: pd.DataFrame) -> None:
        """
        Initialize the graph for motion planning.
        Input parameters:
            - df: pd.DataFrame - the dataframe containing the map topology information.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the graph for motion planning")
        df_simplified = self._get_simplified_map_data(df)
        w_start_segment = np.ndarray((0, 3)) # x, y, z - R3 space
        w_end_segment = np.ndarray((0, 3)) # x, y, z - R3 space
        simplified_start_segment = np.ndarray((0, 3)) # x, y, z - R3 space
        simplified_end_segment = np.ndarray((0, 3)) # x, y, z - R3 space
        for segment in self.carla_client_cli.map_topology:
            w1, w2 = segment
            w_start_segment = np.vstack((w_start_segment, np.array([w1.transform.location.x, w1.transform.location.y, w1.transform.location.z])))
            w_end_segment = np.vstack((w_end_segment, np.array([w2.transform.location.x, w2.transform.location.y, w2.transform.location.z])))
            start_road_id, start_section_id, start_lane_sign = w1.road_id, w1.section_id, int(w1.lane_id > 0)
            end_road_id, end_section_id, end_lane_sign = w2.road_id, w2.section_id, int(w2.lane_id > 0)
            start_segment_df = df_simplified[(df_simplified["road_id"] == start_road_id) & (df_simplified["section_id"] == start_section_id) & (df_simplified["lane_sign"] == start_lane_sign)]
            end_segment_df = df_simplified[(df_simplified["road_id"] == end_road_id) & (df_simplified["section_id"] == end_section_id) & (df_simplified["lane_sign"] == end_lane_sign)]
            simplified_start_segment = np.vstack((simplified_start_segment, np.array([start_segment_df.x, start_segment_df.y, start_segment_df.z]).T))
            simplified_end_segment = np.vstack((simplified_end_segment, np.array([end_segment_df.x, end_segment_df.y, end_segment_df.z]).T))
        if self.verbose:
            plot_3d_matrix(w_start_segment, w_end_segment, figaspect=self.figaspect, title="Map Topology")
            plot_3d_matrix(simplified_start_segment, simplified_end_segment, figaspect=self.figaspect, title="Simplified Map Topology")
            plot_3d_roads(road1=(w_start_segment, w_end_segment), road2=(simplified_start_segment, simplified_end_segment), figaspect=self.figaspect, title="Roads")
            
    def _create_graph(self) -> None:
        """
        Create the graph for motion planning.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Creating the graph for the given map to solve high level motion planning")
        df_map = self._register_map_data()
        self._init_graph(df_map)

    def _plan_route(self) -> None:
        """
        Plan the route for the given graph.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Planning the route for the given graph")
        pass

    def run(self) -> None:
        """
        Run the high level motion planner.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Running the high level motion planner")
        self._create_graph()
        self._plan_route()
        logger.info(f"{self.__LOG_PREFIX__}: High level motion planning completed")
        self.carla_client_cli.clear_environment()