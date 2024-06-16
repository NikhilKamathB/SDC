################################################################################################################
# Script for performing motion planning (high and low level) must be written here.
################################################################################################################


import carla
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Tuple, Type, List
from src.client import CarlaClientCLI
from src.utils.utils import plot_3d_matrix, plot_3d_roads
from src.model.enum import DistanceMetric, SearchAlgorithm

try:
    from algorithmslib import algorithms
except ImportError:
    raise ImportError("The `algorithmslib` package is not installed. Please install it using `setup.sh` script found at the root of the repository. For more details, refer to the documentation.")


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
        self.distance_metric = kwargs.get("distance_metric", DistanceMetric.EUCLIDEAN)
        self.search_algorithm = kwargs.get("search_algorithm", SearchAlgorithm.A_STAR)
        self.node_name_delimiter = kwargs.get("node_name_delimiter", "__")
        self.figaspect = kwargs.get("figaspect", 0.5)
        self.verbose = kwargs.get("verbose", True)
    
    def _get_distance_metric(self) -> Type[algorithms.DistanceMetric]:
        """
        Get the distance metric.
        Return: The distance metric.
        """
        if self.distance_metric == DistanceMetric.EUCLIDEAN.value:
            return algorithms.DistanceMetric.EUCLIDEAN
        elif self.distance_metric == DistanceMetric.MANHATTAN.value:
            return algorithms.DistanceMetric.MANHATTAN
        else:
            raise ValueError(f"Invalid distance metric: {self.distance_metric}")
    
    def _get_search_algorithm(self) -> Type[algorithms.SearchAlgorithm]:
        """
        Get the search algorithm.
        Return: The search algorithm.
        """
        if self.search_algorithm == SearchAlgorithm.BREADTH_FIRST_SEARCH.value:
            return algorithms.SearchAlgorithm.BREADTH_FIRST_SEARCH
        elif self.search_algorithm == SearchAlgorithm.DEPTH_FIRST_SEARCH.value:
            return algorithms.SearchAlgorithm.DEPTH_FIRST_SEARCH
        elif self.search_algorithm == SearchAlgorithm.UNIFORM_COST_SEARCH.value:
            return algorithms.SearchAlgorithm.UNIFORM_COST_SEARCH
        elif self.search_algorithm == SearchAlgorithm.A_STAR.value:
            return algorithms.SearchAlgorithm.A_STAR
        else:
            raise ValueError(f"Invalid search algorithm: {self.search_algorithm}")
    
    def _make_node_name(self, road_id: int, section_id: int, lane_sign: int) -> str:
        """
        Make the node name.
        Input parameters:
            - road_id: int - the road id.
            - section_id: int - the section id.
            - lane_sign: int - the lane sign.
        Return: A string containing the node name.
        """
        return f"{road_id}{self.node_name_delimiter}{section_id}{self.node_name_delimiter}{lane_sign}"

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

    def _init_graph(self, df: pd.DataFrame) -> Tuple[defaultdict, list]:
        """
        Initialize the graph for motion planning.
        Input parameters:
            - df: pd.DataFrame - the dataframe containing the map topology information.
        Return: A tuple containing the node dictionary and the edges.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing the graph for motion planning")
        df_simplified = self._get_simplified_map_data(df)
        w_start_segment = np.ndarray((0, 3)) # x, y, z - R3 space
        w_end_segment = np.ndarray((0, 3)) # x, y, z - R3 space
        simplified_start_segment = np.ndarray((0, 3)) # x, y, z - R3 space
        simplified_end_segment = np.ndarray((0, 3)) # x, y, z - R3 space
        node_dict = defaultdict(list)
        node_edges = []
        for segment in self.carla_client_cli.map_topology:
            w1, w2 = segment
            start_road_id, start_section_id, start_lane_sign = w1.road_id, w1.section_id, int(w1.lane_id > 0)
            end_road_id, end_section_id, end_lane_sign = w2.road_id, w2.section_id, int(w2.lane_id > 0)
            start_segment_df = df_simplified[(df_simplified["road_id"] == start_road_id) & (df_simplified["section_id"] == start_section_id) & (df_simplified["lane_sign"] == start_lane_sign)]
            end_segment_df = df_simplified[(df_simplified["road_id"] == end_road_id) & (df_simplified["section_id"] == end_section_id) & (df_simplified["lane_sign"] == end_lane_sign)]
            node_repr_start = self._make_node_name(start_road_id, start_section_id, start_lane_sign)
            node_repr_end = self._make_node_name(end_road_id, end_section_id, end_lane_sign)
            if node_repr_start not in node_dict:
                node_dict[node_repr_start] = np.array([start_segment_df.x, start_segment_df.y, start_segment_df.z]).T
            if node_repr_end not in node_dict:
                node_dict[node_repr_end] = np.array([end_segment_df.x, end_segment_df.y, end_segment_df.z]).T
            node_edges.append((node_repr_start, node_repr_end))
            if self.verbose:
                w_start_segment = np.vstack((w_start_segment, np.array([w1.transform.location.x, w1.transform.location.y, w1.transform.location.z])))
                w_end_segment = np.vstack((w_end_segment, np.array([w2.transform.location.x, w2.transform.location.y, w2.transform.location.z])))
                simplified_start_segment = np.vstack((simplified_start_segment, np.array([start_segment_df.x, start_segment_df.y, start_segment_df.z]).T))
                simplified_end_segment = np.vstack((simplified_end_segment, np.array([end_segment_df.x, end_segment_df.y, end_segment_df.z]).T))
        if self.verbose:
            plot_3d_matrix(w_start_segment, w_end_segment, figaspect=self.figaspect, title="Map Topology")
            plot_3d_matrix(simplified_start_segment, simplified_end_segment, figaspect=self.figaspect, title="Simplified Map Topology")
            plot_3d_roads(road1=(w_start_segment, w_end_segment), road2=(simplified_start_segment, simplified_end_segment), figaspect=self.figaspect, title="Roads")
        return (node_dict, node_edges)
        
    def _create_graph(self) -> Tuple[dict, dict, dict]:
        """
        Create the graph for motion planning.
        Return: A dictionary containing the graph, the node to index mapping and the index to node mapping.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Creating the graph for the given map to solve high level motion planning")
        df_map = self._register_map_data()
        node_dict, node_edges = self._init_graph(df_map)
        node_to_idx = {node: idx for idx, node in enumerate(node_dict.keys())}
        idx_to_node = {v: k for k, v in node_to_idx.items()}
        num_nodes = len(node_dict)
        edges = [(node_to_idx[edge[0]], node_to_idx[edge[1]]) for edge in node_edges]
        node_values = [node_dict[idx_to_node[idx]] for idx in range(num_nodes)]
        node_names = [idx_to_node[idx] for idx in range(num_nodes)]
        # These are the arguments of the C++ implemented search algorithm
        # Refer - https://github.com/NikhilKamathB/Algorithms/blob/main/include/algorithms.h
        return ({
            "num_nodes": num_nodes,
            "node_values": node_values,
            "node_names": node_names,
            "edges": edges
        }, node_to_idx, idx_to_node)

    def _plan_route(self, start_node_idx: int = 0, goal_node_idx: int = -1, auto: bool = False) -> List[carla.Waypoint]:
        """
        Plan the route for the given graph.
        Input parameters:
            - start_node_idx: int - the start node index - default is 0.
            - goal_node_idx: int - the goal node index - default is -1.
            - auto: bool - whether to run the motion planner automatically.
        Return: A list of carla waypoints.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Planning the route for the given graph")
        graph, node_to_idx, idx_to_node = self._create_graph()
        # Dealing with 3-Dimensional double precision items
        algd3 = algorithms.AlgorithmD3()
        path = algd3.search(
            **graph,
            start_node_idx=list(idx_to_node.keys())[start_node_idx],
            goal_node_idx=list(idx_to_node.keys())[goal_node_idx],
            method=self._get_search_algorithm(),
            distance_metric=self._get_distance_metric(),
            node_prefix_name="",
            bidirectional=False
        )

    def run(self, start_node_idx: int = 0, goal_node_idx: int = -1, auto: bool = False) -> None:
        """
        Run the high level motion planner.
        Input parameters:
            - start_node_idx: int - the start node index - default is 0.
            - goal_node_idx: int - the goal node index - default is -1.
            - auto: bool - whether to run the motion planner automatically.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Running the high level motion planner")
        self._plan_route()
        logger.info(f"{self.__LOG_PREFIX__}: High level motion planning completed")
        self.carla_client_cli.clear_environment()