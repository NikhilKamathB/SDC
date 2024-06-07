################################################################################################################
# Script for performing motion planning (high and low level) must be written here.
################################################################################################################


import logging
import numpy as np
from src.client import CarlaClientCLI
from src.utils.utils import plot_3d_matrix


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
    
    def _create_graph(self) -> None:
        """
        Create the graph for the motion planning.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Creating the graph for the given map to solve high level motion planning")
        topology = self.carla_client_cli.carla_map.get_topology()
        x_seg_start, x_seg_end = np.ndarray((0, 3)), np.ndarray((0, 3)) # collection of R3 vectors
        for segment in topology:
            w1, w2 = segment
            print("w1: ", w1.right_lane_marking.type, w1.left_lane_marking.type, w1.transform.location, w1.transform.rotation)
            print("w2: ", w2.right_lane_marking.type, w2.left_lane_marking.type, w2.transform.location, w2.transform.rotation)
            x_seg_start = np.vstack((x_seg_start, np.array([w1.transform.location.x, w1.transform.location.y, w1.transform.location.z])))
            x_seg_end = np.vstack((x_seg_end, np.array([w2.transform.location.x, w2.transform.location.y, w2.transform.location.z])))
        if self.verbose:
            plot_3d_matrix(x_seg_start, x_seg_end, figaspect=self.figaspect)

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
        pass