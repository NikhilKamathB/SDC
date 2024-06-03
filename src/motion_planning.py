################################################################################################################
# Script for performing motion planning (high and low level) must be written here.
################################################################################################################


import logging
import numpy as np
import matplotlib.pyplot as plt
from src.client import CarlaClientCLI


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
        self.figsize = kwargs.get('figsize', (13, 13))
    
    def _plot_graph(self, x_seg_start: np.ndarray, x_seg_end: np.ndarray) -> None:
        """
        Plot the graph of the map.
        Input parameters:
            - x_seg_start: the starting points of the segments.
            - x_seg_end: the ending points of the segments.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Plotting the graph of the map")
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        for idx in range(x_seg_start.shape[0]):
            ax.plot([x_seg_start[idx, 0], x_seg_end[idx, 0]], [x_seg_start[idx, 1], x_seg_end[idx, 1]], [x_seg_start[idx, 2], x_seg_end[idx, 2]])
        plt.show()
    
    def _create_graph(self) -> None:
        """
        Create the graph for the motion planning.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Creating the graph for the given map to solve high level motion planning")
        topology = self.carla_client_cli.carla_map.get_topology()
        x_seg_start, x_seg_end = np.ndarray((0, 3)), np.ndarray((0, 3)) # collection of R3 vectors
        for segment in topology:
            w1, w2 = segment
            x_seg_start = np.vstack((x_seg_start, np.array([w1.transform.location.x, w1.transform.location.y, w1.transform.location.z])))
            x_seg_end = np.vstack((x_seg_end, np.array([w2.transform.location.x, w2.transform.location.y, w2.transform.location.z])))
        self._plot_graph(x_seg_start, x_seg_end)

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