################################################################################################################
# Agroverse | Base class : Define base tasks/functionalities associated with Agroverse dataset here.
################################################################################################################

import logging
import numpy as np
from typing import Tuple
from av2.utils.typing import NDArrayFloat
from av2.map.map_api import ArgoverseStaticMap
from src.agroverse.utils import visualize_map


logger = logging.getLogger(__name__)


class AV2Base:

    """
    Define base tasks/functionalities associated with Agroverse dataset instance here.
    """

    __LOG_PREFIX__ = "AV2Base"
    static_map: ArgoverseStaticMap = None

    def _visualize_map(self,
                    show_pedesrian_xing: bool = False,
                    drivable_area_alpha: float = 0.5,
                    drivable_area_color: str = "#7A7A7A",
                    lane_segment_style: str = '-',
                    lane_segment_linewidth: float = 1.0,
                    lane_segment_alpha: float = 0.5,
                    lane_segment_color: str = "#E0E0E0",
                    pedestrian_crossing_style: str = '-',
                    pedestrian_crossing_linewidth: float = 1.0,
                    pedestrian_crossing_alpha: float = 0.5,
                    pedestrian_crossing_color: str = "#FF00FF"
                ) -> None:
        """
        Visualize the static map.
        Args:
            show_pedesrian_xing (bool, optional): Show pedestrian crossing. Defaults to False.
            drivable_area_alpha (float, optional): Alpha value for drivable area. Defaults to 0.5.
            drivable_area_color (str, optional): Color of drivable area. Defaults to "#7A7A7A".
            lane_segment_style (str, optional): Style of lane segment. Defaults to '-'.
            lane_segment_linewidth (float, optional): Width of lane segment. Defaults to 1.0.
            lane_segment_alpha (float, optional): Alpha value for lane segment. Defaults to 0.5.
            lane_segment_color (str, optional): Color of lane segment. Defaults to "#E0E0E0".
            pedestrian_crossing_style (str, optional): Style of pedestrian crossing. Defaults to '-'.
            pedestrian_crossing_linewidth (float, optional): Width of pedestrian crossing. Defaults to 1.0.
            pedestrian_crossing_alpha (float, optional): Alpha value for pedestrian crossing. Defaults to 0.5.
            pedestrian_crossing_color (str, optional): Color of pedestrian crossing. Defaults to "#FF0000".
        """
        logger.debug(f"{self.__LOG_PREFIX__}: Visualizing the static map.")
        # Plot drivable area
        visualize_map(
            self.static_map,
            show_pedesrian_xing=show_pedesrian_xing,
            drivable_area_alpha=drivable_area_alpha,
            drivable_area_color=drivable_area_color,
            lane_segment_style=lane_segment_style,
            lane_segment_linewidth=lane_segment_linewidth,
            lane_segment_alpha=lane_segment_alpha,
            lane_segment_color=lane_segment_color,
            pedestrian_crossing_style=pedestrian_crossing_style,
            pedestrian_crossing_linewidth=pedestrian_crossing_linewidth,
            pedestrian_crossing_alpha=pedestrian_crossing_alpha,
            pedestrian_crossing_color=pedestrian_crossing_color
        )
    
    @classmethod
    def transform_bbox(cls, current_location: NDArrayFloat, heading: float, bbox_size: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform bounding box by rotating and translating it based on the heading and current location.
        """
        logger.debug(f"{AV2Base.__LOG_PREFIX__}: Transforming bounding box.")
        bbox_length, bbox_width = bbox_size
        d = np.hypot(bbox_length, bbox_width)
        theta = np.arctan2(bbox_width, bbox_length)
        x = current_location[0] - (d/2) * np.cos(heading + theta)
        y = current_location[1] - (d/2) * np.sin(heading + theta)
        return (x, y)
