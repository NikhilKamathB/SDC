###################################################################################################
# Any utilities used for Agroverse dataset should be defined here.
# Reference:
# 1. https://github.com/argoverse/av2-api/blob/main/src/av2/datasets/motion_forecasting/viz/scenario_visualization.py
###################################################################################################

import cv2
import copy
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Sequence, Tuple, List
from matplotlib.patches import Rectangle
from av2.utils.typing import NDArrayFloat
from av2.map.map_api import ArgoverseStaticMap


def visualize_map(
        map: ArgoverseStaticMap,
        show_pedestrian_xing: bool = False,
        drivable_area_alpha: float = 0.5,
        drivable_area_color: str = "#7A7A7A",
        lane_segment_style: str = '-',
        lane_segment_linewidth: float = 1.0,
        lane_segment_alpha: float = 0.5,
        lane_segment_color: str = "#E0E0E0",
        pedestrian_crossing_style: str = '-',
        pedestrian_crossing_linewidth: float = 1.0,
        pedestrian_crossing_alpha: float = 0.5,
        pedestrian_crossing_color: str = "#FF0000"
) -> None:
    """
    Visualize the static map.
    Args:
        map (ArgoverseStaticMap): Static map instance.
        show_pedestrian_xing (bool, optional): Show pedestrian crossing. Defaults to False.
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
    # Plot drivable area
    for drivable_area in map.vector_drivable_areas.values():
        av2_plot_polygons([drivable_area.xyz],
                          alpha=drivable_area_alpha, color=drivable_area_color)
    # Plot lane segments
    for lane_segemnt in map.vector_lane_segments.values():
        av2_plot_polylines([
            lane_segemnt.left_lane_boundary.xyz,
            lane_segemnt.right_lane_boundary.xyz
        ], style=lane_segment_style, linewidth=lane_segment_linewidth, alpha=lane_segment_alpha, color=lane_segment_color)
    # Plot pedestrian crossing
    if show_pedestrian_xing:
        for pedestrian_crossing in map.vector_pedestrian_crossings.values():
            av2_plot_polylines([
                pedestrian_crossing.edge1.xyz,
                pedestrian_crossing.edge2.xyz,
            ], style=pedestrian_crossing_style, linewidth=pedestrian_crossing_linewidth, alpha=pedestrian_crossing_alpha, color=pedestrian_crossing_color)


def av2_plot_polylines(polylines: Sequence[NDArrayFloat], style: str = '-', linewidth: float = 1.0, alpha: float = 1.0, color: str = "r") -> None:
    """
    Plot polylines.
    Args:
        polylines (Sequence[NDArrayFloat]): Sequence of polylines.
        style (str, optional): Style of the line. Defaults to '-'.
        linewidth (float, optional): Width of the line. Defaults to 1.0.
        alpha (float, optional): Alpha value for the line. Defaults to 1.0.
        color (str, optional): Color of the line. Defaults to "r".
    """
    for polyline in polylines:
        plt.plot(polyline[:, 0], polyline[:, 1], style,
                 linewidth=linewidth, alpha=alpha, color=color)


def av2_plot_polygons(polygons: Sequence[NDArrayFloat], alpha: float = 1.0, color: str = "r") -> None:
    """
    Plot polygons.
    Args:
        polygons (Sequence[NDArrayFloat]): Sequence of polygons.
        alpha (float, optional): Alpha value for the polygon. Defaults to 1.0.
        color (str, optional): Color of the polygon. Defaults to "r".
    """
    for polygon in polygons:
        plt.fill(polygon[:, 0], polygon[:, 1], alpha=alpha, color=color)


def av2_plot_bbox(ax: plt.Axes, pivot_points: Tuple[float, float], heading: float, bbox_size: Tuple[float, float], color: str = 'b', rotation_point: str = "xy", zorder: float = np.inf, add_patch: bool = True) -> np.ndarray:
    """
    Plot bounding box representing the player.
    Args:
        ax (plt.Axes): Matplotlib axes instance.
        pivot_points (Tuple[float, float]): Pivot points of the bounding box.
        heading (float): Heading of the player, in radians.
        color (str): Color of the bounding box.
        bbox_size (Type[float, float]): Size of the bounding box - length and width.
        rotation_point (str, optional): Rotation point of the bounding box. Defaults to "xy".
        zorder (float, optional): Z-order of the bounding box. Defaults to np.inf.
        add_patch (bool, optional): Add patch to the axes. Defaults to True.
    Returns:
        np.ndarray: Corners of the bounding box.
    """
    bbox_length, bbox_width = bbox_size
    player_bbox = Rectangle(
        pivot_points,
        bbox_length,
        bbox_width,
        angle=np.degrees(heading),
        color=color,
        rotation_point=rotation_point,
        zorder=zorder
    )
    if add_patch:
        ax.add_patch(player_bbox)
    return player_bbox.get_corners()


def write_video(frames: List[Image.Image], video_path: str, codec: str = "mpv4", fps: int = 10, cvt_RGB2BGR: bool = True) -> str:
    """
    Given a list of frames, write a video.
    Input parameters:
        - frames: List[Image.Image] - the list of frames.
        - video_path: str - the path of the video.
        - codec: str - the codec to be used for the video.
        - fps: int - the frames per second.
        - cvt_RGB2BGR: bool - whether to convert RGB to BGR or not.
    Output:
        - str: the path of the video.
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    frame_size = frames[0].size
    out = cv2.VideoWriter(video_path, fourcc, fps=fps, frameSize=frame_size)
    for frame in frames:
        _frame = np.array(copy.deepcopy(frame))
        if cvt_RGB2BGR:
            _frame = cv2.cvtColor(_frame, cv2.COLOR_RGB2BGR)
        out.write(_frame)
    out.release()
    return video_path
