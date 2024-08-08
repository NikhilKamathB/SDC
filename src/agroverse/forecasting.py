################################################################################################################
# Agroverse | Forecasting : Define tasks/functionalities associated with Agroverse Forecasting dataset here.
################################################################################################################

import io
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from shapely import set_precision
from matplotlib.patches import Polygon
from typing import Tuple, Union, List, Dict
from av2.map.map_api import ArgoverseStaticMap
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry.polygon import Polygon as ShapelyPolygon
from shapely import unary_union, LineString as ShapelyLineString
from av2.datasets.motion_forecasting import scenario_serialization
from shapely.geometry.multipoint import MultiPoint as ShapelyMultiPoint
from shapely.geometry.multilinestring import MultiLineString as ShapelyMultiLineString
from av2.datasets.motion_forecasting.data_schema import Track, TrackCategory, ObjectType
from av2.datasets.motion_forecasting.viz.scenario_visualization import visualize_scenario
from src.agroverse.base import AV2Base
from src.agroverse.constants import CAMERA_TYPE_NAME
from src.agroverse.utils import av2_plot_polylines, av2_plot_bbox
from src.utils.utils import write_video, read_yaml_file, bilinear_interpolate
from src.agroverse.model.validators import InternalCameraMountResponse, Vehicle as AV2Vehicle


logger = logging.getLogger(__name__)


class AV2Forecasting(AV2Base):

    """
    Define tasks/functionalities associated with Agroverse Forecasting dataset instance here.
    Notes:
    1. `tracks` and `actors` are used interchangeably.
    """

    __LOG_PREFIX__ = "AV2Forecasting"
    _EPSILON = 1e-10
    # Constants as per the agroverse v2 establishment
    _OBSERVATION_DURATION_TIMESTEPS = 50
    _PREDICTION_DURATION_TIMESTEPS = 60
    _STATIC_OBJECT_TYPES = {
        ObjectType.STATIC,
        ObjectType.BACKGROUND,
        ObjectType.CONSTRUCTION,
        ObjectType.RIDERLESS_BICYCLE,
    }
    # Default configurations
    _DEFAULT_ACTOR_PATH_ALPHA = 1.0
    _DRIVABLE_AREA_ALPHA = _LANE_SEGMENT_ALPHA = _PED_XING_ALPHA = 0.5
    _DEFAULT_ACTOR_STYLE = "o"
    _LANE_SEGMENT_STYLE = _PED_XING_STYLE = _DEFAULT_ACTOR_PATH_STYLE = "-"
    _LANE_SEGMENT_LINEWIDTH = _PED_XING_LINEWIDTH = _DEFAULT_ACTOR_PATH_LINEWIDTH = 1.0
    _DEFAULT_ACTOR_MARKERSIZE = 4
    _ESTIMATED_VEHICLE_SIZE = [4.0, 2.0] # Length, Width
    _ESTIMATED_AVEHICLE_SIZE = [4.5, 2.5] # Length, Width
    _ESTIMATED_CYCLIST_SIZE = [2.0, 0.7] # Length, Width
    _SENSOR_CAMERA_SIZE = [1.0, 0.1] # Length, Width
    _DRIVABLE_AREA_COLOR = "#7A7A7A"
    _LANE_SEGMENT_COLOR = "#E0E0E0"
    _PED_XING_COLOR = "#FF00FF"
    _DEFAULT_ACTOR_COLOR = "#D3E8EF"
    _TRACK_COLORS = {
        TrackCategory.TRACK_FRAGMENT: "#FFEE00",
        TrackCategory.UNSCORED_TRACK: "#00FFFF",
        TrackCategory.SCORED_TRACK: "#00FF00",
        TrackCategory.FOCAL_TRACK: "#FF9900",
    }
    _DEFAULT_FONT_SIZE = 5
    _DEFAULT_FONT_COLOR = "black"
    _DEFAULT_FONT_X_OFFSET = 0.25
    _DEFAULT_FONT_Y_OFFSET = 0.25
    _DEFAULT_SHAPELY_PRECISION = 1e-6
    _OCCLUDED_BOUNDARY_EDGE_COLOR = "#000000"
    _OCCLUDED_BOUNDARY_FACE_COLOR = "#000000"
    _OCCLUDED_BOUNDARY_ALPHA = 0.25
    # AV - Autonomous Vehicle
    _AV_ID = "AV"
    _AV_COLOR = _AV_PATH_COLOR = "#FF0000"
    _AV_CAMERA_COLOR = "#000000"
    _AV_CAMERA_COVERAGE_COLOR = "#FFAAAA"
    _AV_CAMERA_COVERAGE_ALPHA = 0.5
    _AV_CAMERA_COVERAGE_LINEWIDTH = 2.0
    _AV_CAMERA_COVERAGE_STYLE = "-"
    # Focal agent
    _FOCAL_AGENT_COLOR = _TRACK_COLORS[TrackCategory.FOCAL_TRACK]

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize Agroverse Forecasting dataset instance.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing Agroverse Forecasting - av2 - dataset instance.")
        self.input_directory = kwargs.get("input_directory", None)
        assert self.input_directory is not None, "Input directory not provided."
        self.output_directory = kwargs.get("output_directory", None)
        assert self.output_directory is not None, "Output directory not provided."
        self.scenario_id = kwargs.get("scenario_id", None)
        assert self.scenario_id is not None, "Scenario ID not provided."
        self.output_filename = kwargs.get("output_filename", None)
        self.raw = kwargs.get("raw", True)
        self.av_configuration_path = kwargs.get("av_configuration_path", "./data/config/agroverse/vehicle0.yaml")
        self.show_pedestrian_xing = kwargs.get("show_pedestrian_xing", False)
        self.plot_occlusions = kwargs.get("plot_occlusions", True)
        self.codec = kwargs.get("codec", "mp4v")
        self.fps = kwargs.get("fps", 10)
        self.static_map_file, self.scenario_file = self._get_input_file_names()
        self.scenario = scenario_serialization.load_argoverse_scenario_parquet(
            Path(os.path.join(self.input_directory, self.scenario_id, self.scenario_file))
        )
        self.static_map = ArgoverseStaticMap.from_json(
            Path(os.path.join(self.input_directory, self.scenario_id, self.static_map_file))
        )
        self.av_configuration = self._get_av_configuration()
    
    def _make_output_directory(self) -> None:
        """
        Make output directory.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Making output directory if not exists.")
        os.makedirs(self.output_directory, exist_ok=True)
    
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
    
    def _get_input_file_names(self) -> Tuple[str, str]:
        """
        Get input file names - static map and scenario.
        As per the agroverse v2 dataset the input directory should contain two files:
        1. log_map_archive_<scenario_id>.json   # static map
        2. scenario_<scenario_id>.parquet    # scenario file
        Returns:
            Tuple[str, str]: Tuple containing static map and scenario file names.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting input file names.")
        static_map_file = f"log_map_archive_{self.scenario_id}.json"
        scenario_file = f"scenario_{self.scenario_id}.parquet"
        return static_map_file, scenario_file

    def _get_av_configuration(self) -> AV2Vehicle:
        """
        Get AV - Autonomous Vehicle configuration.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting AV - Autonomous Vehicle configuration.")
        if os.path.exists(self.av_configuration_path):
            return AV2Vehicle(**read_yaml_file(self.av_configuration_path))
        return AV2Vehicle()
        
    def _generate_scenario_video(self, output_filename: str) -> str:
        """
        Generate scenario video for the given scenario id.
        Args:
            output_filename (str): Output filename.
        Returns:
            str: Path to the generated scenario video.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Generating scenario video for scenario id: {self.scenario_id}")
        output_path = self._get_output_file_path(output_filename)
        visualize_scenario(self.scenario, self.static_map, output_path)
        logger.info(f"{self.__LOG_PREFIX__}: Scenario video generated at: {output_path}")
        return output_path
    
    def _get_timesteps(self, track: Track, timestep: int = None) -> np.ndarray:
        """
        Get timesteps from the scenario.
        Args:
            track (Track): Track instance.
            timestep (int, optional): Timestep. Defaults to None.
        Returns:
            np.ndarray: Array containing timesteps.
        """
        logger.debug(f"{self.__LOG_PREFIX__}: Getting timesteps from the scenario for track id: {track.track_id}")
        return np.array([object_state.timestep for object_state in track.object_states if timestep is not None and object_state.timestep <= timestep])

    def _get_actor_states(self, track: Track, timestep: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get actor states from the scenario.
        Args:
            track (Track): Track instance.
            timestep (int, optional): Timestep. Defaults to None.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing actor positions, actor headings, actor velocities.
        """
        logger.debug(f"{self.__LOG_PREFIX__}: Getting actor states from the scenario for track id: {track.track_id}")
        actor_timesteps = self._get_timesteps(track, timestep)
        if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
            return (np.array([]), np.array([]), np.array([]))
        actor_positions = np.array([list(object_state.position) for object_state in track.object_states if timestep is not None and object_state.timestep <= timestep])
        actor_headings = np.array([object_state.heading for object_state in track.object_states if timestep is not None and object_state.timestep <= timestep])
        actor_velocities = np.array([list(object_state.velocity) for object_state in track.object_states if timestep is not None and object_state.timestep <= timestep])
        return (actor_positions, actor_headings, actor_velocities)

    def _mount_sensors(self, ax: plt.Axes, actor_bbox: np.ndarray, actor_heading: np.ndarray) -> List[Union[InternalCameraMountResponse]]:
        """
        Mount sensors on the actor.
        Args:
            ax (plt.Axes): Matplotlib axes.
            actor_bbox (Bbox): Actor bounding box coordinates.
            actor_heading (np.ndarray): Actor heading.
        Returns:
            List[Union[InternalCameraMountResponse]]: List of sensors and their bounds.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Mounting sensors on the actor.")
        sensor_list = []
        # Mount cameras
        if self.av_configuration.sensors is not None and self.av_configuration.sensors.cameras:
            for idx, camera in enumerate(self.av_configuration.sensors.cameras):
                yaw, _, _ = camera.rotation.get_rotation_radians()
                x, y = camera.location.x, camera.location.y
                mount_point = bilinear_interpolate(patch=actor_bbox, x_frac=x, y_frac=y)
                camera_bbox = av2_plot_bbox(
                    ax=ax,
                    pivot_points=mount_point,
                    heading=actor_heading + yaw,
                    color=self._AV_CAMERA_COLOR,
                    bbox_size=self._SENSOR_CAMERA_SIZE
                )
                sensor_list.append(InternalCameraMountResponse(
                    id=f"camera_{idx}",
                    bounds=camera_bbox,
                    camera=camera
                ))
        return sensor_list
    
    def _get_angle_and_unit_vector(self, point1: np.ndarray, point2: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Get angle and unit vector given two points - as measured from point1 to point2.
        Args:
            point1 (np.ndarray): Point 1 - shape (2,).
            point2 (np.ndarray): Point 2 - shape (2,).
        Returns:
            Tuple[float, np.ndarray]: Tuple containing normalized angle (in radians) and unit vector.
        """
        logger.debug(f"{self.__LOG_PREFIX__}: Getting angle and unit vector given two points.")
        delta = point2 - point1
        theta = np.arctan2(delta[1], delta[0])
        x_hat, y_hat = np.cos(theta), np.sin(theta)
        return (theta, np.array([x_hat, y_hat]))
    
    def _get_line_from_point_and_angle(self, ax: plt.Axes, point: np.ndarray, theta: float) -> np.ndarray:
        """
        Given a point and a reference angle, get a line passing through this point keeping direction in mind.
        Extrimities of the line are the limits of the plot.
        Note:
            1. For the given point and angle, we will have only one point of intersection with the plot limits if the direction is taken into account.
               This is exactly what we will be using to span the line from the point, given the angle, to the plot limits.
            2. Reference: https://tutorial.math.lamar.edu/classes/calciii/eqnsoflines.aspx
        Args:
            ax (plt.Axes): Matplotlib axes.
            point (np.ndarray): Point - shape (2,).
            theta (float): Angle in radians.
        Returns:
            np.ndarray: Point - the other end of the line.
        """
        def _get_vertical_intersction(x: float) -> np.ndarray:
            """
            Check for vertical intersection - equation of the line: X = x
            We need to find `Y`.
            Args:
                x (float): X-coordinate.
            Returns:
                np.ndarray: Tuple containing intersection points.
            """
            t = (x - point[0].item()) / np.cos(theta) # Against unit vector
            y_new = point[1].item() + t * np.sin(theta)
            return np.array([x, y_new])
        
        def _get_horizontal_intersction(y) -> np.ndarray:
            """
            Check for horizontal intersection - equation of the line: Y = y
            We need to find `X`.
            Args:
                y (float): Y-coordinate.
            Returns:
                np.ndarray: Tuple containing intersection points.
            """
            t = (y - point[1].item()) / np.sin(theta) # Against unit vector
            x_new = point[0].item() + t * np.cos(theta)
            return np.array([x_new, y])
        
        logger.debug(f"{self.__LOG_PREFIX__}: Getting line from a given point and angle.")
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        x_hat, y_hat = np.cos(theta), np.sin(theta)
        candidates = []
        # Check for intersection with left axis: x = xmin
        x_dash, y_dash = _get_vertical_intersction(xmin)
        if ymin <= y_dash <= ymax:
            candidates.append((x_dash, y_dash))
        # Check for intersection with right axis: x = xmax
        x_dash, y_dash = _get_vertical_intersction(xmax)
        if ymin <= y_dash <= ymax:
            candidates.append((x_dash, y_dash))
        # Check for intersection with bottom axis: y = ymin
        x_dash, y_dash = _get_horizontal_intersction(ymin)
        if xmin <= x_dash <= xmax:
            candidates.append((x_dash, y_dash))
        # Check for intersection with top axis: y = ymax
        x_dash, y_dash = _get_horizontal_intersction(ymax)
        if xmin <= x_dash <= xmax:
            candidates.append((x_dash, y_dash))
        # Clean candidates to retain lines along theta
        lines = []
        for candidate in candidates:
            _, (x_dash_hat, y_dash_hat) = self._get_angle_and_unit_vector(point, candidate)
            if np.isclose(np.dot([x_hat, y_hat], [x_dash_hat, y_dash_hat]), 1.0):
                lines.append(candidate)
        if not lines or len(lines) > 1:
            logger.warning(f"{self.__LOG_PREFIX__}: Multiple lines found for the given point and angle. Selecting the first line.")
        # Plot line
        point_2 = lines[0]
        plt.plot(
            [point[0], point_2[0]],
            [point[1], point_2[1]], 
            color=self._AV_CAMERA_COVERAGE_COLOR,
            linestyle=self._AV_CAMERA_COVERAGE_STYLE,
            linewidth=self._AV_CAMERA_COVERAGE_LINEWIDTH,
        )
        plt.text(
            point_2[0] + self._DEFAULT_FONT_X_OFFSET,
            point_2[1] + self._DEFAULT_FONT_Y_OFFSET,
            f"({point_2[0]:.2f}, {point_2[1]:.2f})",
            fontsize=self._DEFAULT_FONT_SIZE,
            color=self._DEFAULT_FONT_COLOR
        )
        return np.array(point_2)
    
    def _get_limits(self, ax: plt.Axes) -> List[np.ndarray]:
        """
        Get extreme points of the coverage.
        Args:
            ax (plt.Axes): Matplotlib axes.
        Returns:
            List[np.ndarray]: List containing extreme points.
        """
        logger.debug(f"{self.__LOG_PREFIX__}: Getting coverage extreme points.")
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        extreme_points = [
            np.array([xmin, ymin]),
            np.array([xmax, ymin]),
            np.array([xmax, ymax]),
            np.array([xmin, ymax])
        ]
        return extreme_points

    def _arrange_polygon_bounds(self, polygon_bounds: List[np.ndarray], centroid: np.ndarray = None) -> List[np.ndarray]:
        """
        Arrange polygon bounds/points in anti-clockwise order.
        Args:
            polygon_bounds (List[np.ndarray]): List containing polygon bounds.
            centroid (np.ndarray, optional): Centroid of the polygon. Defaults to None.
        Returns:
            List[np.ndarray]: List containing polygon bounds in anti-clockwise order.
        """
        logger.debug(f"{self.__LOG_PREFIX__}: Arranging polygon bounds in anti-clockwise order.")
        polygon = np.array(polygon_bounds)
        if centroid is None:
            centroid = np.mean(polygon, axis=0)
        angles = np.arctan2(polygon[:, 1] - centroid[1], polygon[:, 0] - centroid[0])
        return [point for _, point in sorted(zip(angles, polygon), key=lambda x: x[0])]

    def _get_polygon_bounds_between_angle(self, points: np.ndarray, pivot_point: np.ndarray, upper_bound_point: np.ndarray, lower_bound_point: np.ndarray) -> List[np.ndarray]:
        """
        Get polygon bounds that lie between two vectors formed by the pivot point and the upper/lower bound points.
        Args:
            points (np.ndarray): Points - polygon bounds.
            pivot_point (np.ndarray): Pivot point.
            upper_bound_point (np.ndarray): Upper bound point.
            lower_bound_point (np.ndarray): Lower bound point.
        Returns:
            List[np.ndarray]: List containing polygon bounds.
        """
        logger.debug(f"{self.__LOG_PREFIX__}: Getting polygon bounds between two angles.")
        # Get angular bounds
        u_theta, _ = self._get_angle_and_unit_vector(pivot_point, upper_bound_point)
        l_theta, _ = self._get_angle_and_unit_vector(pivot_point, lower_bound_point)
        polygon_vertex = [pivot_point, lower_bound_point]
        for point in points:
            theta, _ = self._get_angle_and_unit_vector(pivot_point, point)
            if (
                np.sign(u_theta) == np.sign(l_theta)
                or (np.sign(l_theta) == -1 and np.sign(u_theta) == 1)
                ) and l_theta <= theta <= u_theta:
                polygon_vertex.append(point)
            elif np.sign(l_theta) == 1 and np.sign(u_theta) == -1:
                theta_dash = theta % (2 * np.pi) if theta < 0 else theta
                u_theta_dash = u_theta % (2 * np.pi)
                if l_theta <= theta_dash <= u_theta_dash:
                    polygon_vertex.append(point)
        polygon_vertex.append(upper_bound_point)
        return polygon_vertex

    def _get_sensor_coverage(self, ax: plt.Axes, pivot_point: np.ndarray, upper_bound_point: np.ndarray, lower_bound_point: np.ndarray) -> np.ndarray:
        """
        For a given sensor, get its coverage based on its fov and position - occlusion not considered.
        Args:
            ax (plt.Axes): Matplotlib axes.
            pivot_point (np.ndarray): Pivot point.
            upper_bound_point (np.ndarray): Upper bound point.
            lower_bound_point (np.ndarray): Lower bound point.
        Returns:
            np.ndarray: Sensor coverage - a polygon.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting sensor coverage.")
        extreme_points = self._get_limits(ax)
        polygon_vertex = self._arrange_polygon_bounds(
            self._get_polygon_bounds_between_angle(extreme_points, pivot_point, upper_bound_point, lower_bound_point)
        )
        return np.array(polygon_vertex)

    def _get_actors_in_sensor_coverage(self, sensor_coverage: ShapelyPolygon, timestep: int) -> Dict[ShapelyPolygon, np.ndarray]:
        """
        Get actors in the sensor coverage.
        Args:
            sensor_coverage (ShapelyPolygon): Sensor coverage.
            upper_bound_point (np.ndarray): Upper bound point.
            lower_bound_point (np.ndarray): Lower bound point.
            timestep (int): Timestep.
        Returns:
            Dict[ShapelyPolygon, np.ndarray]: Dictionary containing actors in the sensor coverage.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting actors in the sensor coverage.")
        actors = dict()
        for track in self.scenario.tracks:
            actor_positions, actor_headings, _ = self._get_actor_states(track, timestep)
            if actor_positions.shape[0] == 0 or actor_headings.shape[0] == 0:
                continue
            if track.track_id == self._AV_ID:
                continue
            if track.object_type == ObjectType.VEHICLE:
                bbox = tuple(self._ESTIMATED_VEHICLE_SIZE)
            elif track.object_type == ObjectType.CYCLIST or track.object_type == ObjectType.MOTORCYCLIST:
                bbox = tuple(self._ESTIMATED_CYCLIST_SIZE)
            else:
                continue
            # Get corners of the rectangle, moving anti-clockwise from (x0, y0)
            bbox_bounds = av2_plot_bbox(
                    ax=None,
                    pivot_points=AV2Base.transform_bbox(
                        ref_location=actor_positions[-1],
                        heading=actor_headings[-1],
                        bbox_size=bbox
                    ),
                    heading=actor_headings[-1],
                    bbox_size=bbox,
                    add_patch=False
            )
            # Check if the actor is within the sensor coverage
            bbox_polygon = ShapelyPolygon(bbox_bounds)
            intersections = sensor_coverage.intersection(bbox_polygon)
            if intersections.is_empty:
                continue
            else:
                actors[bbox_polygon] = np.array([point for point in intersections.exterior.coords])
        return actors
    
    def _get_actors_extrimities(self, pivot_point: np.ndarray, actors: Dict[ShapelyPolygon, np.ndarray]) -> Dict[ShapelyPolygon, Tuple[int, int]]:
        """
        Get actors extrimities from a given pivot point.
        Args:
            pivot_point (np.ndarray): Pivot point.
            actors (Dict[ShapelyPolygon, np.ndarray]): Dictionary containing actors.
        Returns:
            Dict[ShapelyPolygon, Tuple[int, int]]: Dictionary containing actors extrimities - tuple containing indices of extrimities.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting actors extrimities.")
        extrimities = dict()
        for actor, vertices in actors.items():
            bbox = vertices[:-1, :] # Last point is the same as the first point
            bbox_vec = bbox - pivot_point
            vec_projection = np.tril(np.dot(bbox_vec, bbox_vec.T))
            vec_magnitude = np.linalg.norm(bbox_vec, axis=1)[:, np.newaxis]
            vec_magnitude_combination = np.tril(vec_magnitude * vec_magnitude.T)
            vec_magnitude_combination = np.where(vec_magnitude_combination == 0, self._EPSILON, vec_magnitude_combination)
            theta = np.tril(np.arccos(np.clip((vec_projection / vec_magnitude_combination), -1.0, 1.0))) # Domain of arccos is [-1, 1]
            mask = np.full(theta.shape, -np.inf)
            mask[np.tril_indices(theta.shape[0])] = theta[np.tril_indices(theta.shape[0])]
            np.fill_diagonal(mask, -np.inf)
            theta_normalized = np.where(np.logical_and(mask < 0, mask != -np.inf), mask % (2 * np.pi), mask)
            idx = np.unravel_index(np.argmax(theta_normalized), theta_normalized.shape)
            extrimities[actor] = idx # Item containing indices of extrimities - (idx1, idx2)
        return extrimities
    
    def _get_visibility_region(self, coverage_polygon: ShapelyPolygon, pivot_point: np.ndarray, actors: Dict[ShapelyPolygon, np.ndarray], extrimities: Dict[ShapelyPolygon, Tuple[int, int]]) -> Dict[ShapelyPolygon, Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Get visibility region for the sensor.
        
        To solve this we follow the below steps:
        1. For all extreme points of the actors, compute the Hausdorf distance form this point to the polygon bounds.
           This will give us the longest distance from the point to the polygon bounds.
        2. Use this information to create a line segment from pivot point to this extreme point and extrapolate it based on the above computed distance.
        3. Check for the first intersection of this line segment with the polygon.
        4. Get all visible bounds of the actor.
        5. Construct a polygon using the above information. This will give us the occluded region.
        6. Negate the above information to get the visible region.
        
        To get the occluded region we need following points that would be defining the polygon:
        1. Extreme points of the actor
        2. Intersection points of the extreme rays with the coverage limits
        3. Leftover bounds of the sensor coverage limits
        4. Pivot point
        Reference
            https://cgm.cs.mcgill.ca/~godfried/teaching/cg-projects/98/normand/main.html
        Args:
            coverage_polygon (ShapelyPolygon): Sensor coverage polygon.
            pivot_point (np.ndarray): Pivot point.
            actors (Dict[ShapelyPolygon, np.ndarray]): Dictionary containing actors.
            extrimities (Dict[ShapelyPolygon, Tuple[float, float]]): Dictionary containing actors extrimities.
        Returns:
            Dict[ShapelyPolygon, Tuple[List[np.ndarray], List[np.ndarray]]]: Dictionary containing the actor and its corresponding occluded and visible regions when viewed from the sensor.
        """

        def _get_line_polygon_intersection(point: np.ndarray) -> np.ndarray:
            """
            Get line-polygon intersection with the line fromed by the pivot point and the extreme point upon extraploation.
            Args:
                point (np.ndarray): Point - extreme point.
            Returns:
                np.ndarray: Intersection point with the polygon.
            """
            logger.debug(f"{self.__LOG_PREFIX__}: Getting extreme line-polygon intersection for the sensor/actor.")
            point = ShapelyPoint(point)
            # Compute the Hausdorf distance from the point to the polygon bounds
            hausdorf_distance = point.hausdorff_distance(coverage_polygon)
            # Create a line segment from the pivot point to the reference point and extrapolate it
            _, u_vec = self._get_angle_and_unit_vector(pivot_point, np.array([point.x, point.y]))
            extended_line = ShapelyLineString(
                [
                    pivot_point,
                    np.array([point.x, point.y]) + u_vec * hausdorf_distance
                ]
            )
            intersection = set_precision(extended_line, self._DEFAULT_SHAPELY_PRECISION).intersection(set_precision(coverage_polygon, self._DEFAULT_SHAPELY_PRECISION))
            intersection_point = None
            if intersection.is_empty:
                logger.warning(f"{self.__LOG_PREFIX__}: No intersection found for the line-polygon. Actor found in the environment but was unable to compute the intersection.")
            else:
                if isinstance(intersection, ShapelyMultiPoint) or isinstance(intersection, ShapelyMultiLineString):
                    intersection_point = np.array([
                        geom.coords for geom in intersection.geoms
                    ]).reshape(-1, 2) # (n, 1, 2) -> (n, 2) because on shapely convention
                else:
                    intersection_point = np.array(intersection.coords) # (n, 2)
                intersection_point_distance = np.linalg.norm(intersection_point - pivot_point, axis=1)
                intersection_point = intersection_point[~(np.isclose(intersection_point_distance, 0))]
                if intersection_point.size == 0:
                    intersection_point = None
                else:
                    if len(intersection_point.shape) == 1:
                        intersection_point = intersection_point[None, :]
                    if intersection_point.shape[0] > 1:
                        # Get the closest intersection point to the pivot point
                        intersection_point = intersection_point[np.argmax(np.linalg.norm(intersection_point - pivot_point, axis=1))]
            if intersection_point is None:
                logger.warning(f"{self.__LOG_PREFIX__}: Intersection point is None. Setting it to the reference point.")
                intersection_point = np.array(point.coords)
            return np.squeeze(intersection_point)
        
        def _get_actor_bbox_contribution(bbox: np.ndarray, extrimities_idx: Tuple[int, int]) -> List[np.ndarray]:
            """
            Get actor bbox contribution for the occluded region.
            Args:
                bbox (np.ndarray): Actor bounding box.
                extrimities_idx (Tuple[int, int]): Tuple containing indices of extrimities.
            Returns:
                List[np.ndarray]: List containing actor bbox contribution.
            """
            logger.debug(f"{self.__LOG_PREFIX__}: Getting actor bbox contribution for the occluded region.")
            actor_polygon = ShapelyPolygon(bbox.tolist())
            visible_polygon = ShapelyPolygon(bbox[extrimities_idx, :].tolist() + [pivot_point.tolist()])
            intersection = actor_polygon.intersection(visible_polygon)
            if isinstance(intersection, ShapelyPolygon):
                intersection = list(np.array(intersection.exterior.coords)[:-1, :]) # Last point is the same as the first point
            else:
                intersection = list(np.array(intersection.coords)) 
            return intersection
            
        logger.info(f"{self.__LOG_PREFIX__}: Getting visibility region for the sensor.")
        visibility_dict = {}
        for actor, bbox in actors.items():
            p1, p2 = bbox[extrimities[actor], :].tolist()
            _, p1_u_vec = self._get_angle_and_unit_vector(pivot_point, p1)
            _, p2_u_vec = self._get_angle_and_unit_vector(pivot_point, p2)
            if np.sign(np.cross(p1_u_vec, p2_u_vec)) >= 0:
                lower_bound, upper_bound = p1, p2
            else:
                lower_bound, upper_bound = p2, p1
            lower_bound_intersection = _get_line_polygon_intersection(lower_bound)
            upper_bound_intersection = _get_line_polygon_intersection(upper_bound)
            coverage_polygon_bounds = np.array(coverage_polygon.exterior.coords)[:-1, :] # Last point is the same as the first point
            included_bounds = self._get_polygon_bounds_between_angle(coverage_polygon_bounds, np.mean(bbox, axis=0), upper_bound_intersection, lower_bound_intersection)
            actor_bbox_boundary = _get_actor_bbox_contribution(bbox, extrimities[actor])
            occluded_polygon = self._arrange_polygon_bounds(
                [lower_bound_intersection, upper_bound_intersection] \
                + included_bounds \
                + actor_bbox_boundary
            )
            visible_polygon = self._arrange_polygon_bounds(
                actor_bbox_boundary + [pivot_point]
            )
            visibility_dict[actor] = (occluded_polygon, visible_polygon)
        return visibility_dict
    
    def _get_consolidated_occluded_region(self, visibility_dict: Dict[ShapelyPolygon, Tuple[List[np.ndarray], List[np.ndarray]]]) -> List[ShapelyPolygon]:
        """
        Given the visibility information, consolidate the occluded region.
        Args:
            visibility_dict (Dict[ShapelyPolygon, Tuple[List[np.ndarray], List[np.ndarray]]]): Dictionary containing the actor and its corresponding occluded and visible regions when viewed from the sensor.
        Returns:
            List[ShapelyPolygon]: List containing the consolidated occluded region.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting consolidated occluded region.")
        occluded_region = []
        for _, (occluded_polygon, _) in visibility_dict.items():
            occluded_region.append(ShapelyPolygon(occluded_polygon))
        consolidated_occluded_region = unary_union(occluded_region)
        if consolidated_occluded_region.is_empty:
            return []
        if isinstance(consolidated_occluded_region, ShapelyPolygon):
            return [consolidated_occluded_region]
        return list(consolidated_occluded_region.geoms)

    def _get_occluded_region(self, pivot_point: np.ndarray, sensor_coverage: np.ndarray, timestep: int) -> List[ShapelyPolygon]:
        """
        Plot visible region for the sensor - occlusion considered.
        Args:
            pivot_point (np.ndarray): Pivot point.
            sensor_coverage (np.ndarray): Sensor coverage - a polygon.
            timestep (int): Timestep.
        Returns:
            List[ShapelyPolygon]: List containing the occluded region.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting occluded region for the sensor.")
        sensor_coverage_polygon = ShapelyPolygon(sensor_coverage)
        # Get all actors in the sensor coverage
        actors = self._get_actors_in_sensor_coverage(sensor_coverage_polygon, timestep)
        # Get actors extrimities between and beyond which the sensor cannot see
        actors_extrimities = self._get_actors_extrimities(pivot_point, actors)
        # Using the above information identify the occluded region and plot/consider visible region
        visibility_dict = self._get_visibility_region(sensor_coverage_polygon, pivot_point, actors, actors_extrimities)
        # Put all the information together
        occluded_polygons = self._get_consolidated_occluded_region(visibility_dict)
        return occluded_polygons
            
    def _plot_coverage(self, ax: plt.Axes, pivot_point: np.ndarray, upper_bound_point: np.ndarray, lower_bound_point: np.ndarray, timestep: int) -> Tuple[List[np.ndarray], List[ShapelyPolygon]]:
        """
        Plot coverage.
        Args:
            ax (plt.Axes): Matplotlib axes.
            pivot_point (np.ndarray): Pivot point.
            upper_bound_point (np.ndarray): Upper bound point.
            lower_bound_point (np.ndarray): Lower bound point.
            timestep (int): Timestep.
        Returns:
            Tuple[List[np.ndarray], List[ShapelyPolygon]]: List containing polygon vertices representing sensor coverage and occluded region.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Plotting coverage.")
        polygon_vertices = self._get_sensor_coverage(ax, pivot_point, upper_bound_point, lower_bound_point)
        polygon = Polygon(polygon_vertices, edgecolor=self._AV_CAMERA_COVERAGE_COLOR, facecolor=self._AV_CAMERA_COVERAGE_COLOR, alpha=self._AV_CAMERA_COVERAGE_ALPHA)
        ax.add_patch(polygon)
        occluded_region_polygons = self._get_occluded_region(pivot_point, polygon_vertices, timestep)
        for occluded_region_polygon in occluded_region_polygons:
            polygon = Polygon(np.array(occluded_region_polygon.exterior.coords), edgecolor=self._OCCLUDED_BOUNDARY_EDGE_COLOR, facecolor=self._OCCLUDED_BOUNDARY_FACE_COLOR, alpha=self._OCCLUDED_BOUNDARY_ALPHA)
            ax.add_patch(polygon)
        return polygon_vertices

    def _cast_rays(self, ax: plt.Axes, actor_heading: float, sensor_list: List[Union[InternalCameraMountResponse]], timestep: int) -> None:
        """
        Cast rays from sensors.
        Args:
            ax (plt.Axes): Matplotlib axes.
            actor_bbox (np.ndarray): Actor bounding box coordinates.
            actor_heading (float): Actor heading.
            sensor_list (List[Union[InternalCameraMountResponse]]): List of sensors.
            timestep (int): Timestep.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Casting rays from sensors.")
        for sensor in sensor_list:
            if sensor.type == CAMERA_TYPE_NAME:
                # Get sensor properties
                fov = sensor.camera.get_fov_radians()
                cp = sensor.get_mid_bounds()
                yaw, _, _ = sensor.camera.rotation.get_rotation_radians()
                # Set upper and lower angular bounds
                upper_fov_bound = actor_heading + yaw + (fov / 2)
                lower_fov_bound = actor_heading + yaw - (fov / 2)
                # Get coverage bounds
                lower_coverage_bound_point = self._get_line_from_point_and_angle(ax, cp, lower_fov_bound)
                upper_coverage_bound_point = self._get_line_from_point_and_angle(ax, cp, upper_fov_bound)
                # Plot coverage
                _ = self._plot_coverage(ax, cp, upper_coverage_bound_point, lower_coverage_bound_point, timestep)

    def _plot_occlusion_map(self, ax: plt.Axes, actor_bbox: np.ndarray, actor_heading: np.ndarray, timestep: int) -> None:
        """
        Plot occlusion map.
        Args:
            ax (plt.Axes): Matplotlib axes.
            actor_bbox (Bbox): Actor bounding box coordinates.
            actor_heading (np.ndarray): Actor heading.
            timestep (int): Timestep.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Plotting occlusion map.")
        # Mount sensors
        sensor_list = self._mount_sensors(ax, actor_bbox, actor_heading)
        # Cast rays from sensors
        self._cast_rays(ax, actor_heading, sensor_list, timestep)

    def _plot_actors_tracks(self, ax: plt.Axes, timestep: int) -> None:
        """
        Plot actor tracks.
        Args:
            ax (plt.Axes): Matplotlib axes.
            timestep (int): Timestep.
        """
        logger.info(
            f"{self.__LOG_PREFIX__}: Plotting actor/tracks for the scenario for timestep: {timestep}")
        for track in self.scenario.tracks:
            # Get actor states
            actor_positions, actor_headings, _ = self._get_actor_states(track, timestep)
            if actor_positions.shape[0] == 0 or actor_headings.shape[0] == 0:
                continue
            # Set actor defaults and associated bbox
            actor_path_color = self._TRACK_COLORS.get(track.category, TrackCategory.TRACK_FRAGMENT)
            actor_color = self._DEFAULT_ACTOR_COLOR
            bbox = None
            if track.track_id == self._AV_ID:
                actor_color = self._AV_COLOR
                actor_path_color = self._AV_PATH_COLOR
                bbox = self._ESTIMATED_AVEHICLE_SIZE
            elif track.category == TrackCategory.FOCAL_TRACK:
                actor_color = self._FOCAL_AGENT_COLOR
            if track.object_type == ObjectType.VEHICLE and track.track_id != self._AV_ID:
                bbox = tuple(self._ESTIMATED_VEHICLE_SIZE)
            elif track.object_type == ObjectType.CYCLIST or track.object_type == ObjectType.MOTORCYCLIST:
                bbox = tuple(self._ESTIMATED_CYCLIST_SIZE)
            if track.object_type in self._STATIC_OBJECT_TYPES:
                continue
            # Plot actor
            if bbox is not None:
                bbox_bounds = av2_plot_bbox(
                    ax=ax,
                    pivot_points=AV2Base.transform_bbox(
                        ref_location=actor_positions[-1],
                        heading=actor_headings[-1],
                        bbox_size=bbox
                    ),
                    heading=actor_headings[-1],
                    color=actor_color,
                    bbox_size=bbox,
                )
                if self.plot_occlusions and track.track_id == self._AV_ID:
                    self._plot_occlusion_map(ax, bbox_bounds, actor_headings[-1], timestep)
            else:
                plt.plot(
                    actor_positions[-1][0],
                    actor_positions[-1][1],
                    self._DEFAULT_ACTOR_STYLE,
                    markersize=self._DEFAULT_ACTOR_MARKERSIZE,
                    color=actor_color
                )
            # Plot actor path
            av2_plot_polylines([actor_positions], style=self._DEFAULT_ACTOR_PATH_STYLE, linewidth=self._DEFAULT_ACTOR_PATH_LINEWIDTH, alpha=self._DEFAULT_ACTOR_PATH_ALPHA, color=actor_path_color)
                
    def _generate_indetail_scenario_video(self, output_filename: str) -> str:
        """
        Generate detailed scenario video for the given scenario id.
        Args:
            output_filename (str): Output filename.
        Returns:
            str: Path to the generated detailed scenario video.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Generating detailed scenario video for scenario id: {self.scenario_id}")
        frames: List[Image.Image] = []
        for timestep in range(self._OBSERVATION_DURATION_TIMESTEPS + self._PREDICTION_DURATION_TIMESTEPS):
        # for timestep in range(1):
            # Plot
            _, ax = plt.subplots()
            self._visualize_map(
                show_pedestrian_xing=self.show_pedestrian_xing,
                drivable_area_alpha=self._DRIVABLE_AREA_ALPHA,
                drivable_area_color=self._DRIVABLE_AREA_COLOR,
                lane_segment_style=self._LANE_SEGMENT_STYLE,
                lane_segment_linewidth=self._LANE_SEGMENT_LINEWIDTH,
                lane_segment_alpha=self._LANE_SEGMENT_ALPHA,
                lane_segment_color=self._LANE_SEGMENT_COLOR,
                pedestrian_crossing_style=self._PED_XING_STYLE,
                pedestrian_crossing_linewidth=self._PED_XING_LINEWIDTH,
                pedestrian_crossing_alpha=self._PED_XING_ALPHA,
                pedestrian_crossing_color=self._PED_XING_COLOR
            )
            self._plot_actors_tracks(ax, timestep)
            # Save plot
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            plt.close()
            buffer.seek(0)
            frames.append(Image.open(buffer))
        # Save video
        video_file_path = self._get_output_file_path(output_filename)
        _ = write_video(frames, video_file_path, fps=self.fps, codec=self.codec)
        logger.info(f"{self.__LOG_PREFIX__}: Detailed scenario video generated.")
    
    def visualize(self) -> Union[str, None]:
        """
        Visualize motion forecasting data.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Visualizing motion forecasting data.")
        if self.output_filename is None:
            self.output_filename = f"{self.scenario_id}.mp4"
        self._make_output_directory()
        if self.raw:
            return self._generate_scenario_video(output_filename=self.output_filename)
        return self._generate_indetail_scenario_video(output_filename=self.output_filename)