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
from typing import Tuple, Union, List
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import Track, TrackCategory, ObjectType
from av2.datasets.motion_forecasting.viz.scenario_visualization import visualize_scenario
from src.agroverse.base import AV2Base
from src.agroverse.constants import CAMERA_TYPE_NAME
from src.agroverse.utils import av2_plot_polylines, av2_plot_bbox
from src.utils.utils import write_video, read_yaml_file, bilinear_interpolate
from src.agroverse.model.validators import Rotation, InternalCameraMountResponse, Vehicle as AV2Vehicle


logger = logging.getLogger(__name__)


class AV2Forecasting(AV2Base):

    """
    Define tasks/functionalities associated with Agroverse Forecasting dataset instance here.
    Notes:
    1. `tracks` and `actors` are used interchangeably.
    """

    __LOG_PREFIX__ = "AV2Forecasting"
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
        logger.info(f"{self.__LOG_PREFIX__}: Initializing Agroverse Forecasting dataset instance.")
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
    
    def _get_line_from_point_and_angle(self, ax: plt.Axes, point: np.ndarray, theta: float) -> Tuple[np.ndarray, np.ndarray]:
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
            Tuple[np.ndarray, np.ndarray]: Tuple containing line points.
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

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        # Normalize theta
        theta = theta % (2 * np.pi) if theta < 0 else theta
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
            delta = point - candidate
            theta_dash = np.arctan2(delta[1], delta[0])
            x_dash_hat, y_dash_hat = np.cos(theta_dash), np.sin(theta_dash)
            if np.isclose(np.dot([x_hat, y_hat], [x_dash_hat, y_dash_hat]), -1.0):
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
        return np.array(lines[0])

    def _cast_rays(self, ax: plt.Axes, actor_heading: float, sensor_list: List[Union[InternalCameraMountResponse]]) -> None:
        """
        Cast rays from sensors.
        Args:
            ax (plt.Axes): Matplotlib axes.
            actor_bbox (np.ndarray): Actor bounding box coordinates.
            actor_heading (float): Actor heading.
            sensor_list (List[Union[InternalCameraMountResponse]]): List of sensors.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Casting rays from sensors.")
        for sensor in sensor_list:
            if sensor.type == CAMERA_TYPE_NAME:
                fov = sensor.camera.get_fov_radians()
                cp = sensor.get_mid_bounds()
                yaw, _, _ = sensor.camera.rotation.get_rotation_radians()
                upper_fov_bound = actor_heading + yaw + (fov / 2)
                lower_fov_bound = actor_heading + yaw - (fov / 2)
                upper_coverage_bound_point = self._get_line_from_point_and_angle(ax, cp, upper_fov_bound)
                lower_coverage_bound_point = self._get_line_from_point_and_angle(ax, cp, lower_fov_bound)

    def _plot_occlusion_map(self, ax: plt.Axes, actor_bbox: np.ndarray, actor_position: np.ndarray, actor_heading: np.ndarray) -> None:
        """
        Plot occlusion map.
        Args:
            ax (plt.Axes): Matplotlib axes.
            actor_bbox (Bbox): Actor bounding box coordinates.
            actor_position (np.ndarray): Actor position.
            actor_heading (np.ndarray): Actor heading.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Plotting occlusion map.")
        # Mount sensors
        sensor_list = self._mount_sensors(ax, actor_bbox, actor_heading)
        # Cast rays from sensors
        self._cast_rays(ax, actor_heading, sensor_list)

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
            # Get valid timesteps
            actor_timesteps = self._get_timesteps(track, timestep)
            if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
                continue
            # Get actor states
            actor_positions, actor_headings, _ = self._get_actor_states(track, timestep)
            # Set actor defaults
            actor_path_color = self._TRACK_COLORS.get(track.category, TrackCategory.TRACK_FRAGMENT)
            actor_color = self._DEFAULT_ACTOR_COLOR
            bbox = None
            if track.track_id == self._AV_ID:
                actor_color = self._AV_COLOR
                actor_path_color = self._AV_PATH_COLOR
                bbox = self._ESTIMATED_AVEHICLE_SIZE
            elif track.category == TrackCategory.FOCAL_TRACK:
                actor_color = self._FOCAL_AGENT_COLOR
            # Plot actor path
            av2_plot_polylines([actor_positions], style=self._DEFAULT_ACTOR_PATH_STYLE, linewidth=self._DEFAULT_ACTOR_PATH_LINEWIDTH, alpha=self._DEFAULT_ACTOR_PATH_ALPHA, color=actor_path_color)
            # Set actor bbox
            if track.object_type in self._STATIC_OBJECT_TYPES:
                continue
            if track.object_type == ObjectType.VEHICLE and track.track_id != self._AV_ID:
                bbox = tuple(self._ESTIMATED_VEHICLE_SIZE)
            elif track.object_type == ObjectType.CYCLIST or track.object_type == ObjectType.MOTORCYCLIST:
                bbox = tuple(self._ESTIMATED_CYCLIST_SIZE)
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
                    self._plot_occlusion_map(ax, bbox_bounds, actor_positions[-1], actor_headings[-1])
            else:
                plt.plot(
                    actor_positions[-1][0],
                    actor_positions[-1][1],
                    self._DEFAULT_ACTOR_STYLE,
                    markersize=self._DEFAULT_ACTOR_MARKERSIZE,
                    color=actor_color
                )
                
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
        # for timestep in range(3):
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