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
from typing import Tuple, Union, Set, List
from av2.map.map_api import ArgoverseStaticMap
from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.data_schema import Track, TrackCategory, ObjectType
from av2.datasets.motion_forecasting.viz.scenario_visualization import visualize_scenario
from src.agroverse.base import AV2Base
from src.utils.utils import write_video
from src.agroverse.utils import av2_plot_polylines, av2_plot_bbox



logger = logging.getLogger(__name__)


class AV2Forecasting(AV2Base):

    """
    Define tasks/functionalities associated with Agroverse Forecasting dataset instance here.
    Notes:
    1. `tracks` and `actors` are used interchangeably.
    """

    __LOG_PREFIX__ = "AV2Forecasting"
    OBSERVATION_DURATION_TIMESTEPS = 50
    PREDICTION_DURATION_TIMESTEPS = 60
    STATIC_OBJECT_TYPES: Set[ObjectType] = {
        ObjectType.STATIC,
        ObjectType.BACKGROUND,
        ObjectType.CONSTRUCTION,
        ObjectType.RIDERLESS_BICYCLE,
    }

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
        self._show_pedesrian_xing = kwargs.get("show_pedesrian_xing", False)
        self._drivalble_area_alpha = kwargs.get("drivable_area_alpha", 0.5)
        self._drivable_area_color = kwargs.get("drivable_area_color", "#7A7A7A")
        self._lane_segment_style = kwargs.get("lane_segment_style", "-")
        self._lane_segment_linewidth = kwargs.get("lane_segment_linewidth", 1.0)
        self._lane_segment_alpha = kwargs.get("lane_segment_alpha", 0.5)
        self._lane_segment_color = kwargs.get("lane_segment_color", "#E0E0E0")
        self._pedestrian_crossing_style = kwargs.get("pedestrian_crossing_style", "-")
        self._pedestrian_crossing_linewidth = kwargs.get("pedestrian_crossing_linewidth", 1.0)
        self._pedestrian_crossing_alpha = kwargs.get("pedestrian_crossing_alpha", 0.5)
        self._pedestrian_crossing_color = kwargs.get("pedestrian_crossing_color", "#FF0000")
        self._default_actor_path_color = kwargs.get("default_actor_path_color", "#D3E8EF")
        self._default_actor_path_linewidth = kwargs.get("default_actor_path_linewidth", 1.0)
        self._default_actor_path_alpha = kwargs.get("default_actor_path_alpha", 1.0)
        self._default_actor_path_style = kwargs.get("default_actor_path_style", "-")
        self._focal_agent_color = kwargs.get("focal_agent_color", "#ECA25B")
        self._av_color = kwargs.get("av_color", "#007672")
        self._default_object_style = kwargs.get("default_object_style", "o")
        self._default_object_markersize = kwargs.get("default_object_markersize", 4)
        self._estimated_vehicle_size = kwargs.get("estimated_vehicle_size", [4.0, 2.0]) # Length, Width
        assert len(self._estimated_vehicle_size) == 2, "Estimated vehicle size should be a list of length 2."
        self._estimated_cyclist_size = kwargs.get("estimated_cyclist_size", [2.0, 0.7]) # Length, Width
        assert len(self._estimated_cyclist_size) == 2, "Estimated cyclist size should be a list of length 2."
        self._codec = kwargs.get("codec", "mp4v")
        self._fps = kwargs.get("fps", 10)
        self.static_map_file, self.scenario_file = self._get_input_file_names()
        self.scenario = scenario_serialization.load_argoverse_scenario_parquet(
            Path(os.path.join(self.input_directory, self.scenario_id, self.scenario_file))
        )
        self.static_map = ArgoverseStaticMap.from_json(
            Path(os.path.join(self.input_directory, self.scenario_id, self.static_map_file))
        )
    
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
        logger.info(f"{self.__LOG_PREFIX__}: Getting timesteps from the scenario for track id: {track.track_id}")
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
        logger.info(f"{self.__LOG_PREFIX__}: Getting actor states from the scenario for track id: {track.track_id}")
        actor_positions = np.array([list(object_state.position) for object_state in track.object_states if timestep is not None and object_state.timestep <= timestep])
        actor_headings = np.array([object_state.heading for object_state in track.object_states if timestep is not None and object_state.timestep <= timestep])
        actor_velocities = np.array([list(object_state.velocity) for object_state in track.object_states if timestep is not None and object_state.timestep <= timestep])
        return (actor_positions, actor_headings, actor_velocities)

    def plot_actors_tracks(self, ax: plt.Axes, timestep: int):
        """
        Plot actor tracks.
        Args:
            ax (plt.Axes): Matplotlib axes.
            timestep (int): Timestep.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Plotting actor/tracks for the scenario.")
        for track in self.scenario.tracks:
            # Get valid timesteps
            actor_timesteps = self._get_timesteps(track, timestep)
            if actor_timesteps.shape[0] < 1 or actor_timesteps[-1] != timestep:
                continue
            # Get actor states
            actor_positions, actor_headings, actor_velocities = self._get_actor_states(track, timestep)
            # Plot actor polylines
            actor_color = self._default_actor_path_color
            if track.category == TrackCategory.FOCAL_TRACK:
                actor_color = self._focal_agent_color
                av2_plot_polylines([actor_positions], style=self._default_actor_path_style, linewidth=self._default_actor_path_linewidth, alpha=self._default_actor_path_alpha, color=actor_color)
            elif track.track_id == "AV":
                actor_color = self._av_color
            elif track.object_type in self.STATIC_OBJECT_TYPES:
                continue
            bbox = None
            if track.object_type == ObjectType.VEHICLE:
                bbox = tuple(self._estimated_vehicle_size)
            elif track.object_type == ObjectType.CYCLIST or track.object_type == ObjectType.MOTORCYCLIST:
                bbox = tuple(self._estimated_cyclist_size)
            if bbox is not None:
                av2_plot_bbox(
                    ax=ax,
                    pivot_points=AV2Base.transform_bbox(
                        current_location=actor_positions[-1],
                        heading=actor_headings[-1],
                        bbox_size=bbox
                    ),
                    heading=actor_headings[-1],
                    color=actor_color,
                    bbox_size=bbox
                )
            else:
                plt.plot(
                    actor_positions[-1][0],
                    actor_positions[-1][1],
                    self._default_object_style,
                    markersize=self._default_object_markersize,
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
        for timestep in range(self.OBSERVATION_DURATION_TIMESTEPS + self.PREDICTION_DURATION_TIMESTEPS):
            # Plot
            _, ax = plt.subplots()
            self._visualize_map(
                show_pedesrian_xing=False,
                drivable_area_alpha=self._drivalble_area_alpha,
                drivable_area_color=self._drivable_area_color,
                lane_segment_style=self._lane_segment_style,
                lane_segment_linewidth=self._lane_segment_linewidth,
                lane_segment_alpha=self._lane_segment_alpha,
                lane_segment_color=self._lane_segment_color,
                pedestrian_crossing_style=self._pedestrian_crossing_style,
                pedestrian_crossing_linewidth=self._pedestrian_crossing_linewidth,
                pedestrian_crossing_alpha=self._pedestrian_crossing_alpha,
                pedestrian_crossing_color=self._pedestrian_crossing_color
            )
            self.plot_actors_tracks(ax, timestep)
            # Save plot
            buffer = io.BytesIO()
            plt.savefig(buffer, format="png")
            plt.close()
            buffer.seek(0)
            frames.append(Image.open(buffer))
        # Save video
        video_file_path = self._get_output_file_path(output_filename)
        _ = write_video(frames, video_file_path, fps=self._fps, codec=self._codec)
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