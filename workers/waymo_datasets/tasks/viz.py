#################################################################################################################
# Vizualization tasks - Waymo specific visualization tasks.
# Reference:
#     1. https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_motion.ipynb
#################################################################################################################

import os
import uuid
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from matplotlib import cm
from celery import shared_task
from typing import Tuple, List
from tasks.base import WaymoBase
from constants import FEATURES_DESCRIPTION


logger = logging.getLogger(__name__)


class WaymoOpenMotionDatasetViz(WaymoBase):

    """
    Define tasks/functionalities associated with Waymo Open Motion Dataset visualization here.
    """

    __LOG_PREFIX__ = "WaymoOpenMotionDatasetViz"

    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize the WaymoOpenMotionDatasetViz instance.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing Waymo Open Motion Dataset Viz.")
        self.input_directory = kwargs.get("input_directory", None)
        self.output_directory = kwargs.get("output_directory", None)
        assert self.input_directory is not None, "Input directory not provided."
        assert self.output_directory is not None, "Output directory not provided."
        self.scenario = kwargs.get("scenario", None)
        self.output_filename = kwargs.get("output_filename", None)
        self.dpi = kwargs.get("dpi", 100)
        self.size_pixels = kwargs.get("size_pixels", 1000)
        self.coverage = kwargs.get("coverage", 10.0)
        self.face_color = kwargs.get("face_color", "white")
        self.label_color = kwargs.get("label_color", "black")
        self.tick_color = kwargs.get("tick_color", "black")
        self.color_map = kwargs.get("color_map", "jet")
        self.animation_interval = kwargs.get("animation_interval", 1000)
        self.save = kwargs.get("save", True)
        self.file_delimiter = "__"
    
    def _init_visualization(self) -> None:
        """
        Initialize visualization.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Initializing visualization.")
        self.output_directory = os.path.join(self.output_directory, self._get_scenario_id_from_scenario(self.scenario))
        self._make_output_directory()

    def _create_figure_and_axes(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create figure and axes.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Creating figure and axes.")
        fig, ax = plt.subplots(1, 1, num=uuid.uuid4())
        size_inches = self.size_pixels / self.dpi
        fig.set_size_inches([size_inches, size_inches])
        fig.set_dpi(self.dpi)
        fig.set_facecolor(self.face_color)
        ax.set_facecolor(self.face_color)
        ax.xaxis.label.set_color(self.label_color)
        ax.yaxis.label.set_color(self.label_color)
        ax.tick_params(axis='x', colors=self.tick_color)
        ax.tick_params(axis='y', colors=self.tick_color)
        fig.set_tight_layout(True)
        ax.grid(False)
        return fig, ax
    
    def _fig_canvas_image(self, fig: plt.Figure) -> np.ndarray:
        """
        Get figure canvas image.
        Input:
            fig (plt.Figure): Figure.
        Returns:
            np.ndarray: [H, W, 3] uint8 array image from fig.canvas.tostring_rgb().
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting figure canvas image.")
        fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def _get_color_map(self, n_agents: int) -> np.ndarray:
        """
        Get color map.
        Input:
            n_agents (int): Number of agents.
        Returns:
            np.ndarray: [N, 4] float array of colors - RGBA. | N = number of agents.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting color map.")
        colors = cm.get_cmap(self.color_map, n_agents)
        colors = colors(range(n_agents))
        np.random.shuffle(colors)
        return colors
    
    def _get_viewport(self, all_states: np.ndarray, all_state_masks: np.ndarray) -> Tuple[float, float, float]:
        """
        Get the viewport; region containing data.
        Input:
            all_states (np.ndarray): [n_agents, n_steps, 2] array of states | n_steps = past + current + future.
            all_state_masks (np.ndarray): [n_agents, n_steps] array of state masks | n_steps = past + current + future.
        Returns:
            Tuple[float, float, float]: center_x, center_y, width.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting viewport.")
        valid_states = all_states[all_state_masks]
        all_x = valid_states[..., 0]
        all_y = valid_states[..., 1]
        center_x = (np.max(all_x) + np.min(all_x)) / 2.0
        center_y = (np.max(all_y) + np.min(all_y)) / 2.0
        range_x = np.ptp(all_x)
        range_y = np.ptp(all_y)
        width = max(range_x, range_y)
        return center_x, center_y, width
    
    def _visualize_one_step(self, states: np.ndarray, state_masks: np.ndarray, roadgraph: np.ndarray, center_x: float, center_y: float, width: float, color_map: np.ndarray, title: str) -> np.ndarray:
        """
        TODO: chekc the size/dims of each params.
        Generate a visualization for one step | N = number of agents.
        Input:
            states (np.ndarray): [N, 2] array of states.
            state_masks (np.ndarray): [N] array of state masks.
            roadgraph (np.ndarray): [roadgraph_points, 2] array of roadgraph - refer ../constants.py for more details.
            center_x (float): Center x.
            center_y (float): Center y.
            width (float): Width.
            color_map (np.ndarray): [N, 4] array of colors - RGBA.
            title (str): Title.
        Returns:
            np.ndarray: [H, W, 3] uint8 array image from fig.canvas.tostring_rgb().
        """
        logger.info(f"{self.__LOG_PREFIX__}: Visualizing one step: {title}.")
        fig, ax = self._create_figure_and_axes()
        rg_pts = roadgraph[:, :2].T
        ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.')
        masked_x = states[state_masks, 0]
        masked_y = states[state_masks, 1]
        colors = color_map[state_masks]
        ax.scatter(masked_x, masked_y, marker='o', color=colors)
        ax.set_title(title)
        size = max(self.coverage, width)
        ax.axis(
            [
                -size/2 + center_x,
                size/2 + center_x,
                -size/2 + center_y,
                size/2 + center_y,
            ]
        )
        ax.set_aspect('equal')
        image = self._fig_canvas_image(fig)
        plt.close(fig)
        return image
    
    def _get_past_states(self, data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get past states.
        Input:
            data (dict): Data from scenario tf record.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of past states and past state masks.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting past states.")
        past_states = tf.stack(
            [data["state/past/x"], data["state/past/y"]], -1
        ).numpy()
        past_state_masks = data["state/past/valid"].numpy() > 0.0
        return past_states, past_state_masks
    
    def _get_current_states(self, data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current states.
        Input:
            data (dict): Data from scenario tf record.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of current states and current state masks.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting current states.")
        current_states = tf.stack(
            [data["state/current/x"], data["state/current/y"]], -1
        ).numpy()
        current_state_masks = data["state/current/valid"].numpy() > 0.0
        return current_states, current_state_masks
    
    def _get_future_states(self, data: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get future states.
        Input:
            data (dict): Data from scenario tf record.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of future states and future state masks.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting future states.")
        future_states = tf.stack(
            [data["state/future/x"], data["state/future/y"]], -1
        ).numpy()
        future_state_masks = data["state/future/valid"].numpy() > 0.0
        return future_states, future_state_masks
    
    def _get_roadgraph_samples(self, data: dict) -> np.ndarray:
        """
        Get roadgraph samples.
        Input:
            data (dict): Data from scenario tf record.
        Returns:
            np.ndarray: [roadgraph_points, 2] array of roadgraph.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting roadgraph samples.")
        roadgraph_xyz = data["roadgraph_samples/xyz"].numpy()
        return roadgraph_xyz

    def _visualize_agents(self, data: dict) -> List[np.ndarray]:
        """
        Visualize all agents in the scenario.
        Input:
            data (dict): Data from scenario tf record.
        Returns:
            images (List[np.ndarray]): List of images for each step.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Visualizing agents.")
        # Past states
        past_states, past_state_masks = self._get_past_states(data)
        # Current states
        current_states, current_state_masks = self._get_current_states(data)
        # Future states
        future_states, future_state_masks = self._get_future_states(data)
        # Roadgraph samples
        roadgraph_xyz = self._get_roadgraph_samples(data)
        # Process
        n_agents, n_past_steps, _ = past_states.shape
        n_future_steps = future_states.shape[1]
        color_map = self._get_color_map(n_agents)
        all_states = np.concatenate([past_states, current_states, future_states], axis=1)
        all_state_masks = np.concatenate([past_state_masks, current_state_masks, future_state_masks], axis=1)
        center_x, center_y, width = self._get_viewport(all_states, all_state_masks)
        images = []
        for i, (s, sm) in enumerate(zip(
            np.split(past_states, n_past_steps, axis=1),
            np.split(past_state_masks, n_past_steps, axis=1)
        )):
            im = self._visualize_one_step(
                states=s[:, 0],
                state_masks=sm[:, 0],
                roadgraph=roadgraph_xyz,
                center_x=center_x,
                center_y=center_y,
                width=width,
                color_map=color_map,
                title=f"Past step {n_past_steps - i}"
            )
            images.append(im)
        s, sm = current_states, current_state_masks
        im = self._visualize_one_step(
            states=s[:, 0],
            state_masks=sm[:, 0],
            roadgraph=roadgraph_xyz,
            center_x=center_x,
            center_y=center_y,
            width=width,
            color_map=color_map,
            title=f"Current step"
        )
        images.append(im)
        for i, (s, sm) in enumerate(zip(
            np.split(future_states, n_future_steps, axis=1),
            np.split(future_state_masks, n_future_steps, axis=1)
        )):
            im = self._visualize_one_step(
                states=s[:, 0],
                state_masks=sm[:, 0],
                roadgraph=roadgraph_xyz,
                center_x=center_x,
                center_y=center_y,
                width=width,
                color_map=color_map,
                title=f"Future step {i+1}"
            )
            images.append(im)
        return images
    
    def _generate_animation(self, images: List[np.ndarray]) -> Tuple[animation.FuncAnimation, str]:
        """
        Generate animation.
        Input:
            images (List[np.ndarray]): List of images for each step.
        Returns:
            Tuple[animation.FuncAnimation, str]: Tuple of animation and output file path.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Generating animation.")

        def animate_func(i: int) -> None:
            """Animation function."""
            ax.imshow(images[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid("off")

        plt.ioff()
        fig, ax = plt.subplots()
        size_inches = self.size_pixels / self.dpi
        fig.set_size_inches([size_inches, size_inches])
        plt.ion()
        anim = animation.FuncAnimation(fig, animate_func, frames=len(images)//2, interval=self.animation_interval)
        plt.close(fig)
        if self.save:
            if self.output_filename is None:
                output_filename = "scenario_animation.avi"
            else:
                output_filename = self.output_filename + ".avi"
            output_file_path = self._get_output_file_path(output_filename)
            anim.save(str(output_file_path))
            return anim, str(output_file_path)
        return anim, None
    
    def get_single_example_from_tf_record(self, scenario: str = None) -> dict:
        """
        Get data from scenario tf record.
        Input:
            scenario (str): Scenario ID - Optional. This must be provided if the member variable scenario is None.
        Returns:
            dict: Data from scenario tf record.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Getting data from scenario tf record.")
        assert scenario or self.scenario, "Scenario ID not provided."
        scenario = scenario or self.scenario
        dataset = tf.data.TFRecordDataset(os.path.join(self.input_directory, scenario), compression_type='')
        data = next(dataset.as_numpy_iterator())
        return tf.io.parse_single_example(data, FEATURES_DESCRIPTION)

    def visualize(self) -> str:
        """
        Visualize Waymo Open Motion Dataset.
        Returns:
            str: Path to the output file.
        """
        logger.info(f"{self.__LOG_PREFIX__}: Visualizing Waymo Open Motion Dataset.")
        self._init_visualization()
        scenario_data = self.get_single_example_from_tf_record(self.scenario)
        images = self._visualize_agents(scenario_data)
        _, output_file_path = self._generate_animation(images)
        logger.info(f"{self.__LOG_PREFIX__}: Animation saved to {output_file_path}.")
        return output_file_path


@shared_task(name="viz.waymo_open_motion_dataset")
def viz_waymo_open_motion_dataset_tf_record(*args, **kwargs):
    """Visualize Waymo Open Dataset TFRecord."""
    return WaymoOpenMotionDatasetViz(*args, **kwargs).visualize()