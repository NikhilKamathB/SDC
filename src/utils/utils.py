###################################################################################################
# Any utilities used in this projects must be defined here.
###################################################################################################

import os
import yaml
import string
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rich.table import Table
from rich.console import Console
from collections import defaultdict
from typing import List, Tuple, Union
from matplotlib.widgets import TextBox
from src.model.enum import Gen1VehicleType, Gen2VehicleType, WalkerType


def generate_random_string(length: int = 10) -> str:
    """
    Generate a random string of the specified length.
    Input parameters:
        - length: int - the length of the string.
    Output:
        - str: the generated string.
    """
    return "".join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=length))

def read_yaml_file(file_path: str) -> dict:
    """
    Read the content of the yaml file.
    Input parameters:
        - file_path: path of the yaml file.
    Output:
        - dict: content of the yaml file.
    """
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def write_yaml_file(file_path: str, data: dict) -> None:
    """
    Write the data to the yaml file.
    Input parameters:
        - file_path: path of the yaml file.
        - data: dict - the data to be written to the file.
    """
    with open(file_path, "w") as file:
        yaml.dump(data, file)

def print_param_table(parmas: dict, title: str = None, header_style: str = "bold magenta", show_header: bool = True) -> None:
    """
    Print the data in a tabular format.
    Input parameters:
        - parmas: dict - the data to be printed.
        - title: str - the title of the table.
        - header_style: str - the style of the header.
        - show_header: bool - whether to show the header or not.
    """
    console = Console()
    table = Table(title=title, header_style=header_style, show_header=show_header)
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value", style="bold green")
    for key, value in parmas.items():
        table.add_row(key, str(value) if value is not None else "< None >")
    console.print(table)

def generate_vehicle_configuration_dict(reference_dict: dict, sample_id: int) -> dict:
    """
    Generate a configuration dictionary for the vehicles.
    Input parameters:
        - reference_dict: dict - the reference dictionary.
        - sample_id: int - the id of the sample.
    """
    blueprint_id = random.choice(list(Gen1VehicleType) + list(Gen2VehicleType)).value
    random_role_id = generate_random_string()
    role_name = f"vehicle_{sample_id}_{random_role_id}"
    sensors = reference_dict.get("sensors", defaultdict(list))
    for sensor, sensor_data in sensors.items():
        for sensor_item in sensor_data:
            sensor_item["fov"] = random.randint(90, 90)
            sensor_item["image_size_x"] = 452
            sensor_item["image_size_y"] = 432
            sensor_item["sensor_tick"] = 0.0
            if sensor == "camera_rgb":
                sensor_item["role_name"] = f"vehicle_camera_rgb_{sample_id}_{random_role_id}"
                sensor_item["bloom_intensity"] = 0.675
                sensor_item["fstop"] = 1.4
                sensor_item["iso"] = 100.0
                sensor_item["gamma"] = 2.2
                sensor_item["lens_flare_intensity"] = 0.1
                sensor_item["shutter_speed"] = 200.0
            elif sensor == "camera_depth":
                sensor_item["role_name"] = f"vehicle_camera_depth_{sample_id}_{random_role_id}"
                sensor_item["lens_circle_falloff"] = 5.0
                sensor_item["lens_circle_multiplier"] = 0.0
                sensor_item["lens_k"] = -1.0
                sensor_item["lens_kcube"] = 0.0
                sensor_item["lens_x_size"] = 0.08
                sensor_item["lens_y_size"] = 0.08
    return {
        "blueprint_id": blueprint_id,
        "role_name": role_name,
        "sensors": sensors
    }

def generate_pedestrian_configuration_dict(reference_dict: dict, sample_id: int) -> dict:
    """
    Generate a configuration dictionary for the pedestrian.
    Input parameters:
        - reference_dict: dict - the reference dictionary.
        - sample_id: int - the id of the sample.
    """
    blueprint_id = random.choice(list(WalkerType)).value
    random_role_id = generate_random_string()
    role_name = f"pedestrian_{sample_id}_{random_role_id}"
    is_invincible = False
    attach_ai = True
    run_probability = random.uniform(0.0, 1.0)
    return {
        "blueprint_id": blueprint_id,
        "role_name": role_name,
        "is_invincible": is_invincible,
        "attach_ai": attach_ai,
        "run_probability": run_probability
    }

def write_txt_report_style_1(files: List[str], output_file: str, sensor_type: str, prefix_tag: bool = False, prefix_dir: bool = False, need_file_name: bool = False) -> None:
    """
    Generate a report in a txt file.
    Style:
        <timstamp> <prefix_tag with data file name> <data file name with dir> <data file_name>
    Input parameters:
        - files: List[str] - the list of files.
        - output_file: str - the name of the output file.
        - sensor_type: str - the type of the sensor.
        - prefix_tag: bool - whether to prefix the tag or not.
        - prefix_dir: bool - whether to prefix the directory or not.
        - need_file_name: bool - whether to include the file name or not.
    """
    with open(output_file, "w") as f:
        res = []
        for file in files:
            line = ""
            file_name = file.split("/")[-1]
            timestamp = file_name.split('_')[1]
            line += str(timestamp) + ' '
            if prefix_tag:
                prefix_tag_file_name = os.path.join(sensor_type, file_name)
                line += prefix_tag_file_name + ' '
            if prefix_dir:
                prefix_dir_file_name = os.path.join(
                    *file.split("/")[-2:])
                line += prefix_dir_file_name + ' '
            if need_file_name:
                line += file_name + ' '
            line += '\n'
            res.append((float(timestamp), line))
        res.sort(key=lambda x: x[0])
        for _, line in res:
            f.write(line)

def plot_3d_matrix(
        matrix1: np.ndarray, 
        matrix2: np.ndarray = None, 
        matrix1_annotations: Tuple[np.ndarray, List[str]] = None, 
        matrix2_annotations: Tuple[np.ndarray, List[str]] = None, 
        matrix1_annotations_offset: float = 0.0125, 
        matrix2_annotations_offset: float = -0.0125, 
        matrix1_color: str = 'r', 
        matrix2_color: str = 'b', 
        matrix1_annotations_color: str = 'k',
        matrix2_annotations_color: str = 'k',
        axis_labels: Tuple[str, str, str] = ['X', 'Y', 'Z'], 
        figaspect: float = 0.5, 
        title: str = "", 
        plot_secondary_graph: bool = True,
        need_user_input: bool = False,
        input_text_attrs: List[Tuple[str, Union[int, str]]] = None,
        input_text_align: str = "center",
        submit_button_display_text: str = "Submit",
        default_height_ratio: float = 1,
        misc_height_ratio: float = 0.075,
        default_response: List[int] = [0, -1]
        ) -> Union[None, List[str]]:
    """
    Plot the 3D matrix.
    If matrix2 is provided, plot dots along with a line from matrix1[i] to matrix2[i].
    You can optionally take input from the user using `TextBox`.
    The first row is reserved for the plot, the rest of the rows are for the input text fields and the submit button if enabled.
    Input parameters:
        - matrix1: np.ndarray - the first matrix.
        - matrix2: np.ndarray - the second matrix.
        - matrix1_annotations: Tuple[np.ndarray, List[str]] - the annotations for the first matrix.
        - matrix2_annotations: Tuple[np.ndarray, List[str]] - the annotations for the second matrix.
        - matrix1_annotations_offset: float - the offset for the annotations of the first matrix.
        - matrix2_annotations_offset: float - the offset for the annotations of the second matrix.
        - matrix1_color: str - the color of the first matrix.
        - matrix2_color: str - the color of the second matrix.
        - matrix1_annotations_color: str - the color of the annotations of the first matrix.
        - matrix2_annotations_color: str - the color of the annotations of the second matrix.
        - axis_labels: Tuple[str, str, str] - the labels for the first matrix.
        - figaspect: float - the aspect ratio of the figure.
        - title: str - the title of the plot.
        - plot_secondary_graph: bool - whether to plot the secondary graph or not.
        - need_user_input: bool - whether to ask for user input or not.
        - input_text_attrs: List[Tuple[str, Union[int, str]]] - the attributes for the input text, for example:
            [
                (<text-field1-name>, <str>),
                (<text-field2-name>, <int>),
                ...
            ]
        - input_text_align: str - the alignment of the input text.
        - submit_button_display_text: str - the text to be displayed on the submit button.
        - misc_height_ratio: float - the height ratio for the misc items - text fields and buttons.
        - default_response: List[int] - the default response if no input text fields are rendered.
    Return: List[str] - the values of the input text fields and None if no input text fields are not rendered.
    """

    def submit(event) -> None:
        """
        Action taken on clicking the submit button.
        """
        nonlocal response
        response = [tbox.text for tbox in input_text_boxes]
        plt.close()

    fig = plt.figure(figsize=plt.figaspect(figaspect))
    misc_rows = 0
    rows, cols = 1, 1
    input_text_boxes = []
    response = default_response
    if matrix2 is not None:
        cols = 2
    if need_user_input:
        misc_rows = len(input_text_attrs) + 1 # Add one for the submit button
        rows += misc_rows
    gs = gridspec.GridSpec(rows, cols, height_ratios=[default_height_ratio] + [misc_height_ratio] * misc_rows)
    # First plot
    # Plot actual information
    ax = fig.add_subplot(gs[0, 0], projection='3d')
    ax.scatter(matrix1[:, 0], matrix1[:, 1], matrix1[:, 2], c=matrix1_color, marker='o', label='matrix1')
    if matrix2 is not None:
        ax.scatter(matrix2[:, 0], matrix2[:, 1], matrix2[:, 2], c=matrix2_color, marker='+', label='matrix2')
    # Plot annotations
    if matrix1_annotations is not None:
        itr_matrix1_range = min(matrix1_annotations[0].shape[0], len(matrix1_annotations[1]))
        for i in range(itr_matrix1_range):
            ax.text(matrix1_annotations[0][i, 0], matrix1_annotations[0][i, 1], matrix1_annotations[0][i, 2] + matrix1_annotations_offset, c=matrix1_annotations_color, s=matrix1_annotations[1][i])
    if matrix2 is not None and matrix2_annotations is not None:
        itr_matrix2_range = min(matrix2_annotations[0].shape[0], len(matrix2_annotations[1]))
        for i in range(itr_matrix2_range):
            ax.text(matrix2_annotations[0][i, 0], matrix2_annotations[0][i, 1], matrix2_annotations[0][i, 2] + matrix2_annotations_offset, c=matrix2_annotations_color, s=matrix2_annotations[1][i])
    ax.legend()
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    # Second plot
    if matrix2 is not None and plot_secondary_graph:
        ax = fig.add_subplot(gs[0, 1], projection='3d')
        for i in range(matrix1.shape[0]):
            ax.plot([matrix1[i, 0], matrix2[i, 0]], [matrix1[i, 1], matrix2[i, 1]], [matrix1[i, 2], matrix2[i, 2]])
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
    # User input
    if need_user_input and input_text_attrs is not None:
        for idx, tbox in enumerate(input_text_attrs):
            tbox_name, tbox_default_value = tbox
            axbox = fig.add_subplot(gs[1+idx, :])
            text_box = TextBox(axbox, tbox_name, initial=str(tbox_default_value), textalignment=input_text_align)
            input_text_boxes.append(text_box)
        axsubmit = fig.add_subplot(gs[1+len(input_text_attrs), :])
        button = plt.Button(axsubmit, submit_button_display_text)
        button.on_clicked(submit)
    fig.suptitle(title)
    plt.show()
    return response

def plot_3d_roads(road1: Tuple[np.ndarray, np.ndarray], road2: Tuple[np.ndarray, np.ndarray] = None, axis_labels: Tuple[str, str, str] = ['X', 'Y', 'Z'], figaspect: float = 0.5, title: str = "") -> None:
    """
    Plot the 3D roads.
    If `road2` params is provided, plot two set of roads for comparison.
    Input parameters:
        - road1: Tuple[np.ndarray, np.ndarray] - the first road - [start, end] np arrays.
        - road2: Tuple[np.ndarray, np.ndarray] - the second road - [start, end] np arrays.
        - axis_labels: Tuple[str, str, str] - the labels for the first matrix.
        - figaspect: float - the aspect ratio of the figure.
        - title: str - the title of the plot.
    """
    assert len(road1) == 2, "Road1 must be a tuple of two numpy arrays."
    if road2 is not None:
        assert len(road2) == 2, "Road2 must be a tuple of two numpy arrays."
    fig = plt.figure(figsize=plt.figaspect(figaspect))
    rows, cols = 1, 1
    if road2 is not None:
        cols = 2
    # First plot
    ax = fig.add_subplot(rows, cols, 1, projection='3d')
    for i in range(road1[0].shape[0]):
        ax.plot([road1[0][i, 0], road1[1][i, 0]], [road1[0][i, 1], road1[1][i, 1]], [road1[0][i, 2], road1[1][i, 2]])
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    if road2 is not None:
        # Second plot
        ax = fig.add_subplot(rows, cols, 2, projection='3d')
        for i in range(road2[0].shape[0]):
            ax.plot([road2[0][i, 0], road2[1][i, 0]], [road2[0][i, 1], road2[1][i, 1]], [road2[0][i, 2], road2[1][i, 2]])
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_zlabel(axis_labels[2])
    fig.suptitle(title)
    plt.show()