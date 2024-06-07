###################################################################################################
# Any utilities used in this projects must be defined here.
###################################################################################################

import os
import yaml
import string
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from rich.table import Table
from rich.console import Console
from collections import defaultdict
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

def plot_3d_matrix(matrix1: np.ndarray, matrix2: np.ndarray = None, figaspect: float = 0.5) -> None:
    """
    Plot the 3D matrix.
    If matrix2 is provided, plot dots along with a line from matrix1[i] to matrix2[i].
    Input parameters:
        - matrix1: np.ndarray - the first matrix.
        - matrix2: np.ndarray - the second matrix.
        - figaspect: float - the aspect ratio of the figure.
    """
    # Set figure twice as wide as it is tall
    fig = plt.figure(figsize=plt.figaspect(figaspect))
    rows, cols = 1, 1
    if matrix2 is not None:
        cols = 2
    # First plot
    ax = fig.add_subplot(rows, cols, 1, projection='3d')
    ax.scatter(matrix1[:, 0], matrix1[:, 1], matrix1[:, 2], c='r', marker='o', label='matrix1')
    if matrix2 is not None:
        ax.scatter(matrix2[:, 0], matrix2[:, 1], matrix2[:, 2], c='b', marker='+', label='matrix2')
    ax.legend()
    if matrix2 is not None:
        # Second plot
        ax = fig.add_subplot(rows, cols, 2, projection='3d')
        for i in range(matrix1.shape[0]):
            ax.plot([matrix1[i, 0], matrix2[i, 0]], [matrix1[i, 1], matrix2[i, 1]], [matrix1[i, 2], matrix2[i, 2]])
    plt.show()