##########################################################################################################
# Initialize the `src` package for the project
##########################################################################################################

import os
import sys
import platform
import logging
from pathlib import Path
from .utils.logger import __setup_logger__

__PATH_FILE__ = Path(__file__)
__PATH_PKG__ = __PATH_FILE__.parent
__PATH_ROOT__ = __PATH_FILE__.parent.parent
__LOGGING_DIR__ = os.getenv("LOGGING_DIR", f"{__PATH_ROOT__}/logs")
__LOGGING_LEVEL__ = os.getenv("LOGGING_LEVEL", logging.INFO)

if not os.path.exists(__LOGGING_DIR__):
    os.makedirs(__LOGGING_DIR__)
__setup_logger__(log_dir=__LOGGING_DIR__, level=__LOGGING_LEVEL__)

sys.path.append(str(__PATH_ROOT__ / "workers" / "waymo_datasets"))

from .utils.utils import print_param_table
from .agroverse.forecasting import AV2Forecasting
from .waymo.forecasting import WaymoForecasting

__all__ = [
    "AV2Forecasting",
    "WaymoForecasting",
    "print_param_table"
]

# check if os is linux
if platform.system == "Linux":
    from .base.vehicle import Vehicle
    from .client import CarlaClientCLI
    from .base.walker import Walker, WalkerAI
    from .model.enum import SensorConvertorType
    from .data_synthesizer import DataSynthesizer
    from .model import validators as PydanticModel
    from .motion_planning import HighLevelMotionPlanner
    from .utils.utils import (
        read_yaml_file as read_yaml,
        write_yaml_file as write_yaml,
        generate_vehicle_configuration_dict as generate_vehicle_config,
        generate_pedestrian_configuration_dict as generate_pedestrian_config,
        write_txt_report_style_1
    )
    __all__ += [
        "read_yaml",
        "write_yaml",
        "PydanticModel",
        "Walker",
        "WalkerAI",
        "Vehicle",
        "DataSynthesizer",
        "CarlaClientCLI",
        "HighLevelMotionPlanner",
        "generate_vehicle_config",
        "generate_pedestrian_config",
        "write_txt_report_style_1",
        "SensorConvertorType"
    ]