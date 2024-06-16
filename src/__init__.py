##########################################################################################################
# Initialize the `src` package for the project
##########################################################################################################

import os
import logging
from pathlib import Path
from .base.vehicle import Vehicle
from .client import CarlaClientCLI
from .base.walker import Walker, WalkerAI
from .data_synthesizer import DataSynthesizer
from .model import validators as PydanticModel
from .motion_planning import HighLevelMotionPlanner
from .model.enum import DistanceMetric, SearchAlgorithm, SensorConvertorType, SpectatorAttachmentMode, TMActorSpeedMode
from .utils.logger import __setup_logger__
from .utils.utils import (
    print_param_table,
    read_yaml_file as read_yaml,
    write_yaml_file as write_yaml,
    generate_vehicle_configuration_dict as generate_vehicle_config,
    generate_pedestrian_configuration_dict as generate_pedestrian_config,
    write_txt_report_style_1
)


__PATH_FILE__ = Path(__file__)
__PATH_PKG__ = __PATH_FILE__.parent
__PATH_ROOT__ = __PATH_FILE__.parent.parent
__LOGGING_DIR__ = os.getenv("LOGGING_DIR", f"{__PATH_ROOT__}/logs")
__LOGGING_LEVEL__ = logging.INFO

if not os.path.exists(__LOGGING_DIR__):
    os.makedirs(__LOGGING_DIR__)
__setup_logger__(log_dir=__LOGGING_DIR__, level=__LOGGING_LEVEL__)

__all__ = [
    "print_param_table",
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
    "DistanceMetric",
    "SearchAlgorithm",
    "SpectatorAttachmentMode",
    "SensorConvertorType",
    "TMActorSpeedMode"
]