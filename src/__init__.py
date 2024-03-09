##########################################################################################################
# Initialize the `src` package for the project
##########################################################################################################

import os
import logging
from pathlib import Path
from .base.vehicle import Vehicle
from .base.walker import Walker, WalkerAI
from .model import pydantic_model as PydanticModel
from .utils.logger import __setup_logger__
from .utils.utils import (
    print_param_table,
    read_yaml_file as read_yaml
)


__PATH_FILE__ = Path(__file__)
__PATH_PKG__ = __PATH_FILE__.parent
__PATH_ROOT__ = __PATH_FILE__.parent.parent
__LOGGING_DIR__ = f"{__PATH_ROOT__}/logs"
__LOGGING_LEVEL__ = logging.INFO

if not os.path.exists(__LOGGING_DIR__):
    os.makedirs(__LOGGING_DIR__)
__setup_logger__(log_dir=__LOGGING_DIR__, level=__LOGGING_LEVEL__)

__all__ = [
    "print_param_table",
    "read_yaml",
    "PydanticModel",
    "Walker",
    "WalkerAI",
    "Vehicle",
]