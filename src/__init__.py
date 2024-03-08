##########################################################################################################
# Initialize the `src` package for the project
##########################################################################################################

import os
import logging
from pathlib import Path
from .utils.logger import __setup_logger__


__PATH_FILE__ = Path(__file__)
__PATH_PKG__ = __PATH_FILE__.parent
__PATH_ROOT__ = __PATH_PKG__.parent.parent

__LOGGING_DIR__ = f"{__PATH_ROOT__}/logs"
__LOGGING_LEVEL__ = logging.INFO

if not os.path.exists(__LOGGING_DIR__):
    os.makedirs(__LOGGING_DIR__)
__setup_logger__(log_dir=__LOGGING_DIR__, level=__LOGGING_LEVEL__)

__all__ = [
    # Add the modules that you want to import here
]