########################################################################################################################
# Utilites for the project
########################################################################################################################

import platform
import functools
from typing import Callable


def only_linux(func: Callable) -> Callable:
    """
    Decorator to check if the function is called on a Linux system.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if platform.system() != "Linux":
            raise OSError("This function can only be called on a Linux system!")
        return func(*args, **kwargs)
    return wrapper