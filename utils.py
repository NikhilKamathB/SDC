########################################################################################################################
# Utilites for the project
########################################################################################################################

import platform
import functools
from rich.live import Live
from rich.spinner import Spinner
from typing import Callable, Coroutine


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

async def display_indefinite_loading_animation(coroutine: Coroutine, spinner_message: str = "Processing...", refresh_per_second: int = 10):
    spinner = Spinner("dots", text=spinner_message)
    with Live(spinner, refresh_per_second=refresh_per_second, transient=True) as live:
        result = await coroutine
        return result