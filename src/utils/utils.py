###################################################################################################
# Any utilities used in this projects must be defined here.
###################################################################################################

import yaml
from rich.table import Table
from rich.console import Console


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