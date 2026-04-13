import json
import typer
from typing import Any

class OutputManager:
    """
    Handles CLI output, supporting both human-readable and machine-readable (JSON) modes.
    """
    def __init__(self, json_mode: bool = False):
        self.json_mode = json_mode
        self.data = {}

    def echo(self, message: str, key: str = "message", err: bool = False):
        """
        Print a message to the user or store it for JSON output.
        """
        if self.json_mode:
            self.data[key] = message
        else:
            if err:
                typer.echo(message, err=True)
            else:
                typer.echo(message)

    def add_data(self, key: str, value: Any):
        """
        Add structured data to the JSON output. No-op in human mode
        since callers handle human output themselves.
        """
        if self.json_mode:
            self.data[key] = value

    def finalize(self):
        """
        Prints the final result as JSON if in JSON mode, otherwise does nothing.
        """
        if self.json_mode and self.data:
            print(json.dumps(self.data, indent=2))
