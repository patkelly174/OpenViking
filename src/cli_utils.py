import json
import typer
from typing import Any


class OutputManager:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def add(self, key: str, value: Any) -> None:
        self._data[key] = value

    def error(self, message: str, code: int = 1) -> None:
        print(json.dumps({"status": "error", "message": message}, indent=2))
        self._data = {}
        raise typer.Exit(code=code)

    def finalize(self) -> None:
        self._data.setdefault("status", "ok")
        print(json.dumps(self._data, indent=2))
        self._data = {}
