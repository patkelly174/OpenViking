from pathlib import Path
import os

class Brain:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.brain_dir = self.project_root / ".ov_brain"

    def resolve_uri(self, uri: str) -> str:
        if uri.startswith("viking://"):
            # Strip prefix
            path_part = uri[len("viking://"):]
            # Strip leading slashes to prevent os.path.join (or Path /) from treating it as absolute
            path_part = path_part.lstrip("/")

            # Resolve the path relative to project root
            resolved_path = (self.project_root / path_part).resolve()

            # Verify it's still within the project root to prevent traversal
            if not resolved_path.is_relative_to(self.project_root):
                raise ValueError(f"Resolved path {resolved_path} is outside the project root {self.project_root}")

            return str(resolved_path)
        return uri
