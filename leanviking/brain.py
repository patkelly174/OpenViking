from pathlib import Path
import os

class Brain:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.brain_dir = self.project_root / ".ov_brain"
        from leanviking.indexer import Indexer
        self.indexer = Indexer(str(self.project_root))

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

    def get_context(self, query: str, top_k: int = 3, include_l2: bool = False) -> str:
        # Sensing: L0 search returns viking:// URIs
        l0_uris = self.indexer.search(query, top_k=top_k)

        context_blocks = []
        for uri in l0_uris:
            resolved = self.resolve_uri(uri)  # absolute path to source file

            # Positioning: load L1 overview for the containing directory
            l1_path = self._get_l1_path(resolved)
            try:
                context_blocks.append(l1_path.read_text())
            except (FileNotFoundError, IOError):
                pass

            # Execution: raw source (L2) — only when caller asks
            if include_l2:
                try:
                    context_blocks.append(Path(resolved).read_text())
                except (FileNotFoundError, IOError):
                    pass

        return "\n\n".join(context_blocks)

    def _get_l1_path(self, path: str) -> Path:
        # Ensure we have a Path object
        p = Path(path)
        if not p.is_absolute():
            p = self.project_root / p

        # If it's a file, we want the overview of its containing directory
        # Use suffix check instead of is_file() to avoid disk hits and handle virtual paths
        if p.suffix != '' or not p.is_dir():
            target_dir = p.parent
        else:
            target_dir = p

        # Get path relative to project root
        try:
            relative_path = target_dir.relative_to(self.project_root)
        except ValueError:
            # If for some reason it's not relative to project_root,
            # we can't map it to .ov_brain/overviews
            return self.brain_dir / "overviews" / "root.l1.txt"

        # The overview path mirrors the project structure
        # e.g. src/core -> .ov_brain/overviews/src/core.l1.txt
        # If relative_path is '.', we use root.l1.txt
        if str(relative_path) == ".":
            return self.brain_dir / "overviews" / "root.l1.txt"

        overview_file = self.brain_dir / "overviews" / relative_path.with_suffix(".l1.txt")

        # Handle case where relative_path is a directory and with_suffix behaves unexpectedly
        # Path('src/core').with_suffix('.l1.txt') -> Path('src/core.l1.txt')
        # This is actually what we want.

        return overview_file
