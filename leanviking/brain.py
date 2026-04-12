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

    def get_context(self, query: str, top_k=3) -> str:
        # Sensing: L0 Search
        l0_paths = self.indexer.search(query, top_k=top_k)

        context_blocks = []
        for path in l0_paths:
            # Positioning: L0 -> L1 jump
            l1_path = self._get_l1_path(path)
            try:
                content = l1_path.read_text()
                context_blocks.append(content)
            except (FileNotFoundError, IOError):
                continue

        return "\n\n".join(context_blocks)

    def _get_l1_path(self, path: str) -> Path:
        # Ensure we have a Path object
        p = Path(path)
        if not p.is_absolute():
            p = self.project_root / p

        # If it's a file, we want the overview of its containing directory
        if p.is_file() or not p.is_dir():
            # We don't check is_file() on disk because it might be a virtual path
            # or the file might not exist yet. We treat it as a file if it has an extension
            # or if we just want the parent's overview.
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
