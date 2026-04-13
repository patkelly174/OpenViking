from pathlib import Path
import os
import warnings
from leanviking.hash_tracker import HashTracker

class Brain:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.brain_dir = self.project_root / ".ov_brain"
        self.tracker = HashTracker(self.brain_dir)
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

    def sync_index(self):
        """Check for project changes and incrementally update the index."""
        # Find files that have changed since last sync
        # Using git ls-files to get current project state
        import subprocess
        result = subprocess.run(
            ["git", "ls-files", "--others", "--cached", "--exclude-standard"],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return

        files = [f for f in result.stdout.strip().split("\n") if f]
        changed_files = set()

        for rel_file in files:
            abs_file = self.project_root / rel_file
            if self.tracker.has_changed(abs_file):
                changed_files.add(rel_file)

        if changed_files:
            self.indexer.update_index(changed_files)
            self.tracker.save()

    def get_context(self, query: str, top_k: int = 3) -> str:
        self.sync_index()
        l0_uris = self.indexer.search(query, top_k=top_k)

        context_blocks = []
        seen_l1: set[str] = set()

        for uri in l0_uris:
            resolved = self.resolve_uri(uri)

            # L1: directory overview (deduplicated)
            l1_path = self._get_l1_path(resolved)
            l1_key = str(l1_path)
            if l1_key not in seen_l1:
                seen_l1.add(l1_key)
                try:
                    content = l1_path.read_text()
                    # Wrap L1 content with the URI that triggered this retrieval
                    context_blocks.append(f"--- [{uri}] ---\n{content}\n---")
                except (FileNotFoundError, IOError):
                    warnings.warn(
                        f"L1 overview not found: {l1_path}. Run 'ov-init' to generate it.",
                        stacklevel=2,
                    )

        return "\n\n".join(context_blocks)

    def _get_l1_path(self, path: str) -> Path:
        # Ensure we have a Path object
        p = Path(path)
        if not p.is_absolute():
            p = self.project_root / p

        # If it's a file, we want the overview of its containing directory
        # Use suffix check instead of is_file() to avoid disk hits and handle virtual paths
        # Paths with a file extension are treated as files; extensionless paths as dirs.
        # This avoids disk hits on virtual/non-existent paths in the index.
        if p.suffix:
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

    def dive(self, uri: str) -> str:
        """
        Resolve a URI and return the raw source content (L2).
        """
        try:
            resolved_path = self.resolve_uri(uri)
            path = Path(resolved_path)

            if not path.exists():
                return f"Error: File {uri} not found on disk."

            if not path.is_file():
                return f"Error: {uri} resolves to a directory, not a file."

            return path.read_text()
        except ValueError as e:
            return f"Error resolving URI {uri}: {str(e)}"
        except Exception as e:
            return f"Unexpected error diving into {uri}: {str(e)}"
