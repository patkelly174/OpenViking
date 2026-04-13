from pathlib import Path
import os
import warnings
from src.hash_tracker import HashTracker

_QUESTION_PREFIXES = (
    "how", "what", "where", "why", "when", "who", "which",
    "is ", "are ", "does ", "do ", "can ", "could ", "would ",
)


class Brain:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.brain_dir = self.project_root / ".ov_brain"
        self.tracker = HashTracker(self.brain_dir)
        from src.indexer import Indexer
        self.indexer = Indexer(str(self.project_root))
        self._summarizer = None  # lazy — only needed for HyDE

    # ------------------------------------------------------------------
    # URI handling
    # ------------------------------------------------------------------

    def resolve_uri(self, uri: str) -> str:
        if uri.startswith("contextflow://"):
            path_part = uri[len("contextflow://"):]
            # Strip anchor (#symbol) before resolving the file path
            path_part = path_part.split("#")[0]
            path_part = path_part.lstrip("/")
            resolved_path = (self.project_root / path_part).resolve()
            if not resolved_path.is_relative_to(self.project_root):
                raise ValueError(
                    f"Resolved path {resolved_path} is outside the project root "
                    f"{self.project_root}"
                )
            return str(resolved_path)
        return uri

    # ------------------------------------------------------------------
    # Index sync
    # ------------------------------------------------------------------

    def sync_index(self):
        """Check for project changes and incrementally update the index."""
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

    # ------------------------------------------------------------------
    # Context retrieval
    # ------------------------------------------------------------------

    def get_context(self, query: str, top_k: int = 3, include_l2: bool = False) -> str:
        """Return context blocks relevant to *query*.

        Each block contains:
        - The matched file's L0 display_text (specific pointer)
        - The parent directory's L1 overview (orientation)
        - Optionally the raw source (L2) when include_l2=True
        """
        self.sync_index()
        # No more HyDE expansion; use Hybrid Search in indexer
        l0_uris = self.indexer.search(query, top_k=top_k)

        display_texts = self.indexer.get_display_texts(l0_uris)

        context_blocks = []
        # Keep track of directory overviews to limit them to top 2 most frequent
        dir_counts: dict[str, int] = {}
        for uri in l0_uris:
            resolved = self.resolve_uri(uri)
            l1_path = self._get_l1_path(resolved)
            dir_counts[str(l1_path)] = dir_counts.get(str(l1_path), 0) + 1

        # Select top 2 most frequent directories for L1 orientation
        top_l1_keys = sorted(dir_counts, key=dir_counts.get, reverse=True)[:2]
        seen_l1: set[str] = set()

        for uri in l0_uris:
            resolved = self.resolve_uri(uri)
            block_parts = [f"--- [{uri}] ---"]

            # L0: matched file's display summary
            l0_text = display_texts.get(uri)
            if l0_text:
                block_parts.append(l0_text)

            # L1: directory overview (Surgical Gating)
            l1_path = self._get_l1_path(resolved)
            l1_key = str(l1_path)
            if l1_key in top_l1_keys and l1_key not in seen_l1:
                seen_l1.add(l1_key)
                try:
                    l1_content = l1_path.read_text()
                    block_parts.append(f"\n[Directory overview]\n{l1_content}")
                except (FileNotFoundError, IOError):
                    warnings.warn(
                        f"L1 overview not found: {l1_path}. Run 'ov-init' to generate it.",
                        stacklevel=2,
                    )

            # L2: raw source (optional)
            if include_l2:
                try:
                    src_path = Path(resolved)
                    if src_path.is_file():
                        block_parts.append(f"\n[Source]\n{src_path.read_text()}")
                except (FileNotFoundError, IOError):
                    pass

            block_parts.append("---")
            context_blocks.append("\n".join(block_parts))

        return "\n\n".join(context_blocks)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_l1_path(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self.project_root / p

        if p.suffix:
            target_dir = p.parent
        else:
            target_dir = p

        try:
            relative_path = target_dir.relative_to(self.project_root)
        except ValueError:
            return self.brain_dir / "overviews" / "root.l1.txt"

        if str(relative_path) == ".":
            return self.brain_dir / "overviews" / "root.l1.txt"

        return self.brain_dir / "overviews" / relative_path.with_suffix(".l1.txt")

    def dive(self, uri: str) -> str:
        """Resolve a URI and return the raw source content (L2).

        Supports anchored URIs (contextflow://path/file.py#symbol_name) — if the anchor
        matches a known symbol in the file, only that symbol's source is returned.
        Otherwise, the full file is returned.
        """
        try:
            # Split anchor before resolution
            anchor = None
            if "#" in uri:
                uri, anchor = uri.split("#", 1)

            resolved_path = self.resolve_uri(uri)
            path = Path(resolved_path)

            if not path.exists():
                return f"Error: File {uri} not found on disk."

            if not path.is_file():
                return f"Error: {uri} resolves to a directory, not a file."

            content = path.read_text()

            if anchor:
                from src.chunker import chunk_file
                # relative path is needed for chunker
                rel_path = path.relative_to(self.project_root)
                chunks = chunk_file(rel_path, content)
                for chunk in chunks:
                    if chunk.anchor == anchor:
                        return chunk.source

            return content
        except ValueError as e:
            return f"Error resolving URI {uri}: {str(e)}"
        except Exception as e:
            return f"Unexpected error diving into {uri}: {str(e)}"
