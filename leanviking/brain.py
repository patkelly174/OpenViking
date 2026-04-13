from pathlib import Path
import os
import warnings
from leanviking.hash_tracker import HashTracker

_QUESTION_PREFIXES = (
    "how", "what", "where", "why", "when", "who", "which",
    "is ", "are ", "does ", "do ", "can ", "could ", "would ",
)


class Brain:
    def __init__(self, project_root=None):
        self.project_root = Path(project_root or os.getcwd()).resolve()
        self.brain_dir = self.project_root / ".ov_brain"
        self.tracker = HashTracker(self.brain_dir)
        from leanviking.indexer import Indexer
        self.indexer = Indexer(str(self.project_root))
        self._summarizer = None  # lazy — only needed for HyDE

    # ------------------------------------------------------------------
    # URI handling
    # ------------------------------------------------------------------

    def resolve_uri(self, uri: str) -> str:
        if uri.startswith("viking://"):
            path_part = uri[len("viking://"):]
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
        effective_query = self._hyde_query(query)
        l0_uris = self.indexer.search(effective_query, top_k=top_k)

        display_texts = self.indexer.get_display_texts(l0_uris)

        context_blocks = []
        seen_l1: set[str] = set()

        for uri in l0_uris:
            resolved = self.resolve_uri(uri)
            block_parts = [f"--- [{uri}] ---"]

            # L0: matched file's display summary
            l0_text = display_texts.get(uri)
            if l0_text:
                block_parts.append(l0_text)

            # L1: directory overview (deduplicated)
            l1_path = self._get_l1_path(resolved)
            l1_key = str(l1_path)
            if l1_key not in seen_l1:
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
    # HyDE
    # ------------------------------------------------------------------

    def _is_question(self, query: str) -> bool:
        q = query.strip().lower()
        return q.endswith("?") or any(q.startswith(p) for p in _QUESTION_PREFIXES)

    def _hyde_query(self, query: str) -> str:
        """If the query looks like a natural-language question, expand it with HyDE."""
        if not self._is_question(query):
            return query
        import asyncio
        summarizer = self._get_summarizer()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, summarizer.hypothetical_answer(query))
                    return future.result(timeout=10)
            return loop.run_until_complete(summarizer.hypothetical_answer(query))
        except Exception:
            return query

    def _get_summarizer(self):
        if self._summarizer is None:
            from leanviking.summarizer import Summarizer
            self._summarizer = Summarizer()
        return self._summarizer

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

        Supports anchored URIs (viking://path/file.py#symbol_name) — the anchor
        is stripped for file resolution; the full file is returned.
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
