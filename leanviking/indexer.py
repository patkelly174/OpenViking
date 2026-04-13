from pathlib import Path
from typing import Optional
import lancedb
import pandas as pd
from leanviking.embedder import get_embedder, BaseEmbedder

_INDEX_TABLE = "l0_index"
# Fetch this many candidates from the vector index before re-ranking.
_OVERFETCH_MULTIPLIER = 6


class Indexer:
    def __init__(
        self,
        project_root: str,
        embedder: BaseEmbedder = None,
        reranker=None,
    ):
        self.project_root = Path(project_root)
        self.index_path = self.project_root / ".ov_brain" / "index"
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.index_path))
        self.embedder = embedder or get_embedder()
        self.reranker = reranker  # None → skip reranking

    def _load_l0(self, path: Path) -> dict | None:
        """Read an L0 file. Returns dict with search_text/display_text keys.

        Supports both the legacy single-string format and the new JSON format.
        """
        import json as _json
        raw = path.read_text().strip()
        try:
            parsed = _json.loads(raw)
            if isinstance(parsed, dict) and "search_text" in parsed:
                return parsed
        except _json.JSONDecodeError:
            pass
        # Legacy: plain text — use for both fields
        return {"search_text": raw, "display_text": raw}

    def update_index(self, changed_files: set[str]):
        """Upsert vectors for only the files that have changed."""
        if not changed_files:
            return

        abstracts_dir = self.project_root / ".ov_brain" / "abstracts"
        if not abstracts_dir.exists():
            return

        data = []
        for rel_file in changed_files:
            # Collect all L0 files for this source file (whole-file + per-symbol chunks)
            file_l0s = list(abstracts_dir.rglob(f"{rel_file}*.l0.txt"))
            if not file_l0s:
                continue

            for l0_path in file_l0s:
                l0 = self._load_l0(l0_path)
                uri = l0.get("uri") if isinstance(l0, dict) else None
                if not uri:
                    rel = str(l0_path.relative_to(abstracts_dir))
                    if rel.endswith(".l0.txt"):
                        rel = rel[: -len(".l0.txt")]
                    uri = f"viking://{rel}"
                data.append({
                    "id": uri,
                    "search_text": l0["search_text"],
                    "display_text": l0["display_text"],
                    "vector": self.embedder.embed(l0["search_text"]),
                })

        if not data:
            return

        df = pd.DataFrame(data)

        try:
            table = self.db.open_table(_INDEX_TABLE)
            table.merge_insert("id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute(df)
        except Exception:
            self.rebuild_index()

    def rebuild_index(self):
        """Full rebuild: re-embed all L0 abstracts and overwrite the table."""
        abstracts_dir = self.project_root / ".ov_brain" / "abstracts"
        if not abstracts_dir.exists():
            return

        data = []
        for file in abstracts_dir.rglob("*.l0.txt"):
            l0 = self._load_l0(file)

            # Prefer the URI stored inside the JSON (includes #anchor if chunked).
            # Fall back to deriving it from the file path for legacy plain-text L0s.
            uri = l0.get("uri") if isinstance(l0, dict) else None
            if not uri:
                rel = str(file.relative_to(abstracts_dir))
                if rel.endswith(".l0.txt"):
                    rel = rel[: -len(".l0.txt")]
                uri = f"viking://{rel}"

            data.append({
                "id": uri,
                "search_text": l0["search_text"],
                "display_text": l0["display_text"],
                "vector": self.embedder.embed(l0["search_text"]),
            })

        if not data:
            return

        df = pd.DataFrame(data)
        self.db.create_table(_INDEX_TABLE, data=df, mode="overwrite")

    def get_display_texts(self, uris: list[str]) -> dict[str, str]:
        """Return {uri: display_text} for the given URIs. Missing URIs are omitted."""
        try:
            table = self.db.open_table(_INDEX_TABLE)
        except Exception:
            return {}
        if not uris:
            return {}
        df = table.to_pandas()
        result = {}
        for uri in uris:
            row = df[df["id"] == uri]
            if not row.empty:
                col = "display_text" if "display_text" in row.columns else "search_text"
                result[uri] = row.iloc[0][col]
        return result

    def search(self, query: str, top_k: int = 3) -> list[str]:
        try:
            table = self.db.open_table(_INDEX_TABLE)
        except Exception:
            return []

        query_vec = self.embedder.embed(query)

        if self.reranker is not None:
            # Over-fetch then re-rank with a cross-encoder for better precision.
            fetch_k = max(top_k * _OVERFETCH_MULTIPLIER, top_k)
            results = table.search(query_vec).limit(fetch_k).to_pandas()
            ids = results["id"].tolist()
            texts = results["search_text"].tolist() if "search_text" in results.columns else ids
            top_indices = self.reranker.rerank(query, texts, top_k=top_k)
            return [ids[i] for i in top_indices]

        results = table.search(query_vec).limit(top_k).to_pandas()
        return results["id"].tolist()
