from pathlib import Path
from typing import Optional
import lancedb
import pandas as pd
from src.embedder import get_embedder, BaseEmbedder

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

        # 1. Vector Search
        query_vec = self.embedder.embed(query)
        vector_results = table.search(query_vec).limit(top_k * _OVERFETCH_MULTIPLIER).to_pandas()
        vector_ids = vector_results["id"].tolist() if not vector_results.empty else []

        # 2. Keyword Search (Simple token-based overlap on search/display text)
        keyword_ids = self._keyword_search(query, table)

        # Merge and deduplicate while preserving order
        merged_ids = []
        seen = set()
        for uid in (keyword_ids + vector_ids):
            if uid not in seen:
                merged_ids.append(uid)
                seen.add(uid)

        if not merged_ids:
            return []

        if self.reranker is not None:
            # Over-fetch then re-rank with a cross-encoder for better precision.
            # We use the merged candidates as the input to the reranker.
            texts = self.get_display_texts(merged_ids)
            # reranker.rerank expects a list of texts in the same order as ids
            ordered_texts = [texts.get(uid, uid) for uid in merged_ids]
            top_indices = self.reranker.rerank(query, ordered_texts, top_k=top_k)
            return [merged_ids[i] for i in top_indices]

        return merged_ids[:top_k]

    def _keyword_search(self, query: str, table: lancedb.Table) -> list[str]:
        """Keyword search using a two-tiered approach: exact phrases and normalized tokens."""
        import re

        # Tier 1: Exact Phrase Match (Preserves ordering and specific identifier structure)
        # We search for the raw query and the lowercase version
        phrase_filters = [
            f"search_text LIKE '%{query}%' OR display_text LIKE '%{query}%'",
            f"search_text LIKE '%{query.lower()}%' OR display_text LIKE '%{query.lower()}%'"
        ]
        combined_phrase_filter = " OR ".join(phrase_filters)

        # Tier 2: Normalized Token Match (Maximizes recall via splitting)
        raw_query = query.lower()
        tokens = re.findall(r'[a-z0-9]+', re.sub(r'([a-z])([A-Z])', r'\1 \2', raw_query))

        token_filters = []
        for token in tokens:
            token_filters.append(f"search_text LIKE '%{token}%' OR display_text LIKE '%{token}%'")
        combined_token_filter = " OR ".join(token_filters)

        merged_ids = []
        try:
            # Execute Phrase Search
            phrase_results = table.search().where(combined_phrase_filter).limit(50).to_pandas()
            if not phrase_results.empty:
                merged_ids.extend(phrase_results["id"].tolist())

            # Execute Token Search (only if we need more candidates or to ensure recall)
            if tokens:
                token_results = table.search().where(combined_token_filter).limit(100).to_pandas()
                if not token_results.empty:
                    # Append token results, deduplicating against phrase results
                    for uid in token_results["id"].tolist():
                        if uid not in merged_ids:
                            merged_ids.append(uid)
        except Exception:
            pass

        return merged_ids
