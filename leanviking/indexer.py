from pathlib import Path
import lancedb
import pandas as pd
from leanviking.embedder import get_embedder, BaseEmbedder

_INDEX_TABLE = "l0_index"


class Indexer:
    def __init__(self, project_root: str, embedder: BaseEmbedder = None):
        self.project_root = Path(project_root)
        self.index_path = self.project_root / ".ov_brain" / "index"
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.index_path))
        self.embedder = embedder or get_embedder()

    def update_index(self, changed_files: set[str]):
        """Upsert vectors for only the files that have changed."""
        if not changed_files:
            return

        abstracts_dir = self.project_root / ".ov_brain" / "abstracts"
        if not abstracts_dir.exists():
            return

        data = []
        for rel_file in changed_files:
            l0_path = abstracts_dir / f"{rel_file}.l0.txt"
            if not l0_path.exists():
                # L0 not yet generated for this file — skip silently
                continue

            uri = f"viking://{rel_file}"
            content = l0_path.read_text()
            data.append({
                "id": uri,
                "text": content,
                "vector": self.embedder.embed(content),
            })

        if not data:
            return

        df = pd.DataFrame(data)

        try:
            table = self.db.open_table(_INDEX_TABLE)
            # merge_insert: upsert rows matched on "id", insert new rows
            table.merge_insert("id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute(df)
        except Exception:
            # Table doesn't exist yet — fall back to full rebuild
            self.rebuild_index()

    def rebuild_index(self):
        """Full rebuild: re-embed all L0 abstracts and overwrite the table."""
        abstracts_dir = self.project_root / ".ov_brain" / "abstracts"
        if not abstracts_dir.exists():
            return

        data = []
        for file in abstracts_dir.rglob("*.l0.txt"):
            rel = str(file.relative_to(abstracts_dir))
            if rel.endswith(".l0.txt"):
                rel = rel[: -len(".l0.txt")]
            uri = f"viking://{rel}"

            content = file.read_text()
            data.append({
                "id": uri,
                "text": content,
                "vector": self.embedder.embed(content),
            })

        if not data:
            return

        df = pd.DataFrame(data)
        self.db.create_table(_INDEX_TABLE, data=df, mode="overwrite")

    def search(self, query: str, top_k: int = 3) -> list[str]:
        try:
            table = self.db.open_table(_INDEX_TABLE)
        except Exception:
            return []
        query_vec = self.embedder.embed(query)
        results = table.search(query_vec).limit(top_k).to_pandas()
        return results["id"].tolist()
