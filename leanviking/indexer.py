from pathlib import Path
import lancedb
import pandas as pd
from leanviking.embedder import Embedder


class Indexer:
    def __init__(self, project_root: str, embedder=None):
        self.project_root = Path(project_root)
        self.index_path = self.project_root / ".ov_brain" / "index"
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.index_path))
        self.embedder = embedder or Embedder()

    def rebuild_index(self):
        abstracts_dir = self.project_root / ".ov_brain" / "abstracts"
        if not abstracts_dir.exists():
            return

        data = []
        for file in abstracts_dir.rglob("*.l0.txt"):
            # abstracts/src/main.py.l0.txt  ->  viking://src/main.py
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
        self.db.create_table("l0_index", data=df, mode="overwrite")

    def search(self, query: str, top_k: int = 3) -> list[str]:
        try:
            table = self.db.open_table("l0_index")
        except Exception:
            return []
        query_vec = self.embedder.embed(query)
        results = table.search(query_vec).limit(top_k).to_pandas()
        return results["id"].tolist()
