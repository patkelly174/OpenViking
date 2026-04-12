import os
from pathlib import Path
import lancedb
import pandas as pd
import numpy as np

class Indexer:
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.index_path = self.project_root / ".ov_brain" / "index"
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.index_path))

    def rebuild_index(self):
        abstracts_dir = self.project_root / ".ov_brain" / "abstracts"
        if not abstracts_dir.exists():
            return

        data = []
        for file in abstracts_dir.glob("*.l0.txt"):
            content = file.read_text()
            data.append({
                "id": str(file.name),
                "text": content,
                "vector": self._embed(content)
            })

        if not data:
            return

        df = pd.DataFrame(data)
        # Create or overwrite the table
        self.db.create_table("l0_index", data=df, mode="overwrite")

    def search(self, query: str, top_k=3):
        try:
            table = self.db.open_table("l0_index")
        except Exception:
            return []
        query_vec = self._embed(query)
        results = table.search(query_vec).limit(top_k).to_pandas()
        return results["id"].tolist()

    def _embed(self, text: str):
        # Simple placeholder embedding:
        # For a real implementation, this would use a model like sentence-transformers.
        # To make tests deterministic, we use a hash of the text.
        import hashlib
        hash_val = int(hashlib.sha256(text.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        return np.random.rand(1536).tolist()
