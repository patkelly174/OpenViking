from leanviking.indexer import Indexer
import os
import shutil
from pathlib import Path

def test_index_rebuild_from_texts(tmp_path):
    # Setup mock .ov_brain structure
    test_dir = tmp_path
    abstracts_dir = test_dir / ".ov_brain" / "abstracts"
    abstracts_dir.mkdir(parents=True)

    file1 = abstracts_dir / "file1.l0.txt"
    file1.write_text("Summary of the core brain logic")

    file2 = abstracts_dir / "file2.l0.txt"
    file2.write_text("Summary of the indexer implementation")

    indexer = Indexer(project_root=str(test_dir))
    indexer.rebuild_index()

    # Test searching for content in file1
    results = indexer.search("core brain logic")
    assert any("file1.l0.txt" in str(r) for r in results)
