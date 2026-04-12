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
    results = indexer.search("Summary of the core brain logic")
    assert results[0] == "file1.l0.txt"

def test_search_uninitialized_index(tmp_path):
    # Setup project root but do NOT build index
    test_dir = tmp_path
    indexer = Indexer(project_root=str(test_dir))

    # Search should return empty list, not crash
    results = indexer.search("anything")
    assert results == []
