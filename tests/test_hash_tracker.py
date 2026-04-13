import json
import pytest
from pathlib import Path

from src.hash_tracker import HashTracker


def test_has_changed_stores_relative_path(tmp_path):
    """Keys in hashes.json must be relative to project_root, not absolute."""
    (tmp_path / ".ov_brain").mkdir()
    src = tmp_path / "src"
    src.mkdir()
    f = src / "main.py"
    f.write_text("hello")

    tracker = HashTracker(tmp_path / ".ov_brain", project_root=tmp_path)
    assert tracker.has_changed(f)
    tracker.save()

    hashes = json.loads((tmp_path / ".ov_brain" / "hashes.json").read_text())
    keys = list(hashes.keys())
    assert len(keys) == 1
    assert not Path(keys[0]).is_absolute(), f"Expected relative key, got: {keys[0]}"
    assert keys[0] == "src/main.py"


def test_has_changed_relative_key_unchanged_on_second_call(tmp_path):
    """File unchanged between calls must return False using relative key."""
    (tmp_path / ".ov_brain").mkdir()
    f = tmp_path / "file.txt"
    f.write_text("content")

    tracker = HashTracker(tmp_path / ".ov_brain", project_root=tmp_path)
    assert tracker.has_changed(f) is True
    assert tracker.has_changed(f) is False


def test_load_clears_cache_if_absolute_paths_detected(tmp_path):
    """Legacy absolute-path cache must be discarded to force re-index."""
    brain_dir = tmp_path / ".ov_brain"
    brain_dir.mkdir()
    hashes_file = brain_dir / "hashes.json"
    hashes_file.write_text(json.dumps({
        "/absolute/path/to/file.py": "abc123",
    }))

    tracker = HashTracker(brain_dir, project_root=tmp_path)
    assert tracker.hashes == {}, "Absolute-path cache should be discarded"


def test_load_keeps_relative_path_cache(tmp_path):
    """Valid relative-path cache must be preserved across loads."""
    brain_dir = tmp_path / ".ov_brain"
    brain_dir.mkdir()
    hashes_file = brain_dir / "hashes.json"
    hashes_file.write_text(json.dumps({
        "src/main.py": "deadbeef",
    }))

    tracker = HashTracker(brain_dir, project_root=tmp_path)
    assert tracker.hashes == {"src/main.py": "deadbeef"}
