from pathlib import Path
from leanviking.brain import Brain
import pytest


def test_viking_uri_resolution(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    resolved = brain.resolve_uri("viking://src/main.py")
    assert resolved == str(tmp_path / "src" / "main.py")


def test_non_viking_uri(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    uri = "https://google.com"
    assert brain.resolve_uri(uri) == uri


def test_viking_uri_leading_slash(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    # Leading slash in path part should be handled safely
    resolved = brain.resolve_uri("viking:///src/main.py")
    assert resolved == str(tmp_path / "src" / "main.py")


def test_viking_uri_path_traversal(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    # Attempt to escape root
    with pytest.raises(ValueError, match="outside the project root"):
        brain.resolve_uri("viking://../../etc/passwd")


def test_viking_uri_empty_path(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    # viking:// should resolve to project root
    resolved = brain.resolve_uri("viking://")
    assert resolved == str(tmp_path)
