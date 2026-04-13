from pathlib import Path
from src.brain import Brain
import pytest


def test_contextflow_uri_resolution(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    resolved = brain.resolve_uri("contextflow://src/main.py")
    assert resolved == str(tmp_path / "src" / "main.py")


def test_non_contextflow_uri(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    uri = "https://google.com"
    assert brain.resolve_uri(uri) == uri


def test_contextflow_uri_leading_slash(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    # Leading slash in path part should be handled safely
    resolved = brain.resolve_uri("contextflow:///src/main.py")
    assert resolved == str(tmp_path / "src" / "main.py")


def test_contextflow_uri_path_traversal(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    # Attempt to escape root
    with pytest.raises(ValueError, match="outside the project root"):
        brain.resolve_uri("contextflow://../../etc/passwd")


def test_contextflow_uri_empty_path(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    # contextflow:// should resolve to project root
    resolved = brain.resolve_uri("contextflow://")
    assert resolved == str(tmp_path)
