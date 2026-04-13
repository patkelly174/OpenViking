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
    with pytest.raises(ValueError, match="Path traversal"):
        brain.resolve_uri("contextflow://../../etc/passwd")


def test_contextflow_uri_empty_path(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    # contextflow:// should resolve to project root
    resolved = brain.resolve_uri("contextflow://")
    assert resolved == str(tmp_path)


def test_contextflow_uri_null_byte(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    with pytest.raises(ValueError, match="Null byte"):
        brain.resolve_uri("contextflow://src/brain.py\x00.txt")


def test_contextflow_uri_hidden_traversal(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    # src/../../../etc/passwd — normpath still contains ..
    with pytest.raises(ValueError, match="Path traversal"):
        brain.resolve_uri("contextflow://src/../../../etc/passwd")


def test_contextflow_uri_dotdot_direct(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    with pytest.raises(ValueError, match="Path traversal"):
        brain.resolve_uri("contextflow://../../etc/passwd")


def test_contextflow_uri_anchor_with_traversal(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    with pytest.raises(ValueError, match="Path traversal"):
        brain.resolve_uri("contextflow://../../etc/passwd#symbol")


def test_contextflow_uri_valid_with_anchor(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    resolved = brain.resolve_uri("contextflow://src/brain.py#resolve_uri")
    assert resolved == str(tmp_path / "src" / "brain.py")
