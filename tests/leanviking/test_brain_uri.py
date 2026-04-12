from leanviking.brain import Brain
import pytest
import os

def test_viking_uri_resolution():
    # Mock project root
    brain = Brain(project_root="/tmp/ov_test")
    resolved = brain.resolve_uri("viking://src/main.py")
    assert resolved == "/tmp/ov_test/src/main.py"

def test_non_viking_uri():
    brain = Brain(project_root="/tmp/ov_test")
    uri = "https://google.com"
    assert brain.resolve_uri(uri) == uri

def test_viking_uri_leading_slash():
    brain = Brain(project_root="/tmp/ov_test")
    # Should still resolve relative to root
    resolved = brain.resolve_uri("viking:///src/main.py")
    assert resolved == "/tmp/ov_test/src/main.py"

def test_viking_uri_path_traversal():
    brain = Brain(project_root="/tmp/ov_test")
    # Attempt to escape root
    with pytest.raises(ValueError, match="outside the project root"):
        brain.resolve_uri("viking://../../etc/passwd")

def test_viking_uri_empty_path():
    brain = Brain(project_root="/tmp/ov_test")
    # viking:// should resolve to project root
    resolved = brain.resolve_uri("viking://")
    assert resolved == "/tmp/ov_test"
