from leanviking.brain import Brain
import os

def test_viking_uri_resolution():
    # Mock project root
    brain = Brain(project_root="/tmp/ov_test")
    resolved = brain.resolve_uri("viking://src/main.py")
    assert resolved == "/tmp/ov_test/src/main.py"
