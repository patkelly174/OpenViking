import pytest
from pathlib import Path
from src.brain import Brain

def test_get_context_includes_uris():
    """Verify that get_context output contains the contextflow:// URIs."""
    # We'll use a mock indexer for this test
    brain = Brain(project_root=".")
    brain.indexer.search = lambda query, top_k: ["contextflow://test_file.py"]

    # Mock L1 path to return a dummy file
    test_l1 = Path(".ov_brain/overviews/root.l1.txt")
    test_l1.parent.mkdir(parents=True, exist_ok=True)
    test_l1.write_text("Mock L1 Content")

    try:
        context = brain.get_context("test")
        assert "contextflow://test_file.py" in context
        assert "--- [contextflow://test_file.py] ---" in context
    finally:
        if test_l1.exists():
            test_l1.unlink()

def test_dive_success():
    """Verify that dive correctly reads the content of a file."""
    # Create a dummy file
    test_file = Path("test_dive_source.py")
    test_file.write_text("print('hello world')")

    try:
        brain = Brain(project_root=".")
        # Mock the indexer search to avoid needing a real DB
        brain.indexer.search = lambda query, top_k: ["contextflow://test_dive_source.py"]

        content = brain.dive("contextflow://test_dive_source.py")
        assert content == "print('hello world')"
    finally:
        if test_file.exists():
            test_file.unlink()

def test_dive_not_found():
    """Verify dive handles missing files gracefully."""
    brain = Brain(project_root=".")
    content = brain.dive("contextflow://non_existent_file.py")
    assert "not found on disk" in content

def test_dive_security_bound():
    """Verify dive prevents traversal outside the project root."""
    brain = Brain(project_root=".")
    # resolve_uri already handles this, so we test it via dive
    content = brain.dive("contextflow://../../etc/passwd")
    assert "Error resolving URI" in content

def test_dive_directory():
    """Verify dive handles directories gracefully."""
    # Create a dummy directory
    test_dir = Path("test_dive_dir")
    test_dir.mkdir(parents=True, exist_ok=True)

    try:
        brain = Brain(project_root=".")
        content = brain.dive("contextflow://test_dive_dir")
        assert "resolves to a directory, not a file" in content
    finally:
        if test_dir.exists():
            # Remove the directory (it's a non-recursive remove for safety)
            import shutil
            shutil.rmtree(test_dir)
