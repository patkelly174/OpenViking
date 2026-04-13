import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from leanviking.brain import Brain

def test_get_context_flow():
    # Setup Brain and mocks
    brain = Brain(project_root="/tmp/project")
    brain.indexer = MagicMock()

    # Mock indexer.search to return a specific L0 file
    brain.indexer.search.return_value = ["viking://src/main.py"]

    # Mock the filesystem for the L1 overview
    # The expected L1 path should be /tmp/project/.ov_brain/overviews/src.l1.txt
    l1_content = "L1 Content"
    expected_l1_path = Path("/tmp/project/.ov_brain/overviews/src.l1.txt")

    with patch("pathlib.Path.read_text") as mock_read:
        mock_read.return_value = l1_content

        # Execute
        result = brain.get_context("query")

        # Assert
        assert result == l1_content
        brain.indexer.search.assert_called_once_with("query", top_k=3)

        # Verify that read_text was called on the correct L1 path
        # We check if any of the calls to read_text were made on a Path object that matches expected_l1_path
        # Since mock_read is patched on the class, we can check the call arguments if they are Path objects
        # or check the 'self' of the call if we use a different approach.
        # Actually, read_text is called on the instance returned by _get_l1_path.
        # In our current implementation, read_text is called as `l1_path.read_text()`.
        # The patched mock_read will be called. We need to verify the path.

        # To properly verify the path, we should check the arguments of the call to read_text.
        # Wait, read_text() takes no arguments. The object it's called on is the Path.
        # When we patch Path.read_text, the first argument to the mock is 'self'.
        called_path = mock_read.call_args[0][0] if mock_read.call_args else None
        assert called_path == expected_l1_path

def test_get_context_uri_resolution():
    brain = Brain(project_root="/tmp/project")
    brain.indexer = MagicMock()

    # Test case: file URI should resolve to parent directory's L1 overview
    brain.indexer.search.return_value = ["viking://src/utils/helper.py"]
    l1_content = "Utils Overview"
    expected_l1_path = Path("/tmp/project/.ov_brain/overviews/src/utils.l1.txt")

    with patch("pathlib.Path.read_text") as mock_read:
        mock_read.return_value = l1_content

        result = brain.get_context("query")

        assert result == l1_content
        called_path = mock_read.call_args[0][0] if mock_read.call_args else None
        assert called_path == expected_l1_path
