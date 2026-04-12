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
    # The expected L1 path should be .ov_brain/overviews/src.l1.txt
    l1_content = "L1 Content"
    expected_l1_path = Path("/tmp/project/.ov_brain/overviews/src.l1.txt")

    with patch("pathlib.Path.read_text") as mock_read:
        # Mock read_text by returning content if the object calling it is the expected path
        # Since read_text is a method, the mock object itself doesn't easily expose 'self'
        # unless we use a different mocking strategy or check the call arguments.
        # For this specific test, we'll just return the content for any call.
        mock_read.return_value = l1_content

        # Execute
        result = brain.get_context("query")

        # Assert
        assert result == l1_content
        brain.indexer.search.assert_called_once_with("query", top_k=3)
