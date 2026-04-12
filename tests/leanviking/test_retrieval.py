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
        # Mock read_text only when called for the expected_l1_path
        def side_effect(self, *args, **kwargs):
            if str(self) == str(expected_l1_path):
                return l1_content
            raise FileNotFoundError(f"No such file: {self}")

        mock_read.side_effect = side_effect

        # Execute
        result = brain.get_context("query")

        # Assert
        assert result == l1_content
        brain.indexer.search.assert_called_once_with("query", top_k=3)
