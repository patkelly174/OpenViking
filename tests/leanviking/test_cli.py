import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from leanviking.cli import app

runner = CliRunner()


def test_ov_init_creates_brain_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main(): pass")

    with (
        patch("subprocess.run") as mock_run,
        patch("leanviking.cli.Summarizer") as MockSummarizer,
        patch("leanviking.cli.Indexer") as MockIndexer,
    ):
        mock_run.return_value.stdout = "src/main.py\n"
        mock_run.return_value.returncode = 0

        instance = MockSummarizer.return_value
        instance.generate_l0 = AsyncMock(return_value="Entry point of the app")
        instance.generate_l1 = AsyncMock(return_value="Source directory overview")

        MockIndexer.return_value.rebuild_index = MagicMock()

        result = runner.invoke(app, [str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / ".ov_brain" / "abstracts").exists()
    assert (tmp_path / ".ov_brain" / "overviews").exists()


def test_ov_init_skips_existing_l0(tmp_path):
    """Semiautomatic: if .l0.txt already exists, don't regenerate it."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main(): pass")

    # Pre-create L0
    abstracts_dir = tmp_path / ".ov_brain" / "abstracts" / "src"
    abstracts_dir.mkdir(parents=True)
    existing_l0 = abstracts_dir / "main.py.l0.txt"
    existing_l0.write_text("Existing summary — should not be replaced")

    with (
        patch("subprocess.run") as mock_run,
        patch("leanviking.cli.Summarizer") as MockSummarizer,
        patch("leanviking.cli.Indexer") as MockIndexer,
    ):
        mock_run.return_value.stdout = "src/main.py\n"
        mock_run.return_value.returncode = 0

        instance = MockSummarizer.return_value
        instance.generate_l0 = AsyncMock(return_value="New summary")
        instance.generate_l1 = AsyncMock(return_value="Overview")
        MockIndexer.return_value.rebuild_index = MagicMock()

        runner.invoke(app, [str(tmp_path)])

    # L0 must be unchanged
    assert existing_l0.read_text() == "Existing summary — should not be replaced"
    instance.generate_l0.assert_not_called()


def test_ov_init_calls_rebuild_index(tmp_path):
    with (
        patch("subprocess.run") as mock_run,
        patch("leanviking.cli.Summarizer") as MockSummarizer,
        patch("leanviking.cli.Indexer") as MockIndexer,
    ):
        mock_run.return_value.stdout = ""
        mock_run.return_value.returncode = 0

        MockSummarizer.return_value.generate_l0 = AsyncMock(return_value="s")
        MockSummarizer.return_value.generate_l1 = AsyncMock(return_value="s")
        mock_indexer = MockIndexer.return_value

        runner.invoke(app, [str(tmp_path)])

    mock_indexer.rebuild_index.assert_called_once()
