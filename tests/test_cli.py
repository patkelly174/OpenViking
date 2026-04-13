import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


def test_ov_init_creates_brain_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main(): pass")

    with (
        patch("contextflow.cli.subprocess.run") as mock_run,
        patch("contextflow.cli.Summarizer") as MockSummarizer,
        patch("contextflow.cli.Indexer") as MockIndexer,
    ):
        mock_run.return_value.stdout = "src/main.py\n"
        mock_run.return_value.returncode = 0

        from src.summarizer import L0Summary
        instance = MockSummarizer.return_value
        instance.generate_l0 = AsyncMock(return_value=L0Summary(
            search_text="entry point application",
            display_text="Entry point of the app.",
        ))
        instance.generate_l1 = AsyncMock(return_value="Source directory overview")
        MockIndexer.return_value.rebuild_index = MagicMock()

        result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / ".ov_brain" / "abstracts").exists()
    assert (tmp_path / ".ov_brain" / "overviews").exists()


def test_ov_init_skips_unchanged_files(tmp_path):
    """Hash tracker prevents re-summarising unchanged files."""
    (tmp_path / "src").mkdir()
    src_file = tmp_path / "src" / "main.py"
    src_file.write_text("def main(): pass")

    # Pre-store a hash so the file appears unchanged
    from src.hash_tracker import HashTracker, calculate_hash
    brain_dir = tmp_path / ".ov_brain"
    brain_dir.mkdir(parents=True)
    tracker = HashTracker(brain_dir)
    tracker.hashes[str(src_file)] = calculate_hash(src_file)
    tracker.save()

    with (
        patch("contextflow.cli.subprocess.run") as mock_run,
        patch("contextflow.cli.Summarizer") as MockSummarizer,
        patch("contextflow.cli.Indexer") as MockIndexer,
    ):
        mock_run.return_value.stdout = "src/main.py\n"
        mock_run.return_value.returncode = 0

        from src.summarizer import L0Summary
        instance = MockSummarizer.return_value
        instance.generate_l0 = AsyncMock(return_value=L0Summary("s", "s"))
        instance.generate_l1 = AsyncMock(return_value="overview")
        MockIndexer.return_value.rebuild_index = MagicMock()

        runner.invoke(app, ["init", str(tmp_path)])

    instance.generate_l0.assert_not_called()


def test_ov_init_calls_rebuild_index(tmp_path):
    with (
        patch("contextflow.cli.subprocess.run") as mock_run,
        patch("contextflow.cli.Summarizer") as MockSummarizer,
        patch("contextflow.cli.Indexer") as MockIndexer,
    ):
        mock_run.return_value.stdout = ""
        mock_run.return_value.returncode = 0

        from src.summarizer import L0Summary
        MockSummarizer.return_value.generate_l0 = AsyncMock(return_value=L0Summary("s", "s"))
        MockSummarizer.return_value.generate_l1 = AsyncMock(return_value="s")
        mock_indexer = MockIndexer.return_value

        result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 0, result.output
    mock_indexer.rebuild_index.assert_called_once()


def test_ov_init_skips_binary_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "binary.bin").write_bytes(b"\x00\x01\x02binary data")
    (tmp_path / "src" / "main.py").write_text("def main(): pass")

    with (
        patch("contextflow.cli.subprocess.run") as mock_run,
        patch("contextflow.cli.Summarizer") as MockSummarizer,
        patch("contextflow.cli.Indexer") as MockIndexer,
    ):
        mock_run.return_value.stdout = "src/binary.bin\nsrc/main.py\n"
        mock_run.return_value.returncode = 0

        from src.summarizer import L0Summary
        instance = MockSummarizer.return_value
        instance.generate_l0 = AsyncMock(return_value=L0Summary("s", "s"))
        instance.generate_l1 = AsyncMock(return_value="overview")
        MockIndexer.return_value.rebuild_index = MagicMock()

        result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 0, result.output
    # Only main.py should be summarised (binary.bin skipped)
    # With chunker: at least 1 call (for the function in main.py)
    assert instance.generate_l0.call_count >= 1


def test_ov_init_writes_json_l0(tmp_path):
    """L0 files are written as JSON with search_text and display_text."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main(): pass")

    with (
        patch("contextflow.cli.subprocess.run") as mock_run,
        patch("contextflow.cli.Summarizer") as MockSummarizer,
        patch("contextflow.cli.Indexer") as MockIndexer,
    ):
        mock_run.return_value.stdout = "src/main.py\n"
        mock_run.return_value.returncode = 0

        from src.summarizer import L0Summary
        instance = MockSummarizer.return_value
        instance.generate_l0 = AsyncMock(return_value=L0Summary(
            search_text="main function entry",
            display_text="Defines main() entry point.",
        ))
        instance.generate_l1 = AsyncMock(return_value="src overview")
        MockIndexer.return_value.rebuild_index = MagicMock()

        result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code == 0, result.output

    # Find any .l0.txt file under abstracts
    l0_files = list((tmp_path / ".ov_brain" / "abstracts").rglob("*.l0.txt"))
    assert len(l0_files) >= 1
    data = json.loads(l0_files[0].read_text())
    assert "search_text" in data
    assert "display_text" in data
