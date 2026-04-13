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
        patch("src.cli.subprocess.run") as mock_run,
        patch("src.cli.Summarizer") as MockSummarizer,
        patch("src.cli.Indexer") as MockIndexer,
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
    data = json.loads(result.output)
    assert data["status"] == "ok"
    assert "brain_dir" in data
    assert "files_processed" in data
    assert "chunks_indexed" in data
    assert "l1_dirs_updated" in data


def test_ov_init_skips_unchanged_files(tmp_path):
    """Hash tracker prevents re-summarising unchanged files."""
    (tmp_path / "src").mkdir()
    src_file = tmp_path / "src" / "main.py"
    src_file.write_text("def main(): pass")

    from src.hash_tracker import HashTracker, calculate_hash
    brain_dir = tmp_path / ".ov_brain"
    brain_dir.mkdir(parents=True)
    tracker = HashTracker(brain_dir, project_root=tmp_path)
    tracker.hashes["src/main.py"] = calculate_hash(src_file)
    tracker.save()

    with (
        patch("src.cli.subprocess.run") as mock_run,
        patch("src.cli.Summarizer") as MockSummarizer,
        patch("src.cli.Indexer") as MockIndexer,
    ):
        mock_run.return_value.stdout = "src/main.py\n"
        mock_run.return_value.returncode = 0

        from src.summarizer import L0Summary
        instance = MockSummarizer.return_value
        instance.generate_l0 = AsyncMock(return_value=L0Summary("s", "s"))
        instance.generate_l1 = AsyncMock(return_value="overview")
        MockIndexer.return_value.rebuild_index = MagicMock()

        result = runner.invoke(app, ["init", str(tmp_path)])

    instance.generate_l0.assert_not_called()
    data = json.loads(result.output)
    assert data["status"] == "ok"
    assert data["files_processed"] == 0


def test_ov_init_calls_rebuild_index(tmp_path):
    with (
        patch("src.cli.subprocess.run") as mock_run,
        patch("src.cli.Summarizer") as MockSummarizer,
        patch("src.cli.Indexer") as MockIndexer,
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
    assert json.loads(result.output)["status"] == "ok"


def test_ov_init_skips_binary_files(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "binary.bin").write_bytes(b"\x00\x01\x02binary data")
    (tmp_path / "src" / "main.py").write_text("def main(): pass")

    with (
        patch("src.cli.subprocess.run") as mock_run,
        patch("src.cli.Summarizer") as MockSummarizer,
        patch("src.cli.Indexer") as MockIndexer,
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
    assert instance.generate_l0.call_count >= 1
    data = json.loads(result.output)
    assert data["status"] == "ok"
    assert data["files_processed"] == 1


def test_ov_init_writes_json_l0(tmp_path):
    """L0 files are written as JSON with search_text and display_text."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def main(): pass")

    with (
        patch("src.cli.subprocess.run") as mock_run,
        patch("src.cli.Summarizer") as MockSummarizer,
        patch("src.cli.Indexer") as MockIndexer,
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
    l0_files = list((tmp_path / ".ov_brain" / "abstracts").rglob("*.l0.txt"))
    assert len(l0_files) >= 1
    l0_data = json.loads(l0_files[0].read_text())
    assert "search_text" in l0_data
    assert "display_text" in l0_data


def test_ov_init_error_on_git_failure(tmp_path):
    with patch("src.cli.subprocess.run") as mock_run:
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "not a git repo"

        result = runner.invoke(app, ["init", str(tmp_path)])

    assert result.exit_code != 0
    data = json.loads(result.output)
    assert data["status"] == "error"
    assert "git ls-files failed" in data["message"]


def test_sync_outputs_json(tmp_path):
    with patch("src.brain.Brain") as MockBrain:
        MockBrain.return_value.sync_index = MagicMock()
        result = runner.invoke(app, ["sync", "--project-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["status"] == "ok"


def test_query_outputs_json_results(tmp_path):
    with patch("src.brain.Brain") as MockBrain:
        mock_brain = MockBrain.return_value
        mock_brain.sync_index = MagicMock()
        mock_brain.indexer.search.return_value = ["contextflow://src/auth.py#verify_token"]
        mock_brain.indexer.get_display_texts.return_value = {
            "contextflow://src/auth.py#verify_token": "verify_token validates JWT"
        }
        mock_brain.resolve_uri.return_value = str(tmp_path / "src" / "auth.py")
        mock_brain._get_l1_path.return_value = tmp_path / ".ov_brain" / "overviews" / "src.l1.txt"

        result = runner.invoke(app, ["query", "JWT auth", "--project-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["status"] == "ok"
    assert len(data["results"]) == 1
    assert data["results"][0]["uri"] == "contextflow://src/auth.py#verify_token"
    assert data["results"][0]["display_text"] == "verify_token validates JWT"
    assert data["results"][0]["l1_summary"] is None


def test_query_empty_results(tmp_path):
    with patch("src.brain.Brain") as MockBrain:
        mock_brain = MockBrain.return_value
        mock_brain.sync_index = MagicMock()
        mock_brain.indexer.search.return_value = []
        mock_brain.indexer.get_display_texts.return_value = {}

        result = runner.invoke(app, ["query", "nothing", "--project-root", str(tmp_path)])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["status"] == "ok"
    assert data["results"] == []


def test_dive_outputs_json(tmp_path):
    with patch("src.brain.Brain") as MockBrain:
        MockBrain.return_value.dive.return_value = "def verify_token(): ..."

        result = runner.invoke(app, [
            "dive", "contextflow://src/auth.py#verify_token",
            "--project-root", str(tmp_path),
        ])

    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["status"] == "ok"
    assert data["uri"] == "contextflow://src/auth.py#verify_token"
    assert data["content"] == "def verify_token(): ..."
