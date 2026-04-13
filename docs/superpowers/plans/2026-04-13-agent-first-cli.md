# Agent-First CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove human-readable CLI output entirely — every command always emits structured JSON to stdout.

**Architecture:** Simplify `OutputManager` to always-JSON with no mode toggle; remove `--json` flag and `@app.callback()` from `cli.py`; update all four commands (`init`, `sync`, `query`, `dive`) to build structured output and route errors through `om.error()`.

**Tech Stack:** Python, Typer, pytest

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/cli_utils.py` | Modify | Always-JSON OutputManager — `add()`, `error()`, `finalize()` |
| `src/cli.py` | Modify | Remove `--json` flag/callback; update all commands; track init counts |
| `tests/test_cli_utils.py` | Create | Unit tests for new OutputManager |
| `tests/test_cli.py` | Modify | Add JSON output assertions to existing tests |

---

### Task 1: Rewrite OutputManager

**Files:**
- Modify: `src/cli_utils.py`
- Create: `tests/test_cli_utils.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cli_utils.py`:

```python
import json
import pytest
import typer
from src.cli_utils import OutputManager


def test_finalize_outputs_json_with_status_ok(capsys):
    om = OutputManager()
    om.finalize()
    out = json.loads(capsys.readouterr().out)
    assert out == {"status": "ok"}


def test_finalize_includes_added_data(capsys):
    om = OutputManager()
    om.add("files_processed", 5)
    om.add("brain_dir", ".ov_brain")
    om.finalize()
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["files_processed"] == 5
    assert out["brain_dir"] == ".ov_brain"


def test_finalize_resets_data_after_call(capsys):
    om = OutputManager()
    om.add("key", "value")
    om.finalize()
    capsys.readouterr()  # clear buffer
    om.finalize()
    out = json.loads(capsys.readouterr().out)
    assert "key" not in out


def test_error_outputs_json_to_stdout(capsys):
    om = OutputManager()
    with pytest.raises(typer.Exit):
        om.error("something went wrong")
    out = json.loads(capsys.readouterr().out)
    assert out == {"status": "error", "message": "something went wrong"}


def test_error_exits_with_given_code():
    om = OutputManager()
    with pytest.raises(typer.Exit) as exc_info:
        om.error("fail", code=2)
    assert exc_info.value.exit_code == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/patrick/projects/ContextFlow
uv run pytest tests/test_cli_utils.py -v
```

Expected: 5 failures — `OutputManager` still has old interface.

- [ ] **Step 3: Rewrite cli_utils.py**

Replace the entire contents of `src/cli_utils.py`:

```python
import json
import typer
from typing import Any


class OutputManager:
    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def add(self, key: str, value: Any) -> None:
        self._data[key] = value

    def error(self, message: str, code: int = 1) -> None:
        print(json.dumps({"status": "error", "message": message}, indent=2))
        raise typer.Exit(code=code)

    def finalize(self) -> None:
        self._data.setdefault("status", "ok")
        print(json.dumps(self._data, indent=2))
        self._data = {}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli_utils.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/cli_utils.py tests/test_cli_utils.py
git commit -m "refactor: OutputManager always-JSON, no mode toggle"
```

---

### Task 2: Update cli.py

**Files:**
- Modify: `src/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Update tests to assert JSON output structure**

Replace the entire contents of `tests/test_cli.py`:

```python
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

        result = runner.invoke(app, ["init", str(tmp_path)])

    instance.generate_l0.assert_not_called()
    data = json.loads(result.output)
    assert data["status"] == "ok"
    assert data["files_processed"] == 0


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
    assert json.loads(result.output)["status"] == "ok"


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
    assert instance.generate_l0.call_count >= 1
    data = json.loads(result.output)
    assert data["status"] == "ok"
    # binary.bin skipped — only main.py counted
    assert data["files_processed"] == 1


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
    l0_files = list((tmp_path / ".ov_brain" / "abstracts").rglob("*.l0.txt"))
    assert len(l0_files) >= 1
    l0_data = json.loads(l0_files[0].read_text())
    assert "search_text" in l0_data
    assert "display_text" in l0_data


def test_ov_init_error_on_git_failure(tmp_path):
    with patch("contextflow.cli.subprocess.run") as mock_run:
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
    assert data["results"][0]["l1_summary"] is None  # file doesn't exist


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
```

- [ ] **Step 2: Run tests to confirm failures**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: several failures on JSON assertions and new tests (`test_sync_outputs_json`, `test_query_*`, `test_dive_*`, `test_ov_init_error_on_git_failure`).

- [ ] **Step 3: Rewrite cli.py**

Replace the entire contents of `src/cli.py`:

```python
import asyncio
import json
import subprocess
from pathlib import Path

import typer
from src.indexer import Indexer
from src.summarizer import Summarizer
from src.hash_tracker import HashTracker
from src.cli_utils import OutputManager
from src.chunker import chunk_file

app = typer.Typer(help="ContextFlow local brain tools.")

om = OutputManager()


def _read_source_file(path: Path) -> str | None:
    """Read a text file. Returns None for binary files."""
    try:
        raw = path.read_bytes()
        if b"\x00" in raw:
            return None
        return raw.decode("utf-8", errors="ignore")
    except (OSError, IsADirectoryError):
        return None


def _install_git_hooks(root: Path) -> None:
    hook_path = root / ".git" / "hooks" / "post-commit"
    hook_content = f"""#!/bin/sh
echo "ContextFlow: syncing brain after commit..."
uv run python -m contextflow.cli sync --project-root {root}
"""
    try:
        hook_path.write_text(hook_content)
        hook_path.chmod(0o755)
    except Exception as e:
        om.error(f"Error installing git hooks: {e}")


@app.command()
def init(
    project_root: str = typer.Argument(default=".", help="Project root directory"),
    mode: str = typer.Option("remote", help="Embedding mode: 'remote' or 'local'"),
    install_hooks: bool = typer.Option(False, "--install-hooks", help="Install git post-commit hooks for auto-sync"),
    force: bool = typer.Option(False, "--force", help="Force a full index rebuild"),
):
    """Bootstrap the .ov_brain directory: generate L0/L1 summaries and rebuild index."""
    asyncio.run(_init_async(Path(project_root).resolve(), mode, install_hooks, force))


async def _init_async(root: Path, mode: str, install_hooks: bool, force: bool):
    brain_dir = root / ".ov_brain"
    abstracts_dir = brain_dir / "abstracts"
    overviews_dir = brain_dir / "overviews"
    abstracts_dir.mkdir(parents=True, exist_ok=True)
    overviews_dir.mkdir(parents=True, exist_ok=True)

    tracker = HashTracker(brain_dir)
    summarizer = Summarizer()

    result = subprocess.run(
        ["git", "ls-files", "--others", "--cached", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        om.error(f"git ls-files failed: {result.stderr.strip()}")
        return

    files = [f for f in result.stdout.strip().split("\n") if f]

    if force:
        hashes_file = brain_dir / "hashes.json"
        if hashes_file.exists():
            hashes_file.unlink()

    # --- L0 generation ---
    l0_tasks = []
    l0_dest_paths = []
    changed_files: set[str] = set()
    files_processed = 0
    chunks_indexed = 0

    for rel_file in files:
        abs_file = root / rel_file
        if not tracker.has_changed(abs_file):
            continue

        changed_files.add(rel_file)
        content = _read_source_file(abs_file)
        if content is None:
            continue

        chunks = chunk_file(rel_file, content)
        files_processed += 1
        chunks_indexed += len(chunks)

        for chunk in chunks:
            anchor_suffix = f"#{chunk.anchor}" if chunk.anchor else ""
            l0_filename = f"{rel_file}{anchor_suffix}.l0.txt"
            l0_path = abstracts_dir / l0_filename
            l0_path.parent.mkdir(parents=True, exist_ok=True)
            l0_tasks.append(summarizer.generate_l0(chunk.source, f"{rel_file}:{chunk.start_line}"))
            l0_dest_paths.append((l0_path, chunk.uri))

    if l0_tasks:
        summaries = await asyncio.gather(*l0_tasks)
        for (path, uri), summary in zip(l0_dest_paths, summaries):
            path.write_text(json.dumps({
                "uri": uri,
                "search_text": summary.search_text,
                "display_text": summary.display_text,
            }, ensure_ascii=False))

    # --- L1 synthesis ---
    dirs_to_files: dict[str, list[str]] = {}
    for rel_file in files:
        parent = str(Path(rel_file).parent)
        dirs_to_files.setdefault(parent, []).append(rel_file)

    l1_tasks = []
    l1_dest_paths = []

    for dir_path, dir_files in dirs_to_files.items():
        dir_changed = any(f in changed_files for f in dir_files)

        child_l0s = []
        for rel_file in dir_files:
            file_l0_path = abstracts_dir / f"{rel_file}.l0.txt"
            if file_l0_path.exists():
                try:
                    data = json.loads(file_l0_path.read_text())
                    child_l0s.append((rel_file, data.get("display_text", "")))
                except Exception:
                    pass
            else:
                pattern = f"{rel_file}#*.l0.txt"
                for p in sorted(abstracts_dir.glob(pattern)):
                    try:
                        data = json.loads(p.read_text())
                        child_l0s.append((rel_file, data.get("display_text", "")))
                    except Exception:
                        pass
                    break

        if not child_l0s or not dir_changed:
            continue

        l1_path = (
            overviews_dir / "root.l1.txt"
            if dir_path == "."
            else overviews_dir / f"{dir_path}.l1.txt"
        )
        l1_path.parent.mkdir(parents=True, exist_ok=True)
        l1_tasks.append(summarizer.generate_l1(dir_path, child_l0s))
        l1_dest_paths.append(l1_path)

    l1_dirs_updated = len(l1_tasks)
    if l1_tasks:
        overviews = await asyncio.gather(*l1_tasks)
        for path, overview in zip(l1_dest_paths, overviews):
            path.write_text(overview)

    tracker.save()

    # --- Rebuild vector index ---
    from src.embedder import get_embedder
    mode_file = brain_dir / "embedding_mode"
    prev_mode = mode_file.read_text().strip() if mode_file.exists() else None
    if prev_mode and prev_mode != mode and not force:
        om.error(
            f"Embedding mode changed from '{prev_mode}' to '{mode}'. "
            "Re-run with --force to rebuild the index with the new mode."
        )
        return
    mode_file.write_text(mode)

    indexer = Indexer(str(root), embedder=get_embedder(mode=mode))
    indexer.rebuild_index()

    if install_hooks:
        _install_git_hooks(root)

    om.add("brain_dir", str(brain_dir))
    om.add("files_processed", files_processed)
    om.add("chunks_indexed", chunks_indexed)
    om.add("l1_dirs_updated", l1_dirs_updated)
    om.finalize()


@app.command()
def sync(
    project_root: str = typer.Option(".", help="Project root directory"),
):
    """Incrementally sync the index for any changed files."""
    from src.brain import Brain
    brain = Brain(project_root)
    brain.sync_index()
    om.finalize()


@app.command()
def query(
    query_text: str,
    top_k: int = typer.Option(3, help="Number of results to return"),
    project_root: str = typer.Option(".", help="Project root directory"),
):
    """Search the brain for context relevant to a topic."""
    from src.brain import Brain
    brain = Brain(project_root)
    brain.sync_index()
    l0_uris = brain.indexer.search(query_text, top_k=top_k)

    display_texts = brain.indexer.get_display_texts(l0_uris)
    results = []
    for uri in l0_uris:
        l1_path = brain._get_l1_path(brain.resolve_uri(uri))
        try:
            l1_summary: str | None = l1_path.read_text()
        except Exception:
            l1_summary = None
        results.append({
            "uri": uri,
            "display_text": display_texts.get(uri),
            "l1_summary": l1_summary,
        })

    om.add("results", results)
    om.finalize()


@app.command()
def dive(
    uri: str,
    project_root: str = typer.Option(".", help="Project root directory"),
):
    """Read the full source of a contextflow:// URI."""
    from src.brain import Brain
    brain = Brain(project_root)
    content = brain.dive(uri)
    om.add("uri", uri)
    om.add("content", content)
    om.finalize()
```

- [ ] **Step 4: Run all CLI tests**

```bash
uv run pytest tests/test_cli.py tests/test_cli_utils.py -v
```

Expected: all pass.

- [ ] **Step 5: Run full test suite to check for regressions**

```bash
uv run pytest -v
```

Expected: all tests pass. No regressions in `test_retrieval.py`, `test_summarizer.py`, `test_breadcrumbs.py`.

- [ ] **Step 6: Commit**

```bash
git add src/cli.py tests/test_cli.py
git commit -m "refactor: agent-first CLI — always JSON, remove --json flag"
```

---

### Task 3: Final verification

- [ ] **Step 1: Confirm no human output paths remain**

```bash
grep -n "typer.echo" /Users/patrick/projects/ContextFlow/src/cli.py
```

Expected: no output. Any match is a bug — remove it.

- [ ] **Step 2: Confirm --json flag is gone**

```bash
grep -n "json_mode\|--json\|app.callback" /Users/patrick/projects/ContextFlow/src/cli.py /Users/patrick/projects/ContextFlow/src/cli_utils.py
```

Expected: no output.

- [ ] **Step 3: Commit plan and close**

```bash
git add docs/superpowers/plans/2026-04-13-agent-first-cli.md
git commit -m "docs: add agent-first CLI implementation plan"
```
