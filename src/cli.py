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
    hook_content = """#!/bin/sh
echo "ContextFlow: syncing brain after commit..."
uv run python -m contextflow.cli sync --project-root $(git rev-parse --show-toplevel)
"""
    error_msg = None
    try:
        hook_path.write_text(hook_content)
        hook_path.chmod(0o755)
    except Exception as e:
        error_msg = str(e)
    if error_msg is not None:
        om.error(f"Error installing git hooks: {error_msg}")


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
                        break
                    except Exception:
                        pass

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
