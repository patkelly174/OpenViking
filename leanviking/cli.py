import asyncio
import subprocess
from pathlib import Path

import typer
from leanviking.indexer import Indexer
from leanviking.summarizer import Summarizer
from leanviking.hash_tracker import HashTracker
from leanviking.cli_utils import OutputManager

app = typer.Typer(help="OpenViking local brain tools.")

# Global output manager — set once by the --json callback
om = OutputManager()

@app.callback()
def main(
    json: bool = typer.Option(False, "--json", help="Output results in JSON format"),
):
    """OpenViking local brain tools."""
    om.json_mode = json

_MAX_FILE_BYTES = 64 * 1024  # 64 KB


def _read_source_file(path: Path) -> str | None:
    """Read a text file for summarization. Returns None for binary or oversized files.
    Returns content with line numbers prefixed.
    """
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        raw = path.read_bytes()
        if b"\x00" in raw:  # NUL byte → binary file
            return None
        text = raw.decode("utf-8", errors="ignore")
        return "\n".join(f"{i+1}: {line}" for i, line in enumerate(text.splitlines()))
    except (OSError, IsADirectoryError):
        return None


def _install_git_hooks(root: Path):
    """Install a post-commit hook that triggers ov-init."""
    hook_path = root / ".git" / "hooks" / "post-commit"
    hook_content = f"""#!/bin/sh
echo "OpenViking: syncing brain after commit..."
uv run python -m leanviking.cli sync --project-root {root}
"""
    try:
        hook_path.write_text(hook_content)
        hook_path.chmod(0o755)
        typer.echo("Git post-commit hook installed successfully.")
    except Exception as e:
        typer.echo(f"Error installing git hooks: {e}", err=True)


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

    # Crawl project files via git (respects .gitignore automatically)
    result = subprocess.run(
        ["git", "ls-files", "--others", "--cached", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        typer.echo(
            f"Error: git ls-files failed: {result.stderr.strip()}", err=True
        )
        raise typer.Exit(code=1)
    files = [f for f in result.stdout.strip().split("\n") if f]

    if force:
        # Clear hashes to force a full rebuild
        hashes_file = brain_dir / "hashes.json"
        if hashes_file.exists():
            hashes_file.unlink()

    # --- L0 generation (smart: update if hash changed) ---
    l0_tasks = []
    l0_dest_paths = []
    changed_files = set()

    for rel_file in files:
        abs_file = root / rel_file
        l0_path = abstracts_dir / f"{rel_file}.l0.txt"

        if not tracker.has_changed(abs_file):
            continue

        changed_files.add(rel_file)
        content = _read_source_file(abs_file)
        if content is None:
            continue

        l0_path.parent.mkdir(parents=True, exist_ok=True)
        l0_tasks.append(summarizer.generate_l0(content, rel_file))
        l0_dest_paths.append(l0_path)

    if l0_tasks:
        summaries = await asyncio.gather(*l0_tasks)
        for path, summary in zip(l0_dest_paths, summaries):
            path.write_text(summary)

    # --- L1 synthesis: update if any child file changed ---
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
            l0_path = abstracts_dir / f"{rel_file}.l0.txt"
            if l0_path.exists():
                child_l0s.append((rel_file, l0_path.read_text()))

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

    if l1_tasks:
        overviews = await asyncio.gather(*l1_tasks)
        for path, overview in zip(l1_dest_paths, overviews):
            path.write_text(overview)

    tracker.save()

    # --- Rebuild vector index ---
    from leanviking.embedder import get_embedder, _DEFAULT_REMOTE_MODEL, _DEFAULT_LOCAL_MODEL
    # Detect embedding mode change to prevent dimension mismatch in LanceDB
    mode_file = brain_dir / "embedding_mode"
    prev_mode = mode_file.read_text().strip() if mode_file.exists() else None
    if prev_mode and prev_mode != mode and not force:
        typer.echo(
            f"Warning: embedding mode changed from '{prev_mode}' to '{mode}'. "
            "Re-run with --force to rebuild the index with the new mode.",
            err=True,
        )
        raise typer.Exit(code=1)
    mode_file.write_text(mode)

    indexer = Indexer(str(root), embedder=get_embedder(mode=mode))
    indexer.rebuild_index()

    if install_hooks:
        _install_git_hooks(root)

    om.echo(f"ov-brain initialized at {brain_dir}")
    om.finalize()


@app.command()
def sync(
    project_root: str = typer.Option(".", help="Project root directory"),
):
    """Incrementally sync the index for any changed files."""
    from leanviking.brain import Brain
    brain = Brain(project_root)
    brain.sync_index()
    om.echo("Brain synced.")
    om.finalize()


@app.command()
def query(
    query_text: str,
    top_k: int = typer.Option(3, help="Number of results to return"),
    project_root: str = typer.Option(".", help="Project root directory"),
):
    """Search the brain for context relevant to a topic."""
    from leanviking.brain import Brain
    brain = Brain(project_root)

    # Always sync before searching — both modes get fresh results
    brain.sync_index()
    l0_uris = brain.indexer.search(query_text, top_k=top_k)

    if not l0_uris:
        om.echo("No relevant context found in the brain.")
        om.finalize()
        return

    if om.json_mode:
        results = []
        for uri in l0_uris:
            l1_path = brain._get_l1_path(brain.resolve_uri(uri))
            try:
                summary = l1_path.read_text()
            except Exception:
                summary = None
            results.append({"uri": uri, "summary": summary})
        om.add_data("results", results)
    else:
        context = brain.get_context(query_text, top_k=top_k)
        typer.echo(context)

    om.finalize()


@app.command()
def dive(
    uri: str,
    project_root: str = typer.Option(".", help="Project root directory"),
):
    """Read the full source of a viking:// URI found in a brain query."""
    from leanviking.brain import Brain
    brain = Brain(project_root)

    content = brain.dive(uri)

    if om.json_mode:
        om.add_data("uri", uri)
        om.add_data("content", content)
    else:
        typer.echo(content)

    om.finalize()
