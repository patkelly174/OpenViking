import asyncio
import subprocess
from pathlib import Path

import typer

from leanviking.indexer import Indexer
from leanviking.summarizer import Summarizer

app = typer.Typer(help="OpenViking local brain tools.")


@app.command()
def init(
    project_root: str = typer.Argument(default=".", help="Project root directory"),
):
    """Bootstrap the .ov_brain directory: generate L0/L1 summaries and rebuild index."""
    asyncio.run(_init_async(Path(project_root).resolve()))


async def _init_async(root: Path):

    brain_dir = root / ".ov_brain"
    abstracts_dir = brain_dir / "abstracts"
    overviews_dir = brain_dir / "overviews"
    abstracts_dir.mkdir(parents=True, exist_ok=True)
    overviews_dir.mkdir(parents=True, exist_ok=True)

    summarizer = Summarizer()

    # Crawl project files via git (respects .gitignore automatically)
    result = subprocess.run(
        ["git", "ls-files", "--others", "--cached", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    files = [f for f in result.stdout.strip().split("\n") if f]

    # --- L0 generation (semiautomatic: skip if .l0.txt already exists) ---
    l0_tasks = []
    l0_dest_paths = []

    for rel_file in files:
        abs_file = root / rel_file
        l0_path = abstracts_dir / f"{rel_file}.l0.txt"

        if l0_path.exists():
            continue  # semiautomatic: keep the committed version

        try:
            content = abs_file.read_text(errors="ignore")
        except (OSError, IsADirectoryError):
            continue

        l0_path.parent.mkdir(parents=True, exist_ok=True)
        l0_tasks.append(summarizer.generate_l0(content))
        l0_dest_paths.append(l0_path)

    if l0_tasks:
        summaries = await asyncio.gather(*l0_tasks)
        for path, summary in zip(l0_dest_paths, summaries):
            path.write_text(summary)

    # --- L1 synthesis: one overview per directory ---
    dirs_to_files: dict[str, list[str]] = {}
    for rel_file in files:
        parent = str(Path(rel_file).parent)
        dirs_to_files.setdefault(parent, []).append(rel_file)

    l1_tasks = []
    l1_dest_paths = []

    for dir_path, dir_files in dirs_to_files.items():
        child_l0s = []
        for rel_file in dir_files:
            l0_path = abstracts_dir / f"{rel_file}.l0.txt"
            if l0_path.exists():
                child_l0s.append((Path(rel_file).name, l0_path.read_text()))

        if not child_l0s:
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

    # --- Rebuild vector index ---
    indexer = Indexer(str(root))
    indexer.rebuild_index()

    typer.echo(f"ov-brain initialized at {brain_dir}")
