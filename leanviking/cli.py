import typer

app = typer.Typer(help="OpenViking local brain tools.")


@app.command()
def init(
    project_root: str = typer.Argument(default=".", help="Project root directory"),
):
    """Bootstrap the .ov_brain directory. Full implementation coming in Task 6."""
    raise NotImplementedError(
        "ov-init is not yet implemented. Run after all leanviking tasks are complete."
    )
