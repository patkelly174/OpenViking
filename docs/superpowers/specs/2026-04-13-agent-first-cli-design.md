# Agent-First CLI Design

**Date:** 2026-04-13
**Status:** Approved

## Overview

Refactor the ContextFlow CLI to be agent-first: JSON is the only output format. Human-readable output is removed entirely. Every command always emits a JSON object to stdout with a `status` field. Errors are JSON on stdout with a non-zero exit code.

## Workflow

- **Human:** runs `init` to bootstrap or rebuild the brain
- **Agent:** calls `sync` → `query` → `dive` in normal operation

## Output Contract

All commands print a single JSON object to stdout.

### `init`
```json
{
  "status": "ok",
  "brain_dir": ".ov_brain",
  "files_processed": 12,
  "chunks_indexed": 47,
  "l1_dirs_updated": 3
}
```

### `sync`
```json
{ "status": "ok" }
```

### `query`
```json
{
  "status": "ok",
  "results": [
    {
      "uri": "contextflow://src/auth.py#verify_token",
      "display_text": "...",
      "l1_summary": "..."
    }
  ]
}
```

### `dive`
```json
{
  "status": "ok",
  "uri": "contextflow://src/auth.py#verify_token",
  "content": "..."
}
```

### Errors (all commands)
```json
{ "status": "error", "message": "git ls-files failed: ..." }
```
Exit code: non-zero. Always on stdout.

## Components

### `OutputManager` (cli_utils.py)

Simplified — no mode toggle, always JSON:

- `add(key, value)` — accumulate response fields
- `error(message, code=1)` — print error JSON and raise `typer.Exit`
- `finalize()` — inject `"status": "ok"`, print JSON to stdout

### `cli.py`

- Remove `--json` flag and `@app.callback()`
- Remove all `typer.echo` human-readable output
- All errors routed through `om.error()`
- `init` tracks and reports `files_processed`, `chunks_indexed`, `l1_dirs_updated`
- `sync`, `query`, `dive` updated to match output contracts above

## What Does Not Change

- Command names and arguments
- Core logic (`Brain`, `Indexer`, `Summarizer`, `Chunker`)
- `--force`, `--install-hooks`, `--mode`, `top_k`, `project_root` options
