---
name: openviking-agent-native
description: Capabilities and usage guide for OpenViking's agent-native CLI.
type: skill
---

# OpenViking Agent-Native CLI

OpenViking provides a structured interface for AI agents to interact with a project's local knowledge brain. It uses a "Query-then-Dive" workflow to minimize context window usage.

## Core Workflow
1. **Query**: Use `ov-query` to find the most relevant files and their summaries.
2. **Dive**: Use `ov-dive` to read the full source code of specific files identified in the query.
3. **Sync**: Use `ov-init` if the agent knows the codebase has changed significantly.

## Command Reference

### `ov-query`
Semantic search across the project's L0/L1 index.
- **Usage**: `python -m leanviking.cli query "your search term" --json`
- **JSON Response**:
  ```json
  {
    "results": [
      {
        "uri": "viking://src/main.py",
        "summary": "L1: Core logic for ...\nL0: ..."
      }
    ],
    "synced": true
  }
  ```

### `ov-dive`
Retrieve the full content of a specific file using its URI.
- **Usage**: `python -m leanviking.cli dive "viking://src/main.py" --json`
- **JSON Response**:
  ```json
  {
    "uri": "viking://src/main.py",
    "content": "def main():\n    print('Hello World')"
  }
  ```

### `ov-init`
Initialize or force-rebuild the brain.
- **Usage**: `python -m leanviking.cli init . --json`
- **Flags**:
    - `--force`: Clears all hashes and performs a complete rebuild.
    - `--mode local`: Uses local embeddings instead of OpenAI.
    - `--install-hooks`: Sets up git post-commit hooks for automatic syncing.

## Strategic Guidance for Agents
- **Context Efficiency**: Do not use `ov-dive` on every file. Use `ov-query` first to identify the top 3 candidates.
- **URI Format**: Always use the `viking://` URI format when calling `ov-dive`.
- **Freshness**: If the user mentions they just modified a file, the `ov-query` command will automatically sync the index, but you can also call `ov-init` to be sure.
