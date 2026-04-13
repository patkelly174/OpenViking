# ContextFlow

Agent-native context database. Solves context window exhaustion via a structured knowledge brain.

## 🎯 Goal

ContextFlow minimizes token usage and prevents context "drift" by providing AI agents with a **"Query-then-Dive"** navigation pattern. Instead of stuffing entire files into a prompt, agents search a semantic index of summaries and only "dive" into the raw source code when needed.

---

## 🚀 Getting Started

### Prerequisites

- **Python**: $\ge$ 3.10
- **Rust Toolchain**: Required for the core CLI build.
- **CMake**: $\ge$ 3.15

### Installation

```bash
pip install .
# or using uv
uv pip install .
```

### Initialization

Build the knowledge brain for your project:

```bash
python -m contextflow.cli init .
```

**Common Flags:**

- `--force`: Clear existing hashes and perform a complete rebuild.
- `--mode local`: Use local embeddings (fastembed) instead of OpenAI.
- `--install-hooks`: Automatically set up git post-commit hooks to keep the brain in sync.

### Usage (Agent Workflow)

The intended workflow for an AI agent is:

1. **Query**: Search for a concept or symbol.

   ```bash
   python -m contextflow.cli query "how is auth handled" --json
   ```

   _Returns a list of `contextflow://` URIs and their L0 summaries._

2. **Dive**: Retrieve the actual source code for a specific URI.

   ```bash
   python -m contextflow.cli dive "contextflow://src/auth.py#login_user" --json
   ```

   _Returns the full raw content of the file or the specific symbol._

3. **Sync**: Refresh the index after changes.
   ```bash
   python -m contextflow.cli init .
   ```

---

## 🧠 Under the Hood

ContextFlow implements a hierarchical indexing strategy to balance precision and context.

### 1. Hierarchical Indexing (L0 $\to$ L1 $\to$ L2)

- **L0 (Vector Layer)**: Granular summaries of symbols (functions, classes). These are stored as `.l0.txt` files and embedded into a **LanceDB** vector table.
- **L1 (Orientation Layer)**: Directory-level overviews (`.l1.txt`) that describe the purpose of a folder. When an agent finds an L0 match, the corresponding L1 overview is provided to give the agent "spatial orientation" in the codebase.
- **L2 (Source Layer)**: The raw source code on disk. This is only accessed during a `dive` operation.

### 2. The Retrieval Pipeline

ContextFlow doesn't just do simple vector search; it uses a **Retrieve & Re-rank** pipeline:

1. **HyDE (Hypothetical Document Embeddings)**: For natural language questions, OpenViking generates a "fake" answer first and uses _that_ to search the vector space, significantly improving hit rates.
2. **Over-fetching**: The system fetches more candidates than requested (typically 6x).
3. **Re-ranking**: A cross-encoder reranker scores the candidates against the original query to ensure the top results are the most relevant.

### 3. The `viking://` URI

The `contextflow://` protocol is the primary pointer in the system.

- `contextflow://path/to/file.py` $\to$ Points to a whole file.
- `contextflow://path/to/file.py#symbol` $\to$ Points to a specific function/class.
  The `Brain` resolver ensures these URIs are resolved safely within the project root.

### 4. Incremental Syncing

To avoid rebuilding the entire brain on every change:

- **Git Tracking**: Uses `git ls-files` to identify tracked files.
- **Hash Tracking**: Maintains a store of file hashes. Only modified files are re-indexed.
- **Upsert**: Uses LanceDB's `merge_insert` to update existing vectors or insert new ones without duplicating entries.

### 5. Memory System

Beyond code, ContextFlow tracks "Truths"—persistent memories of project decisions, user preferences, and architectural constraints stored as markdown files in `.ov_brain/memories/`.
