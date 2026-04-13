import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.brain import Brain


def _write_l0(path: Path, search_text: str, display_text: str, uri: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"uri": uri, "search_text": search_text, "display_text": display_text}))


def test_get_context_returns_l0_and_l1(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    brain.indexer = MagicMock()
    brain.indexer.search.return_value = ["viking://src/main.py"]
    brain.indexer.get_display_texts.return_value = {
        "viking://src/main.py": "main.py: entry point, exports main() at src/main.py:1."
    }

    # Write an L1 overview
    overviews_dir = tmp_path / ".ov_brain" / "overviews"
    overviews_dir.mkdir(parents=True)
    (overviews_dir / "src.l1.txt").write_text("src/ — application source code.")

    result = brain.get_context("query")

    assert "main.py: entry point" in result
    assert "src/ — application source code." in result
    brain.indexer.search.assert_called_once_with(
        brain._hyde_query("query"), top_k=3
    )


def test_get_context_deduplicates_l1(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    brain.indexer = MagicMock()
    brain.indexer.search.return_value = [
        "viking://src/a.py",
        "viking://src/b.py",
        "viking://src/c.py",
    ]
    brain.indexer.get_display_texts.return_value = {
        "viking://src/a.py": "A summary.",
        "viking://src/b.py": "B summary.",
        "viking://src/c.py": "C summary.",
    }

    overviews_dir = tmp_path / ".ov_brain" / "overviews"
    overviews_dir.mkdir(parents=True)
    (overviews_dir / "src.l1.txt").write_text("src module handles routing and controllers")

    result = brain.get_context("query", top_k=3)
    assert result.count("src module handles routing and controllers") == 1


def test_get_context_includes_l2_when_requested(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    brain.indexer = MagicMock()
    brain.indexer.search.return_value = ["viking://src/main.py"]
    brain.indexer.get_display_texts.return_value = {
        "viking://src/main.py": "main.py display text"
    }

    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "main.py").write_text("def main(): pass")

    overviews_dir = tmp_path / ".ov_brain" / "overviews"
    overviews_dir.mkdir(parents=True)
    (overviews_dir / "src.l1.txt").write_text("L1 Overview")

    result = brain.get_context("query", include_l2=True)

    assert "def main(): pass" in result
    assert "L1 Overview" in result
    assert "main.py display text" in result


def test_get_context_excludes_l2_by_default(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    brain.indexer = MagicMock()
    brain.indexer.search.return_value = ["viking://src/main.py"]
    brain.indexer.get_display_texts.return_value = {
        "viking://src/main.py": "Display text only"
    }

    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "main.py").write_text("def main(): pass")

    overviews_dir = tmp_path / ".ov_brain" / "overviews"
    overviews_dir.mkdir(parents=True)
    (overviews_dir / "src.l1.txt").write_text("L1 only")

    result = brain.get_context("query")

    assert "def main(): pass" not in result


def test_get_context_warns_on_missing_l1(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    brain.indexer = MagicMock()
    brain.indexer.search.return_value = ["viking://src/main.py"]
    brain.indexer.get_display_texts.return_value = {
        "viking://src/main.py": "Display text"
    }

    with pytest.warns(UserWarning, match="L1 overview not found"):
        brain.get_context("query")


def test_hyde_non_question_passthrough(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    result = brain._hyde_query("verify_token JWT middleware")
    # Not a question — should return unchanged
    assert result == "verify_token JWT middleware"


def test_hyde_question_detected(tmp_path):
    brain = Brain(project_root=str(tmp_path))
    # Questions start with question words
    assert brain._is_question("how does auth work?") is True
    assert brain._is_question("where is verify_token") is True
    assert brain._is_question("verify_token JWT") is False
