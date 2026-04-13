import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from leanviking.memory import Memory


@pytest.mark.asyncio
async def test_extract_truths_creates_memory_files(tmp_path):
    memory = Memory(project_root=str(tmp_path))

    mock_json = (
        '{"truths": ['
        '{"key": "auth-jwt", "content": "Use JWT for authentication"},'
        '{"key": "typed-python", "content": "Prefer typed Python"}'
        ']}'
    )

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value.choices[0].message.content = mock_json

        truths = await memory.extract_truths("We use JWT and prefer typed Python.")

    assert len(truths) == 2
    memories_dir = tmp_path / ".ov_brain" / "memories"
    assert (memories_dir / "auth-jwt.md").exists()
    assert (memories_dir / "typed-python.md").exists()


@pytest.mark.asyncio
async def test_extract_truths_overwrites_existing(tmp_path):
    memory = Memory(project_root=str(tmp_path))

    memories_dir = tmp_path / ".ov_brain" / "memories"
    memories_dir.mkdir(parents=True, exist_ok=True)
    (memories_dir / "auth-jwt.md").write_text("# auth-jwt\n\nOld content\n")

    mock_json = '{"truths": [{"key": "auth-jwt", "content": "Updated: JWT with refresh tokens"}]}'

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value.choices[0].message.content = mock_json

        await memory.extract_truths("we now use refresh tokens")

    content = (memories_dir / "auth-jwt.md").read_text()
    assert "Updated" in content
    assert "Old content" not in content


@pytest.mark.asyncio
async def test_extract_truths_sanitizes_key_for_filename(tmp_path):
    memory = Memory(project_root=str(tmp_path))

    mock_json = '{"truths": [{"key": "use spaces/slashes", "content": "some truth"}]}'

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value.choices[0].message.content = mock_json

        truths = await memory.extract_truths("anything")

    memories_dir = tmp_path / ".ov_brain" / "memories"
    files = list(memories_dir.glob("*.md"))
    assert len(files) == 1
    assert "/" not in files[0].name
    assert " " not in files[0].name


def test_list_truths_returns_all_stored(tmp_path):
    memory = Memory(project_root=str(tmp_path))
    memories_dir = tmp_path / ".ov_brain" / "memories"
    memories_dir.mkdir(parents=True, exist_ok=True)
    (memories_dir / "truth-a.md").write_text("# truth-a\n\nContent A\n")
    (memories_dir / "truth-b.md").write_text("# truth-b\n\nContent B\n")

    truths = memory.list_truths()

    keys = {t["key"] for t in truths}
    assert "truth-a" in keys
    assert "truth-b" in keys
    assert len(truths) == 2
