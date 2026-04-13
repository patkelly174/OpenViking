import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from leanviking.summarizer import Summarizer, L0Summary


@pytest.mark.asyncio
async def test_generate_l0_returns_l0summary():
    s = Summarizer()
    content = "def add(a, b): return a + b"

    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        '{"search_text": "addition arithmetic function", '
        '"display_text": "Implements add() that returns the sum of two numbers."}'
    )

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await s.generate_l0(content, "math.py")

    assert isinstance(result, L0Summary)
    assert "addition" in result.search_text
    assert "add()" in result.display_text


@pytest.mark.asyncio
async def test_generate_l0_empty_content():
    s = Summarizer()
    result = await s.generate_l0("", "empty.py")
    assert isinstance(result, L0Summary)
    assert result.search_text != ""
    assert result.display_text != ""


@pytest.mark.asyncio
async def test_generate_l0_strips_json_fences():
    s = Summarizer()
    content = "x = 1"

    mock_response = MagicMock()
    mock_response.choices[0].message.content = (
        '```json\n{"search_text": "variable assignment", "display_text": "Assigns x to 1."}\n```'
    )

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await s.generate_l0(content, "x.py")

    assert isinstance(result, L0Summary)
    assert result.search_text == "variable assignment"


@pytest.mark.asyncio
async def test_generate_l0_bad_json_fallback():
    s = Summarizer()
    content = "x = 1"

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Not JSON at all"

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await s.generate_l0(content, "x.py")

    assert isinstance(result, L0Summary)
    assert result.display_text == "Not JSON at all"


@pytest.mark.asyncio
async def test_generate_l1():
    s = Summarizer()
    child_l0s = [("main.py", "Entry point of the app"), ("utils.py", "Helper functions")]

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Overview of src: entry via main.py, helpers in utils.py."

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        overview = await s.generate_l1("src", child_l0s)

    assert "main.py" in overview
    assert overview == "Overview of src: entry via main.py, helpers in utils.py."


@pytest.mark.asyncio
async def test_hypothetical_answer():
    s = Summarizer()

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "verify_token function JWT validation auth middleware"

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await s.hypothetical_answer("how is JWT validated?")

    assert "JWT" in result or "verify_token" in result
