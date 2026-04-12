import pytest
from unittest.mock import patch, AsyncMock
from leanviking.summarizer import Summarizer

@pytest.mark.asyncio
async def test_generate_l0():
    sum_gen = Summarizer()
    # Mock content
    content = "def add(a, b): return a + b"

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value.choices[0].message.content = "This file implements a basic addition function."

        summary = await sum_gen.generate_l0(content)

        assert summary == "This file implements a basic addition function."
        mock_completion.assert_called_once()

@pytest.mark.asyncio
async def test_generate_l1():
    sum_gen = Summarizer()
    child_l0s = [("main.py", "Entry point of the app"), ("utils.py", "Helper functions")]

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
        mock_completion.return_value.choices[0].message.content = "Overview of src: contains main.py and utils.py."

        overview = await sum_gen.generate_l1("src", child_l0s)

        assert overview == "Overview of src: contains main.py and utils.py."
        assert "main.py" in overview
        mock_completion.assert_called_once()
