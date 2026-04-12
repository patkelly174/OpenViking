import pytest
from leanviking.summarizer import Summarizer

@pytest.mark.asyncio
async def test_generate_l0():
    sum_gen = Summarizer()
    # Mock content
    content = "def add(a, b): return a + b"
    summary = await sum_gen.generate_l0(content)
    assert len(summary) > 0
    assert isinstance(summary, str)

@pytest.mark.asyncio
async def test_generate_l1():
    sum_gen = Summarizer()
    child_l0s = [("main.py", "Entry point of the app"), ("utils.py", "Helper functions")]
    overview = await sum_gen.generate_l1("src", child_l0s)
    assert len(overview) > 0
    assert "main.py" in overview
