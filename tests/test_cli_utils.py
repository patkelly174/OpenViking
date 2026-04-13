import json
import pytest
import typer
from src.cli_utils import OutputManager


def test_finalize_outputs_json_with_status_ok(capsys):
    om = OutputManager()
    om.finalize()
    out = json.loads(capsys.readouterr().out)
    assert out == {"status": "ok"}


def test_finalize_includes_added_data(capsys):
    om = OutputManager()
    om.add("files_processed", 5)
    om.add("brain_dir", ".ov_brain")
    om.finalize()
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "ok"
    assert out["files_processed"] == 5
    assert out["brain_dir"] == ".ov_brain"


def test_finalize_resets_data_after_call(capsys):
    om = OutputManager()
    om.add("key", "value")
    om.finalize()
    capsys.readouterr()  # clear buffer
    om.finalize()
    out = json.loads(capsys.readouterr().out)
    assert "key" not in out


def test_error_outputs_json_to_stdout(capsys):
    om = OutputManager()
    with pytest.raises(typer.Exit):
        om.error("something went wrong")
    out = json.loads(capsys.readouterr().out)
    assert out == {"status": "error", "message": "something went wrong"}


def test_error_exits_with_given_code():
    om = OutputManager()
    with pytest.raises(typer.Exit) as exc_info:
        om.error("fail", code=2)
    assert exc_info.value.exit_code == 2


def test_error_exits_with_default_code_1(capsys):
    om = OutputManager()
    with pytest.raises(typer.Exit) as exc_info:
        om.error("something failed")
    assert exc_info.value.exit_code == 1


def test_error_discards_accumulated_data(capsys):
    om = OutputManager()
    om.add("key", "val")
    with pytest.raises(typer.Exit):
        om.error("error message")
    out = json.loads(capsys.readouterr().out)
    assert out == {"status": "error", "message": "error message"}
    assert "key" not in out
