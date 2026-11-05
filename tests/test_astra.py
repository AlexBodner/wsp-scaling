from types import SimpleNamespace

import pytest

pytest.importorskip("vlm_jigsaw")

from vlm_jigsaw.models import TOOL_USE_DETECTED  # noqa: E402

from wsp import astra  # noqa: E402


@pytest.fixture(autouse=True)
def _calls_stay_in_the_test_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(astra, "CALLS_ROOT", tmp_path / "anonymous")


def _fake_client(stop_reason, seen):
    class FakeCodex:
        def __init__(self, model, effort, calls, timeout_s, max_attempts):
            seen.update(model=model, effort=effort, max_attempts=max_attempts, has_log=calls is not None)

        def complete(self, request):
            seen["prompt"] = request.parts[0].text
            return SimpleNamespace(
                text="reply",
                stop_reason=stop_reason,
                input_tokens=1,
                output_tokens=2,
                raw={"seal": {"problems": ["tool item"]}},
            )

    return FakeCodex


def test_a_call_uses_astra_at_the_agreed_effort_once_and_logs(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(astra, "CodexCLI", _fake_client("stop", seen))
    monkeypatch.setattr(astra, "sealed_codex_binary", dict)
    result = astra.call_astra("the prompt", tmp_path, "capitals")
    assert seen == {"model": "gpt-6-astra", "effort": "xhigh", "max_attempts": 1, "has_log": True, "prompt": "the prompt"}
    assert result["reply"] == "reply"


def test_a_broken_seal_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(astra, "CodexCLI", _fake_client(TOOL_USE_DETECTED, {}))
    monkeypatch.setattr(astra, "sealed_codex_binary", dict)
    with pytest.raises(RuntimeError, match="seal broken"):
        astra.call_astra("the prompt", tmp_path, "capitals")


def test_a_call_runs_where_its_path_says_nothing_about_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(astra, "CALLS_ROOT", tmp_path / "anonymous")
    run_dir = tmp_path / "write-neo-1"
    run_dir.mkdir()
    calls = astra._anonymous_calls(run_dir)
    assert "write-neo" not in str(calls)
    assert (run_dir / "calls").is_symlink()
    (calls / "000001.json").write_text("{}")
    astra.bring_calls_home(run_dir)
    assert not (run_dir / "calls").is_symlink()
    assert (run_dir / "calls" / "000001.json").exists()
