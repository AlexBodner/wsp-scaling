import json
import sys
from pathlib import Path

import numpy as np
import pytest

from wsp.code_reader import OUTPUT_CHARACTERS, last_python_block, run_code, run_code_reading
from wsp.reading import python_prompt

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="the sandbox uses macOS's sandbox-exec")


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    path = tmp_path_factory.mktemp("weights") / "blocks.npz"
    np.savez(path, **{"L0.qkv.W": np.ones((2, 3), dtype=np.float32)})
    return path


def test_code_sees_the_weights_and_its_output_comes_back(weights):
    assert run_code("print(W['L0.qkv.W'].shape)", weights).strip() == "(2, 3)"


@pytest.mark.parametrize(
    "secret",
    [
        Path.home() / ".cache" / "huggingface",
        Path(__file__).parent.parent / "runs",
        Path(__file__).parent.parent / "data",
        Path(__file__).parent.parent / "wsp" / "reading.py",
        Path.home() / ".ssh",
        Path.home() / ".codex",
        Path("/System/Volumes/Data") / str(Path.home()).lstrip("/") / ".cache" / "huggingface",
        Path("/private/tmp"),
    ],
)
def test_code_cannot_read_outside_its_folder(weights, secret):
    output = run_code(f"import os\ntry:\n    print('READ', os.listdir({str(secret)!r}) if os.path.isdir({str(secret)!r}) else open({str(secret)!r}).read(5))\nexcept Exception as error:\n    print('REFUSED', type(error).__name__)", weights)
    assert "REFUSED" in output and "READ" not in output


def test_code_cannot_write_outside_its_folder(weights):
    target = Path(__file__).parent / "sandbox_escape.txt"
    output = run_code(f"try:\n    open({str(target)!r}, 'w').write('x')\n    print('WROTE')\nexcept Exception as error:\n    print('REFUSED', type(error).__name__)", weights)
    assert "REFUSED" in output and not target.exists()


def test_code_has_no_network(weights):
    code = "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 443), timeout=5)\n    print('CONNECTED')\nexcept Exception as error:\n    print('REFUSED', type(error).__name__)"
    output = run_code(code, weights)
    assert "REFUSED" in output and "CONNECTED" not in output


def test_code_cannot_start_other_programs(weights):
    code = "import subprocess\ntry:\n    print('RAN', subprocess.run(['/bin/ls', '/'], capture_output=True).returncode)\nexcept Exception as error:\n    print('REFUSED', type(error).__name__)"
    output = run_code(code, weights)
    assert "REFUSED" in output and "RAN" not in output


def test_a_crash_reports_its_exit_code(weights):
    assert "exit code 3" in run_code("import sys\nsys.exit(3)", weights)


def test_code_cannot_import_the_project(weights):
    assert "ModuleNotFoundError" in run_code("import wsp", weights)


def test_a_long_run_is_stopped(weights):
    assert "STOPPED" in run_code("import time\ntime.sleep(30)", weights, seconds=2)


def test_long_output_is_cut(weights):
    output = run_code(f"print('x' * {OUTPUT_CHARACTERS * 2})", weights)
    assert "output cut" in output and len(output) < OUTPUT_CHARACTERS + 100


def test_scientific_libraries_load_in_the_sandbox(weights):
    code = "import scipy, sklearn, torch, pandas, sympy, networkx\nprint('LOADED', float(torch.tensor([1.0]).sum()))"
    assert "LOADED 1.0" in run_code(code, weights)


def test_no_path_the_code_can_see_names_the_project(weights):
    output = run_code("import os\nprint(os.getcwd(), sorted(os.listdir('.')))", weights)
    assert "wsp" not in output and "blocks.npz" in output


def test_last_python_block_takes_the_last_one():
    assert last_python_block("```python\na\n```\ntext\n```python\nb\n```") == "b\n"
    assert last_python_block("no code") is None


def _reader(replies):
    remaining = iter(replies)
    return lambda prompt, item: next(remaining)


def test_a_run_replays_outputs_and_scores_both_ways(weights, tmp_path):
    prompts = []

    def call(prompt, item):
        prompts.append(prompt)
        return "```python\nprint('SEEN')\n```" if item == "turn-1" else "```answer\n1.1\n2.2\n3.3\n4.4\n5.5\n```"

    result = run_code_reading("START", weights, {(1, 1)}, {(1, 1), (2, 2)}, 7, call, lambda: 0.0, tmp_path)
    assert "SEEN" in prompts[1] and result.turns == 2 and result.code_runs == 1
    assert result.hits_induction == 1 and result.hits_top_ten == 2
    assert json.loads((tmp_path / "result.json").read_text())["stopped_by"] == "answer"


def test_the_spend_cap_makes_the_next_turn_the_last(weights, tmp_path):
    result = run_code_reading("START", weights, set(), set(), 7, _reader(["```python\nprint(1)\n```", "still no answer"]), lambda: 99.0, tmp_path, max_cost=30.0)
    assert result.turns == 2 and result.answer is None and "spend cap" in result.stopped_by


def test_a_short_answer_gets_one_reminder_and_does_not_end_the_run(weights, tmp_path):
    full = "```answer\n1.1\n2.2\n3.3\n4.4\n5.5\n```"
    result = run_code_reading("START", weights, set(), set(), 7, _reader(["```answer\n3.7\n```", full]), lambda: 0.0, tmp_path)
    assert result.turns == 2 and len(result.answer) == 5


def test_a_stopped_run_resumes_from_its_transcript(weights, tmp_path):
    with pytest.raises(RuntimeError):
        def crash(prompt, item):
            if item == "turn-1":
                return "```python\nprint('FIRST')\n```"
            raise RuntimeError("stopped")

        run_code_reading("START", weights, set(), set(), 7, crash, lambda: 0.0, tmp_path)
    seen = []

    def finish(prompt, item):
        seen.append((item, "FIRST" in prompt))
        return "```answer\n1.1\n2.2\n3.3\n4.4\n5.5\n```"

    result = run_code_reading("START", weights, set(), set(), 7, finish, lambda: 0.0, tmp_path)
    assert seen == [("turn-2", True)] and result.turns == 2 and result.code_runs == 1


def test_a_forward_pass_word_is_flagged(weights, tmp_path):
    result = run_code_reading("START", weights, set(), set(), 7, _reader(["```python\n# softmax\nprint(1)\n```", "```answer\n1.1\n2.2\n3.3\n4.4\n5.5\n```"]), lambda: 0.0, tmp_path)
    assert result.forward_pass_flags == [1]


def test_python_prompt_gives_no_method_and_does_not_name_the_model():
    prompt = python_prompt()
    assert "Q[r]" not in prompt and "GPT" not in prompt and "HINT" not in prompt and "induction" not in prompt.lower()
    assert "NOT available" in prompt and "forward pass" in prompt.lower()
