"""Lets a reader analyze a disguised GPT-2's blocks with Python, in a sandbox, as in the post's Experiment 3.

Only the blocks are available: without the token and position tables, nothing can be run on text. Each
run is a fresh process that can read the weights and its own folder, and has no network.
"""

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import psutil

from wsp.disguise import DisguisedModel
from wsp.viewer import parse_answer

MAX_RUNS = 20
RUN_SECONDS = 600
OUTPUT_CHARACTERS = 20_000
MEMORY_LIMIT_BYTES = 8 * 1024**3
THREADS = "4"
HEADS_TO_NAME = 5
# Codex takes the prompt as one argument, and macOS caps arguments and environment at 1,048,576 bytes.
PROMPT_BYTES_FINAL_TURN = 600_000
PROMPT_BYTES_REFUSED = 900_000
MAX_COST_USD = 40.0
COST_WARNING_SHARE = 0.7
_UNAVAILABLE = ("E", "P", "F.scale", "F.shift")
_PYTHON_BLOCK = re.compile(r"```python[ \t]*\n(.*?)```", re.DOTALL)
# Words that suggest a forward pass. A hit is only a flag for the audit, and stops nothing.
_FORWARD_PASS_WORDS = re.compile(r"softmax|gelu|relu|layer_?norm|forward|attention\(|torch\.nn", re.IGNORECASE)
_LAST_TURN_NOTICE = "THIS IS YOUR LAST TURN: end this reply with an answer block, or you score zero."
_LOW_BUDGET_NOTICE = "THE BUDGET IS RUNNING OUT: about 3 turns remain, so finish soon."
_SANDBOX_ROOT = "/private/tmp"
_BOOTSTRAP = """import numpy as np
with np.load({weights!r}) as _archive:
    W = {{name: _archive[name] for name in _archive.files}}
del _archive
"""
# Python aborts at startup under a strict allow-list, so this denies what could leak or change anything.
_SANDBOX_PROFILE = """(version 1)
(allow default)
(deny network*)
(deny appleevent-send)
(deny process-exec)
(allow process-exec (literal "{python}"))
(deny file-write*)
(deny file-read-data
  (subpath "/Users")
  (subpath "/System/Volumes/Data/Users")
  (subpath "/System/Volumes/Data/private/tmp")
  (subpath "/Volumes")
  (subpath "/private/tmp")
  (subpath "/private/var/folders")
  (subpath "/private/var/root")
  (subpath "/Library/Keychains"))
(allow file-read-data
  (literal "{weights}")
  (subpath "{work}"))
(allow file-write*
  (literal "/dev/null")
  (subpath "{work}"))
"""


@dataclass
class CodeReadingResult:
    """What one run did and how it scored."""

    seed: int
    turns: int
    code_runs: int
    answer: list[tuple[int, int]] | None
    hits_top_ten: int
    hits_induction: int
    forward_pass_flags: list[int]
    stopped_by: str


def save_blocks(disguised: DisguisedModel, path: Path) -> None:
    """Saves the disguised blocks as 32-bit floats, leaving out the token, position and final tensors."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = {name: values.astype(np.float32) for name, values in disguised.tensors.items() if name not in _UNAVAILABLE}
    np.savez(path, **blocks)


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _memory_used(process: subprocess.Popen) -> int:
    try:
        parent = psutil.Process(process.pid)
        return sum(item.memory_info().rss for item in [parent, *parent.children(recursive=True)])
    except psutil.Error:
        return 0


def run_code(code: str, weights: Path, seconds: int = RUN_SECONDS, keep: tuple[str, Path] | None = None) -> str:
    """Runs one piece of code in a fresh sandboxed process and returns its output, cut to the limit.

    The process works in an anonymous folder that holds a hard link to the weights, so no path it can
    see names the project or the seed. `keep` names a file the code writes in its folder, and where to save it.
    """
    with tempfile.TemporaryDirectory(dir=_SANDBOX_ROOT, prefix="run_") as folder:
        work = Path(folder).resolve()
        linked = work / "blocks.npz"
        os.link(weights.resolve(), linked)
        profile = work / "sandbox.sb"
        profile.write_text(_SANDBOX_PROFILE.format(python=os.path.realpath(sys.executable), weights=linked, work=work))
        script = work / "run.py"
        script.write_text(_BOOTSTRAP.format(weights=str(linked)) + code)
        environment = {
            "HOME": str(work),
            "TMPDIR": str(work),
            "PATH": "/usr/bin:/bin",
            "OMP_NUM_THREADS": THREADS,
            "MKL_NUM_THREADS": THREADS,
            "OPENBLAS_NUM_THREADS": THREADS,
            "VECLIB_MAXIMUM_THREADS": THREADS,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        log_path = work / "output.txt"
        started = time.monotonic()
        ended = None
        with log_path.open("w") as log:
            process = subprocess.Popen(
                ["/usr/bin/sandbox-exec", "-f", str(profile), os.path.realpath(sys.executable), "-I", str(script)],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, cwd=work, env=environment, start_new_session=True,
            )
            while process.poll() is None:
                if time.monotonic() - started > seconds:
                    ended = f"STOPPED: this run passed the limit of {seconds} seconds."
                    break
                if _memory_used(process) > MEMORY_LIMIT_BYTES:
                    ended = f"STOPPED: this run passed the memory limit of {MEMORY_LIMIT_BYTES // 1024**3} GB."
                    break
                time.sleep(0.5)
            if ended:
                _kill_group(process)
                process.wait()
        output = log_path.read_text(errors="replace")
        if keep and (work / keep[0]).exists():
            keep[1].parent.mkdir(parents=True, exist_ok=True)
            os.replace(work / keep[0], keep[1])
    if len(output) > OUTPUT_CHARACTERS:
        output = output[:OUTPUT_CHARACTERS] + f"\n[output cut at {OUTPUT_CHARACTERS:,} characters]"
    if ended:
        output = f"{output}\n{ended}"
    elif process.returncode:
        output = f"{output}\n[exit code {process.returncode}]"
    return output or "[no output]"


def last_python_block(reply: str) -> str | None:
    """Returns the code in the reply's last python block, or None if there is none."""
    blocks = _PYTHON_BLOCK.findall(reply)
    return blocks[-1] if blocks else None


def _turn_text(turn: int, reply: str, output: str | None, code_runs: int) -> str:
    return f"\n\nYOUR REPLY IN TURN {turn}:\n{reply}\n\nOUTPUT:\n{output}\n\nRuns used: {code_runs} of {MAX_RUNS}."


def run_code_reading(
    prompt: str,
    weights: Path,
    induction_heads: set[tuple[int, int]],
    top_ten: set[tuple[int, int]],
    seed: int,
    call: Callable[[str, str], str],
    spent: Callable[[], float],
    run_dir: Path,
    max_cost: float = MAX_COST_USD,
    last_turn_notice: str = _LAST_TURN_NOTICE,
) -> CodeReadingResult:
    """Runs the reader until it answers or a limit is reached, and saves every turn.

    Args:
        prompt: The opening prompt.
        weights: The saved disguised blocks.
        induction_heads: The heads scoring above the gap, which only the scorer sees.
        top_ten: The ten highest-scoring heads, which only the scorer sees.
        seed: The disguise seed, for the record.
        call: Sends one prompt and returns the reply: a sealed Astra call, or a scripted reader.
        spent: Returns the run's cost so far at list prices.
        run_dir: Where the run is saved.
        max_cost: The spend at which the next turn becomes the last.
        last_turn_notice: What the reader is told when a turn is its last.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    transcript, answer, turns, code_runs, flags, stopped_by = "", None, 0, 0, [], "answer"
    final_turn, warned, short_answer_seen = False, False, False
    log_path = run_dir / "transcript.jsonl"
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            turns = entry["turn"]
            if entry["code"] is not None:
                code_runs += 1
                if _FORWARD_PASS_WORDS.search(entry["code"]):
                    flags.append(code_runs)
            transcript += _turn_text(turns, entry["reply"], entry["output"], code_runs)
    try:
        with log_path.open("a", encoding="utf-8") as log:
            while answer is None:
                turns += 1
                full_prompt = prompt + transcript
                prompt_bytes = len(full_prompt.encode())
                if prompt_bytes > PROMPT_BYTES_REFUSED:
                    raise ValueError(f"prompt is {prompt_bytes} bytes, over the {PROMPT_BYTES_REFUSED} limit")
                reply = call(full_prompt, f"turn-{turns}")
                answer = parse_answer(reply, HEADS_TO_NAME)
                output = None
                if answer is not None and len(answer) < HEADS_TO_NAME and not final_turn and not short_answer_seen:
                    # A quoted example or a short list should not end the run, so it gets one reminder.
                    output = f"Your answer block names {len(answer)} distinct heads, and it must name {HEADS_TO_NAME}. It was not accepted."
                    answer, short_answer_seen = None, True
                code = None if answer is not None or final_turn or output else last_python_block(reply)
                if code is not None:
                    code_runs += 1
                    if _FORWARD_PASS_WORDS.search(code):
                        flags.append(code_runs)
                    output = run_code(code, weights)
                elif answer is None and not final_turn and not output:
                    output = "No python or answer block found."
                log.write(json.dumps({"turn": turns, "prompt_bytes": prompt_bytes, "reply": reply, "code": code, "output": output}) + "\n")
                log.flush()
                if answer is not None:
                    stopped_by = "answer"
                    break
                if final_turn:
                    break
                transcript += _turn_text(turns, reply, output, code_runs)
                reason = None
                if code_runs >= MAX_RUNS:
                    reason = f"the limit of {MAX_RUNS} runs"
                elif len((prompt + transcript).encode()) > PROMPT_BYTES_FINAL_TURN:
                    reason = "the transcript limit"
                elif spent() >= max_cost:
                    reason = f"the ${max_cost:g} spend cap"
                if reason:
                    final_turn, stopped_by = True, reason
                    transcript += f"\n{last_turn_notice}"
                elif not warned and spent() >= COST_WARNING_SHARE * max_cost:
                    warned = True
                    transcript += f"\n{_LOW_BUDGET_NOTICE}"
    except Exception as error:
        # No result.json is written, so a crashed run cannot be mistaken for a scored one.
        (run_dir / "ABORTED.md").write_text(f"Stopped at turn {turns}: {type(error).__name__}: {error}\nNo score.\n")
        raise
    result = CodeReadingResult(
        seed=seed,
        turns=turns,
        code_runs=code_runs,
        answer=answer,
        hits_top_ten=sum(head in top_ten for head in (answer or [])),
        hits_induction=sum(head in induction_heads for head in (answer or [])),
        forward_pass_flags=flags,
        stopped_by=stopped_by,
    )
    (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result
