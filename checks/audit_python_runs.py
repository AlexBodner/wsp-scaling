"""Rechecks the audit of the two Python runs from public logs. Makes no Astra calls.

Static checks read the saved code. It then replays that code on the same scrambled weights, matches each scrambled
head to real GPT-2 without using the scramble's record, and scores the answers. Codex's session files hold its own
instructions and stay private, so `--session-files` needs the machine that made the runs.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np

from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH, disguise
from wsp.network import TokenShuffle, real_network, writable_tensors
from wsp.reading import head_scores

ROOT = Path(__file__).parent.parent
RUNS = {
    1004: ROOT / "runs/reading-python-1__live__20261104T014623Z",
    1005: ROOT / "runs/reading-python-2__live__20261104T015236Z",
}
_UNAVAILABLE = ("E", "P", "F.scale", "F.shift")
_FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "system", "popen", "socket", "listdir", "glob", "walk", "getenv", "load", "save", "savez"}
_FORWARD_WORDS = re.compile(r"softmax|gelu|layer_?norm|torch\.nn|\bforward\w*", re.IGNORECASE)
NAMED = 5


def turns(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "transcript.jsonl").read_text().splitlines()]


def calls(run: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted((run / "calls").glob("[0-9]*.json"))]


def check_logs(run: Path, frozen_prompt_hash: str) -> list[str]:
    """Prints what the public logs show for one run and returns its thread ids."""
    records = calls(run)
    first_prompt = records[0]["request"]["user_text"]
    print(f"  prompt matches frozen hash: {hashlib.sha256(first_prompt.encode()).hexdigest() == frozen_prompt_hash}")
    kinds = sorted({str((event.get("item") or {}).get("type")) for record in records for event in record["response"]["events"]} - {"None"})
    print(f"  event item types over {len(records)} calls: {kinds}")
    print(f"  later prompts only extend earlier ones: {all(b['request']['user_text'].startswith(a['request']['user_text']) for a, b in zip(records, records[1:]))}")
    for turn in turns(run):
        if not turn["code"]:
            continue
        tree = ast.parse(turn["code"])
        imports = {(node.module if isinstance(node, ast.ImportFrom) else alias.name) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        names = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "") for node in ast.walk(tree) if isinstance(node, ast.Call)}
        print(f"  turn {turn['turn']}: imports {sorted(imports)}, forbidden calls {sorted(names & _FORBIDDEN_CALLS)}, forward words {sorted(set(_FORWARD_WORDS.findall(turn['code'])))}")
    return [record["seal"]["thread_id"] for record in records]


def products(tensors: dict, layer: int, head: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns one head's QK^T and VO, which the scramble leaves unchanged."""
    weight, output, start = tensors[f"L{layer}.qkv.W"], tensors[f"L{layer}.out.W"], head * HEAD_SIZE
    q, k, v = (weight[:, offset + start : offset + start + HEAD_SIZE] for offset in (0, WIDTH, 2 * WIDTH))
    return q @ k.T, v @ output[start : start + HEAD_SIZE]


def match_heads(scrambled: dict, real: dict) -> np.ndarray:
    """Returns layer x scrambled head -> real head, by the closest products."""
    matched = np.zeros((LAYERS, HEADS), dtype=int)
    for layer in range(LAYERS):
        reals = [products(real, layer, head) for head in range(HEADS)]
        for head in range(HEADS):
            a, b = products(scrambled, layer, head)
            matched[layer, head] = np.argmin([np.linalg.norm(a - x) / np.linalg.norm(x) + np.linalg.norm(b - y) / np.linalg.norm(y) for x, y in reals])
    return matched


def replay(run: Path, weights: dict) -> None:
    """Runs each logged code block on the weights and compares what it prints with what was logged."""
    for turn in turns(run):
        if not turn["code"]:
            continue
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile(turn["code"], str(run), "exec"), {"np": np, "W": dict(weights), "__name__": "__main__"})
        print(f"  turn {turn['turn']} replay matches its logged output: {printed.getvalue() == turn['output']}")


def raw_statistic_top(tensors: dict) -> list[tuple[int, int]]:
    """Ranks heads by cosine with the best earlier head on raw weights, with no scale or centering."""
    keys = [(layer, head) for layer in range(LAYERS) for head in range(HEADS)]
    pairs = [products(tensors, *key) for key in keys]
    a = np.stack([x.ravel() for x, _ in pairs])
    b = np.stack([y.ravel() for _, y in pairs])
    cosine = a @ b.T / np.outer(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
    rows = sorted(((cosine[i, j], keys[i]) for i in range(len(keys)) for j in range(len(keys)) if keys[j][0] < keys[i][0]), key=lambda row: -row[0])
    top = []
    for _, head in rows:
        if head not in top:
            top.append(head)
        if len(top) == NAMED:
            return top
    raise ValueError("fewer heads than requested")


def check_session_files(run: Path) -> None:
    """Compares billed input with the text visible in each private session file."""
    import tiktoken

    count = lambda text: len(tiktoken.get_encoding("o200k_base").encode(text, disallowed_special=()))
    for number, record in enumerate(calls(run), start=1):
        lines = [json.loads(line) for line in Path(record["seal"]["rollout"]).read_text().splitlines()]
        instructions = next(line for line in lines if line["type"] == "session_meta")["payload"]["base_instructions"]["text"]
        messages = [line["payload"] for line in lines if line["type"] == "response_item" and line["payload"].get("type") == "message" and line["payload"]["role"] in ("developer", "user")]
        visible = count(instructions) + sum(count(part.get("text", "")) for message in messages for part in message["content"])
        kinds = sorted({line["payload"].get("type") for line in lines if line["type"] == "response_item"})
        print(f"  call {number}: visible {visible}, Codex's instructions {count(instructions)}, billed {record['usage']['input_tokens']}, gap {record['usage']['input_tokens'] - visible}, items {kinds}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-files", action="store_true", help="also read Codex's private session files")
    arguments = parser.parse_args()

    frozen = json.loads((ROOT / "prompts" / "READING_MANIFEST.json").read_text())["python"]
    threads = []
    for seed, run in RUNS.items():
        print(f"run {run.name}")
        threads += check_logs(run, frozen)
    print(f"unique sessions: {len(set(threads))} of {len(threads)}")

    shuffle = TokenShuffle()
    model = real_network(shuffle).double()
    model.set_attn_implementation("eager")
    order = np.argsort(-head_scores(model).ravel())[:NAMED]
    truth = {(int(i) // HEADS, int(i) % HEADS) for i in order}
    print(f"top five heads of real GPT-2: {sorted(truth)}")
    real = {name: tensor.detach().numpy().astype(np.float64) for name, tensor in writable_tensors(real_network(shuffle)).items()}

    for seed, run in RUNS.items():
        print(f"seed {seed}")
        scrambled = disguise(seed, shuffle).tensors
        matched = match_heads(scrambled, real)
        answer = [tuple(pick) for pick in json.loads((run / "result.json").read_text())["answer"]]
        real_answer = [(layer, int(matched[layer, head])) for layer, head in answer]
        print(f"  answer {answer} is real {real_answer}: {sum(pick in truth for pick in real_answer)} of {NAMED}")
        raw_top = [(layer, int(matched[layer, head])) for layer, head in raw_statistic_top(scrambled)]
        print(f"  statistic on raw weights, no scale or centering, picks real {raw_top}: {sum(pick in truth for pick in raw_top)} of {NAMED}")
        replay(run, {name: value.astype(np.float32) for name, value in scrambled.items() if name not in _UNAVAILABLE})

    if arguments.session_files:
        for run in RUNS.values():
            print(f"session files of {run.name}")
            check_session_files(run)


if __name__ == "__main__":
    main()
