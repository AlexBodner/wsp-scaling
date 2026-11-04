"""Runs the reading test with Python and no hint. `--live` makes sealed Astra calls, and `--scripted` makes none.

The scripted reader computes a known statistic through the whole path, to test the harness for free.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wsp.code_reader import MAX_COST_USD, run_code_reading, save_blocks
from wsp.disguise import HEADS, disguise
from wsp.network import TokenShuffle, real_network
from wsp.reading import disguised_scores, head_scores, positive_heads, python_prompt

ROOT = Path(__file__).parent.parent
ARMS = {"python-1": 1004, "python-2": 1005, "scripted": 1006}
INDUCTION_HEADS = 5
# List prices for gpt-6-astra, from CONTEXT.md.
INPUT_PER_MILLION, CACHED_PER_MILLION, OUTPUT_PER_MILLION = 10, 1, 50
_SCRIPTED_CODE = """scores = []
for layer in range(12):
    weight = W[f"L{layer}.qkv.W"]
    for head in range(12):
        q, k = weight[:, 64 * head : 64 * (head + 1)], weight[:, 768 + 64 * head : 768 + 64 * (head + 1)]
        scores.append((float((q * k).sum()), layer, head))
print("TOP", " ".join(f"{layer}.{head}" for _, layer, head in sorted(scores, reverse=True)[:5]))
"""


def spent_so_far(run_dir: Path) -> float:
    """Returns the run's cost so far at list prices, from its saved calls."""
    total = 0.0
    for path in (run_dir / "calls").glob("[0-9]*.json"):
        usage = json.loads(path.read_text())["usage"]
        fresh = usage["input_tokens"] - usage["cached_input_tokens"]
        total += fresh * INPUT_PER_MILLION + usage["cached_input_tokens"] * CACHED_PER_MILLION + usage["output_tokens"] * OUTPUT_PER_MILLION
    return total / 1e6


def _scripted_reader(prompt: str, item: str) -> str:
    if item == "turn-1":
        return f"```python\n{_SCRIPTED_CODE}```"
    picks = prompt.rsplit("TOP ", 1)[1].splitlines()[0].split()
    return "```answer\n" + "\n".join(picks) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="make sealed Astra calls")
    mode.add_argument("--scripted", action="store_true", help="use the scripted reader, with no model calls")
    parser.add_argument("--resume", type=Path, help="continue a stopped run from its folder")
    arguments = parser.parse_args()
    if arguments.live == (arguments.arm == "scripted"):
        raise ValueError("the scripted arm runs only with --scripted, and the python arms only with --live")

    prompt = python_prompt()
    manifest = json.loads((ROOT / "prompts" / "READING_MANIFEST.json").read_text())
    if hashlib.sha256(prompt.encode()).hexdigest() != manifest["python"]:
        raise ValueError("the python prompt no longer matches its frozen hash")

    seed = ARMS[arguments.arm]
    shuffle = TokenShuffle()
    real = real_network(shuffle).double()
    real.set_attn_implementation("eager")
    disguised = disguise(seed, shuffle)
    scores = disguised_scores(head_scores(real), disguised)
    order = np.argsort(-scores.ravel())[:INDUCTION_HEADS]
    induction_heads = {(int(i) // HEADS, int(i) % HEADS) for i in order}
    weights = ROOT / "data" / "reading_python" / f"blocks_{seed}.npz"
    if not weights.exists():
        save_blocks(disguised, weights)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "live" if arguments.live else "scripted"
    run_dir = arguments.resume or ROOT / "runs" / f"reading-{arguments.arm}__{label}__{stamp}"
    run_dir.mkdir(parents=True, exist_ok=bool(arguments.resume))

    if arguments.live:
        from wsp.astra import call_astra

        def call(text: str, item: str) -> str:
            return call_astra(text, run_dir, item)["reply"]

        spent = lambda: spent_so_far(run_dir)
    else:
        call, spent = _scripted_reader, lambda: 0.0

    result = run_code_reading(prompt, weights, induction_heads, positive_heads(scores), seed, call, spent, run_dir, MAX_COST_USD)
    print(run_dir)
    print(json.dumps(result.__dict__, indent=2, default=list))


if __name__ == "__main__":
    main()
