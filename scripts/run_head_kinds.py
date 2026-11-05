"""Asks a reader with Python to find other kinds of head on a scrambled GPT-2. `--live` makes sealed Astra calls, and `--scripted` makes none."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.run_reading_python import spent_so_far
from wsp.code_reader import run_code_reading, save_blocks
from wsp.disguise import HEADS, disguise
from wsp.head_kinds import NAMED, QUESTIONS, TOP_TEN, kind_prompt, kind_scores
from wsp.network import TokenShuffle, real_network
from wsp.reading import disguised_scores

ROOT = Path(__file__).parent.parent
SEEDS = {"duplicate": 1101, "previous": 1102, "self": 1103, "name_mover": 1104, "s_inhibition": 1105, "open": 1106}
MANIFEST = ROOT / "prompts" / "HEAD_KINDS_MANIFEST.json"
MAX_COST = 5.0
_SCRIPTED_CODE = 'print("TOP", " ".join(f"{layer}.{layer}" for layer in range(4, 9)))\n'


def freeze_prompts() -> None:
    """Writes each kind's prompt and its hash, so a run can check nothing moved."""
    folder = ROOT / "prompts" / "head_kinds"
    folder.mkdir(exist_ok=True)
    hashes = {}
    for kind in SEEDS:
        text = kind_prompt(kind)
        (folder / f"{kind}.txt").write_text(text)
        hashes[kind] = hashlib.sha256(text.encode()).hexdigest()
    MANIFEST.write_text(json.dumps(hashes, indent=1))
    print(f"froze {len(hashes)} prompts")


def _scripted_reader(prompt: str, item: str) -> str:
    if item == "turn-1":
        return f"```python\n{_SCRIPTED_CODE}```"
    picks = prompt.rsplit("TOP ", 1)[1].splitlines()[0].split()
    return "```answer\n" + "\n".join(picks) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=[*SEEDS, "freeze"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="make sealed Astra calls")
    mode.add_argument("--scripted", action="store_true", help="use the scripted reader, with no model calls")
    arguments = parser.parse_args()
    if arguments.kind == "freeze":
        freeze_prompts()
        return
    if not arguments.live and not arguments.scripted:
        raise ValueError("choose --live or --scripted")

    prompt = kind_prompt(arguments.kind)
    if hashlib.sha256(prompt.encode()).hexdigest() != json.loads(MANIFEST.read_text())[arguments.kind]:
        raise ValueError("the prompt no longer matches its frozen hash")

    seed = SEEDS[arguments.kind]
    shuffle = TokenShuffle()
    real = real_network(shuffle).double()
    real.set_attn_implementation("eager")
    disguised = disguise(seed, shuffle)
    scores = kind_scores(real, shuffle)
    if arguments.kind == "open":
        top, top_ten = set(), set()
    else:
        order = np.argsort(-disguised_scores(scores[arguments.kind], disguised).ravel())
        top = {(int(i) // HEADS, int(i) % HEADS) for i in order[:NAMED]}
        top_ten = {(int(i) // HEADS, int(i) % HEADS) for i in order[:TOP_TEN]}
    weights = ROOT / "data" / "head_kinds" / f"blocks_{seed}.npz"
    if not weights.exists():
        save_blocks(disguised, weights)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "live" if arguments.live else "scripted"
    run_dir = ROOT / "runs" / f"kind-{arguments.kind}__{label}__{stamp}"
    run_dir.mkdir(parents=True)
    if arguments.live:
        from wsp.astra import call_astra

        def call(text: str, item: str) -> str:
            return call_astra(text, run_dir, item)["reply"]

        spent = lambda: spent_so_far(run_dir)
    else:
        call, spent = _scripted_reader, lambda: 0.0
    result = run_code_reading(prompt, weights, top, top_ten, seed, call, spent, run_dir, MAX_COST)
    (run_dir / "disguise_map.json").write_text(json.dumps(disguised.original_head.tolist()))
    print(run_dir)
    print(json.dumps(result.__dict__, indent=2, default=list))
    print(f"list price so far ${spent():.2f}")
    if weights.exists() and arguments.live:
        weights.unlink()


if __name__ == "__main__":
    main()
