"""Asks a reader with Python to find induction heads on another scrambled GPT-2 family model. `--live` makes sealed Astra calls, and `--scripted` makes none."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.run_reading_python import spent_so_far
from wsp.code_reader import run_code_reading, save_blocks
from wsp.family import MODELS, check_products, disguise_family, family_prompt, induction_scores, load

ROOT = Path(__file__).parent.parent
ARMS = {"medium-1": ("medium", 1201), "medium-2": ("medium", 1202), "distil-1": ("distil", 1203), "distil-2": ("distil", 1204), "distil-scripted": ("distil", 1205)}
MANIFEST = ROOT / "prompts" / "FAMILY_MANIFEST.json"
NAMED = 5
FREEZE = ("distil",)
TOP_TEN = 10
MAX_COST = 6.0
_SCRIPTED_CODE = 'print("TOP", " ".join(f"{layer}.{layer}" for layer in range(1, 6)))\n'


def freeze_prompts() -> None:
    folder = ROOT / "prompts" / "family"
    folder.mkdir(exist_ok=True)
    hashes = {}
    for short_name in FREEZE:
        text = family_prompt(load(short_name))
        (folder / f"{short_name}.txt").write_text(text)
        hashes[short_name] = hashlib.sha256(text.encode()).hexdigest()
    MANIFEST.write_text(json.dumps(hashes, indent=1))
    print(f"froze {len(hashes)} prompts")


def _scripted_reader(prompt: str, item: str) -> str:
    if item == "turn-1":
        return f"```python\n{_SCRIPTED_CODE}```"
    picks = prompt.rsplit("TOP ", 1)[1].splitlines()[0].split()
    return "```answer\n" + "\n".join(picks) + "\n```"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=[*ARMS, "freeze"])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="make sealed Astra calls")
    mode.add_argument("--scripted", action="store_true", help="use the scripted reader, with no model calls")
    arguments = parser.parse_args()
    if arguments.arm == "freeze":
        freeze_prompts()
        return
    if arguments.live == arguments.arm.endswith("scripted"):
        raise ValueError("the scripted arm runs only with --scripted, and the others only with --live")

    short_name, seed = ARMS[arguments.arm]
    model = load(short_name)
    prompt = family_prompt(model)
    if hashlib.sha256(prompt.encode()).hexdigest() != json.loads(MANIFEST.read_text())[short_name]:
        raise ValueError("the prompt no longer matches its frozen hash")
    scores = induction_scores(model)
    disguised = disguise_family(model, seed)
    worst = check_products(model, disguised)
    if worst > 1e-6:
        raise ValueError(f"the scramble changed QK^T or VO by {worst}")
    print(f"scramble keeps the products, worst relative change {worst:.2e}")
    heads = model.config.n_head
    truth = np.stack([scores[layer, disguised.original_head[layer]] for layer in range(model.config.n_layer)])
    order = np.argsort(-truth.ravel())
    top = {(int(i) // heads, int(i) % heads) for i in order[:NAMED]}
    top_ten = {(int(i) // heads, int(i) % heads) for i in order[:TOP_TEN]}
    weights = ROOT / "data" / "family" / f"blocks_{seed}.npz"
    if not weights.exists():
        save_blocks(disguised, weights)
    original = disguised.original_head.tolist()
    del model, disguised

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "live" if arguments.live else "scripted"
    run_dir = ROOT / "runs" / f"family-{arguments.arm}__{label}__{stamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "truth.json").write_text(json.dumps({"model": MODELS[short_name], "scores": scores.tolist(), "disguise_map": original}))
    if arguments.live:
        from wsp.astra import call_astra

        def call(text: str, item: str) -> str:
            return call_astra(text, run_dir, item)["reply"]

        spent = lambda: spent_so_far(run_dir)
    else:
        call, spent = _scripted_reader, lambda: 0.0
    result = run_code_reading(prompt, weights, top, top_ten, seed, call, spent, run_dir, MAX_COST)
    print(run_dir)
    print(json.dumps(result.__dict__, indent=1, default=list))
    print(f"list price ${spent():.2f}")
    weights.unlink()


if __name__ == "__main__":
    main()
