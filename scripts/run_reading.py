"""Runs one arm of the reading test: read, blind or hinted. `--live` makes sealed Astra calls.

Each turn is one fresh sealed call that replays the transcript, so private reasoning is not kept.
"""

import argparse
import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from wsp.disguise import disguise
from wsp.network import TokenShuffle, real_network
from wsp.reading import (
    BLIND_SEED,
    HEADS_TO_NAME,
    HINTED_SEED,
    PASS_HITS,
    PASS_OUT_OF,
    READ_BUDGET,
    READ_SEEDS,
    blind_prompt,
    disguised_scores,
    head_scores,
    hinted_prompt,
    positive_heads,
    read_prompt,
    run_reading,
)

ROOT = Path(__file__).parent.parent
ARMS = {
    "read-1": (read_prompt, READ_SEEDS[0], READ_BUDGET),
    "read-2": (read_prompt, READ_SEEDS[1], READ_BUDGET),
    "blind": (blind_prompt, BLIND_SEED, 0),
    "hinted": (hinted_prompt, HINTED_SEED, READ_BUDGET),
}


def _frozen(arm: str, text: str) -> None:
    manifest = json.loads((ROOT / "prompts" / "READING_MANIFEST.json").read_text())
    name = arm.split("-")[0]
    if hashlib.sha256(text.encode()).hexdigest() != manifest[name]:
        raise ValueError(f"the {name} prompt no longer matches its frozen hash")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--live", action="store_true", required=True, help="make sealed Astra calls")
    arguments = parser.parse_args()
    build, seed, budget = ARMS[arguments.arm]
    prompt = build()
    _frozen(arguments.arm, prompt)

    shuffle = TokenShuffle()
    real = real_network(shuffle).double()
    real.set_attn_implementation("eager")
    disguised = disguise(seed, shuffle)
    positives = positive_heads(disguised_scores(head_scores(real), disguised))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / f"reading-{arguments.arm}__live__{stamp}"
    run_dir.mkdir(parents=True)
    from wsp.astra import call_astra

    def call(text: str, item: str) -> str:
        reply = call_astra(text, run_dir, item)
        with (run_dir / "usage.jsonl").open("a") as usage:
            usage.write(json.dumps({"item": item, "input_tokens": reply["input_tokens"], "output_tokens": reply["output_tokens"], "stop_reason": reply["stop_reason"]}) + "\n")
        return reply["reply"]

    result = run_reading(prompt, disguised, positives, seed, budget, call, run_dir)
    print(run_dir)
    print(json.dumps({"hits": result.hits, "of": HEADS_TO_NAME, "turns": result.turns, "values_read": result.values_read, "answer": result.answer, "pass_needs": f"{PASS_HITS}/{PASS_OUT_OF} over both reads"}))


if __name__ == "__main__":
    main()
