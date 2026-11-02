"""Runs one write or control experiment: gets a reply, applies it, scores it and saves everything.

The reply comes from a sealed Astra call with `--live`, or from a file with `--reply-file`, which
follows the same path after the call. Nothing else differs.
"""

import argparse
import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from wsp.control import CONTROL_DATA, score_breakout, zero_policy
from wsp.edits import apply_edits
from wsp.guards import compare_with_real
from wsp.network import TokenShuffle, real_network, writable_tensors, zero_network
from wsp.tasks import frequency_bits, language_stories, score_capitals, score_language, score_reverse, token_counts

ROOT = Path(__file__).parent.parent
EXPERIMENTS = ("control", "capitals", "reverse", "language")
CONTROL_EPISODES = 1000
CONTROL_SEED = 0


def _frozen_prompt(experiment: str) -> str:
    manifest = json.loads((ROOT / "prompts" / "MANIFEST.json").read_text())
    path = CONTROL_DATA / "message.txt" if experiment == "control" else ROOT / "prompts" / f"{experiment}.txt"
    text = path.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != manifest[experiment]["sha256"]:
        raise ValueError(f"the {experiment} prompt no longer matches its frozen hash")
    return text


def score_network(experiment: str, model, shuffle: TokenShuffle) -> dict:
    """Scores a GPT-2 network on one task."""
    if experiment == "capitals":
        return score_capitals(model, shuffle)
    if experiment == "reverse":
        return score_reverse(model, shuffle)
    stories = language_stories(shuffle)
    return score_language(model, stories, frequency_bits(stories, token_counts(shuffle)))


def _plain(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def run(experiment: str, reply: str, run_dir: Path, episodes: int = CONTROL_EPISODES) -> dict:
    """Applies a reply, scores the network and writes the record into the run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reply.md").write_text(reply, encoding="utf-8")
    if experiment == "control":
        tensors = zero_policy()
        report = apply_edits(reply, tensors, allow_arithmetic=True)
        result = {"edit_report": dataclasses.asdict(report), "score": score_breakout(tensors, episodes, CONTROL_SEED)}
    else:
        shuffle = TokenShuffle()
        model = zero_network()
        tensors = writable_tensors(model)
        report = apply_edits(reply, tensors)
        real_tensors = writable_tensors(real_network(shuffle))
        result = {
            "edit_report": dataclasses.asdict(report),
            "nonzero_weights": int(sum((tensor != 0).sum() for tensor in tensors.values())),
            "copy_check": compare_with_real(tensors, real_tensors),
            "score": _plain(score_network(experiment, model, shuffle)),
        }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--live", action="store_true", help="make the sealed Astra call")
    source.add_argument("--reply-file", type=Path)
    parser.add_argument("--label", default="run")
    parser.add_argument("--episodes", type=int, default=CONTROL_EPISODES)
    arguments = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / f"{arguments.experiment}__{arguments.label}__{stamp}"
    run_dir.mkdir(parents=True)
    prompt = _frozen_prompt(arguments.experiment)
    if arguments.live:
        from wsp.astra import call_astra

        call = call_astra(prompt, run_dir, arguments.experiment)
        (run_dir / "call.json").write_text(json.dumps({k: v for k, v in call.items() if k != "reply"}, indent=2) + "\n")
        reply = call["reply"]
    else:
        reply = arguments.reply_file.read_text(encoding="utf-8")
    result = run(arguments.experiment, reply, run_dir, arguments.episodes)
    print(run_dir)
    print(json.dumps(result["score"], indent=2))


if __name__ == "__main__":
    main()
