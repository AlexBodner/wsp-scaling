"""Runs one Predict experiment: a fresh call predicts a written network's top tokens on fixed texts."""

import argparse
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

from wsp.edits import apply_edits, last_edit_block
from wsp.network import TokenShuffle, writable_tensors, zero_network
from wsp.predict import predict_texts, score_predictions
from wsp.prompts import predict_prompt

ROOT = Path(__file__).parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("capitals", "reverse", "language"))
    parser.add_argument("network_reply", type=Path, help="the reply holding the network's edit block")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--live", action="store_true", help="make the sealed Astra call")
    source.add_argument("--reply-file", type=Path)
    parser.add_argument("--label", default="run")
    arguments = parser.parse_args()

    shuffle = TokenShuffle()
    model = zero_network()
    tensors = writable_tensors(model)
    network_reply = arguments.network_reply.read_text(encoding="utf-8")
    report = apply_edits(network_reply, tensors)
    block = last_edit_block(network_reply)
    if block is None:
        raise ValueError(f"{arguments.network_reply} has no edit block, so use the hand-built network")
    texts = predict_texts(arguments.experiment)
    set_rows = set(tensors["E"].abs().sum(dim=1).nonzero().view(-1).tolist())
    prompt = predict_prompt(f"```edits\n{block}```", texts, set_rows, shuffle)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / f"predict-{arguments.experiment}__{arguments.label}__{stamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    if arguments.live:
        from wsp.astra import call_astra

        call = call_astra(prompt, run_dir, f"predict-{arguments.experiment}")
        (run_dir / "call.json").write_text(json.dumps({k: v for k, v in call.items() if k != "reply"}, indent=2) + "\n")
        reply = call["reply"]
    else:
        reply = arguments.reply_file.read_text(encoding="utf-8")
    (run_dir / "reply.md").write_text(reply, encoding="utf-8")
    score = score_predictions(reply, model, texts, shuffle)
    result = {"network_edit_report": dataclasses.asdict(report), "score": {**score, "match": dataclasses.asdict(score["match"])}}
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(run_dir)
    print(json.dumps(result["score"], indent=2))


if __name__ == "__main__":
    main()
