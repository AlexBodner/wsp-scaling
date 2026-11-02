"""Scores trained GPT-2 on every task, renumbered to the shuffled IDs, for scale."""

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from wsp.guards import compare_with_real
from wsp.network import TokenShuffle, real_network, writable_tensors
from wsp.tasks import frequency_bits, language_stories, score_capitals, score_language, score_reverse, token_counts

ROOT = Path(__file__).parent.parent


def _plain(value):
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def main() -> None:
    shuffle = TokenShuffle()
    model = real_network(shuffle)
    stories = language_stories(shuffle)
    counts = token_counts(shuffle)
    tensors = writable_tensors(model)
    result = {
        "capitals": _plain(score_capitals(model, shuffle)),
        "reverse": _plain(score_reverse(model, shuffle)),
        "language": _plain(score_language(model, stories, frequency_bits(stories, counts))),
        "uniform_bits_per_token": float(np.log2(len(shuffle.tokenizer))),
        "tokens_in_training_subset": int((counts > 0).sum()),
        "copy_check_on_itself": compare_with_real({"E": tensors["E"].detach()}, {"E": tensors["E"].detach()}),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / f"references__trained-gpt2__{stamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(run_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
