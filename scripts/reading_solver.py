"""The free positive control: a scripted reader that does the arithmetic in code, through the same viewer.

It skips the first third of layers, spreads its budget over the remaining heads, and ranks them by
summed Q[r] . K[r] over random rows r. It goes through the same text path as Astra.
"""

import json
import re
from pathlib import Path

import numpy as np

from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH, disguise
from wsp.network import TokenShuffle, real_network
from wsp.reading import HEADS_TO_NAME, READ_BUDGET, disguised_scores, head_scores, positive_heads, read_prompt, run_reading

ROOT = Path(__file__).parent.parent
SOLVER_SEEDS = range(2000, 2030)
_SECTION = re.compile(r"L(\d+)\.qkv\.W\[(\d+), (\d+):\d+\] =\n([^\n]+)")


class ScriptedReader:
    """Asks for rows of Q and K, then ranks heads from the numbers it was sent back."""

    def __init__(self, seed: int, budget: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.budget = budget

    def __call__(self, prompt: str, item: str) -> str:
        if item == "turn-1":
            return self._requests()
        return self._answer(prompt)

    def _requests(self) -> str:
        heads = [(layer, head) for layer in range(LAYERS // 3, LAYERS) for head in range(HEADS)]
        rows_total = self.budget // (2 * HEAD_SIZE)
        lines = []
        for position, (layer, head) in enumerate(heads):
            count = rows_total // len(heads) + (position < rows_total % len(heads))
            for row in self.rng.choice(WIDTH, count, replace=False):
                query, key = head * HEAD_SIZE, WIDTH + head * HEAD_SIZE
                lines.append(f"L{layer}.qkv.W[{row}, {query}:{query + HEAD_SIZE}]")
                lines.append(f"L{layer}.qkv.W[{row}, {key}:{key + HEAD_SIZE}]")
        return "```read\n" + "\n".join(lines) + "\n```"

    def _answer(self, prompt: str) -> str:
        rows: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}
        for layer, row, start, numbers in _SECTION.findall(prompt):
            layer, row, start = int(layer), int(row), int(start)
            part = "key" if start >= WIDTH else "query"
            head = (start - WIDTH if part == "key" else start) // HEAD_SIZE
            rows.setdefault((layer, head, row), {})[part] = np.array(numbers.split(), dtype=float)
        scores: dict[tuple[int, int], list[float]] = {}
        for (layer, head, _), parts in rows.items():
            scores.setdefault((layer, head), []).append(float(parts["query"] @ parts["key"]))
        best = sorted(scores, key=lambda head: -np.mean(scores[head]))[:HEADS_TO_NAME]
        return "```answer\n" + "\n".join(f"{layer}.{head}" for layer, head in best) + "\n```"


def main() -> None:
    shuffle = TokenShuffle()
    run_dir = ROOT / "runs" / "reading-solver"
    real = real_network(shuffle).double()
    real.set_attn_implementation("eager")
    real_scores = head_scores(real)
    outcomes = []
    for seed in SOLVER_SEEDS:
        disguised = disguise(seed, shuffle)
        positives = positive_heads(disguised_scores(real_scores, disguised))
        result = run_reading(read_prompt(), disguised, positives, seed, READ_BUDGET, ScriptedReader(seed, READ_BUDGET), run_dir / f"seed{seed}")
        outcomes.append({"seed": seed, "hits": result.hits, "values_read": result.values_read, "turns": result.turns})
        print(outcomes[-1], flush=True)
    hits = np.array([outcome["hits"] for outcome in outcomes])
    summary = {"mean_precision_at_5": float(hits.mean() / HEADS_TO_NAME), "std": float(hits.std() / HEADS_TO_NAME), "runs": len(hits), "outcomes": outcomes}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary["mean_precision_at_5"], summary["std"])


if __name__ == "__main__":
    main()
