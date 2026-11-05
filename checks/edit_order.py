"""Asks what Astra's edits take away on short lists: which word does the edited model pick instead of the right one? Makes no Astra calls."""
import json
from collections import Counter

import numpy as np
import torch

from checks.edit_demo import RUNS, ROOT, SEED, TOP_FIVE, WORDS, astra_edit, head_list_edit, zeroed_outputs
from wsp.disguise import disguise, model_from
from wsp.network import TokenShuffle
from wsp.write_task import apply_edit

LISTS = 300
LABELS = ("right word", "first word again", "second word again", "not from the list")


def short_lists(shuffle: TokenShuffle, seed: int, prefix: str = "") -> list[tuple[list[int], list[int]]]:
    """Returns comma lists a, b, c, a, b, after an optional prefix, with the single tokens of a, b and c."""
    rng = np.random.default_rng(seed)
    lists = []
    for _ in range(LISTS):
        first, second, third = [WORDS[i] for i in rng.choice(len(WORDS), 3, replace=False)]
        tokens = [shuffle.encode(f" {word}")[0] for word in (first, second, third)]
        lists.append((shuffle.encode(f"{prefix}{first}, {second}, {third}, {first}, {second},"), tokens))
    return lists


@torch.no_grad()
def picks(model, lists) -> Counter:
    """Counts what the model's top next token is, for lists a, b, c, a, b, where the right word is c."""
    counts: Counter = Counter()
    for prompt, (first, second, third) in lists:
        top = int(model(input_ids=torch.tensor([prompt])).logits[0, -1].argmax())
        counts[{third: LABELS[0], first: LABELS[1], second: LABELS[2]}.get(top, LABELS[3])] += 1
    return counts


def show(label: str, model, lists) -> None:
    counts = picks(model, lists)
    print(f"{label:28s} " + " | ".join(f"{name} {counts[name] / LISTS:.2f}" for name in LABELS))


def main() -> None:
    shuffle = TokenShuffle()
    for seed_offset in (0, 1):
        lists = short_lists(shuffle, SEED + seed_offset)
        print(f"\nlists from seed {SEED + seed_offset}")
        disguised = disguise(RUNS["run 2"][1], shuffle)
        show("unedited", model_from(disguised), lists)
        show("top 5 zeroed", model_from(apply_edit(disguised, zeroed_outputs(disguised, TOP_FIVE))), lists)
        for run, (folder, seed) in RUNS.items():
            disguised = disguise(seed, shuffle)
            run_dir = ROOT / "runs" / folder
            targets = [tuple(head) for head in json.loads((run_dir / "write_score.json").read_text())["touched_heads"]]
            show(f"Astra, {run}", model_from(apply_edit(disguised, astra_edit(disguised, run_dir))), lists)
            show(f"head list, {run}'s heads", model_from(apply_edit(disguised, head_list_edit(disguised, targets))), lists)


if __name__ == "__main__":
    main()
