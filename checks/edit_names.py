"""Tests repeated names on Astra's two GPT-2 edits, with rare names and with common ones, by top token and by the right surname's log probability. Makes no Astra calls."""
import numpy as np
import torch

from checks.edit_demo import FIRST, LAST, ROOT, RUNS, astra_edit
from wsp.disguise import disguise, model_from
from wsp.network import TokenShuffle
from wsp.write_task import apply_edit

COMMON_FIRST = "John Mary David Sarah Michael Anna Peter Laura James Emma".split()
COMMON_LAST = "Smith Johnson Williams Brown Jones Miller Davis Wilson Taylor Clark".split()
TEMPLATES = {"new sentence": "Yesterday {a} {b} went home, and {a}", "second sentence": "Yesterday {a} {b} went home early. {a}", "met": "I met {a} {b} at the store. Later {a}"}
BATCH = 48


def cases(shuffle: TokenShuffle, first: list[str], last: list[str]) -> dict[str, list[tuple[list[int], int]]]:
    return {name: [(shuffle.encode(template.format(a=a, b=b)), shuffle.encode(f" {b}")[0]) for a in first for b in last] for name, template in TEMPLATES.items()}


@torch.no_grad()
def scores(model, items: list[tuple[list[int], int]]) -> tuple[float, float]:
    """Returns how often the wanted token is on top, and its mean log probability."""
    groups: dict[int, list[tuple[list[int], int]]] = {}
    for prompt, answer in items:
        groups.setdefault(len(prompt), []).append((prompt, answer))
    right, log_probability = 0, 0.0
    for group in groups.values():
        for start in range(0, len(group), BATCH):
            part = group[start : start + BATCH]
            logits = model(torch.tensor([prompt for prompt, _ in part])).logits[:, -1]
            wanted = torch.tensor([answer for _, answer in part])
            right += int((logits.argmax(-1) == wanted).sum())
            log_probability += float(logits.log_softmax(-1).gather(1, wanted[:, None]).sum())
    return right / len(items), log_probability / len(items)


def main() -> None:
    shuffle = TokenShuffle()
    sets = {"the repo's rare names": cases(shuffle, FIRST, LAST), "common names": cases(shuffle, COMMON_FIRST, COMMON_LAST)}
    print("top token right (mean log probability of the right surname)")
    disguised = disguise(RUNS["run 2"][1], shuffle)
    models = {"unedited": lambda: model_from(disguised)}
    for run, (folder, seed) in RUNS.items():
        models[f"Astra, {run}"] = lambda folder=folder, seed=seed: (lambda edited_from: model_from(apply_edit(edited_from, astra_edit(edited_from, ROOT / "runs" / folder))))(disguise(seed, shuffle))
    for arm, build in models.items():
        model = build()
        for label, items in sets.items():
            found = {name: scores(model, prompts) for name, prompts in items.items()}
            print(f"{arm}, {label}, {len(next(iter(items.values())))} prompts each: " + " | ".join(f"{name} {right:.2f} ({log_probability:.2f})" for name, (right, log_probability) in found.items()), flush=True)


if __name__ == "__main__":
    main()
