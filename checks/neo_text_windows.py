"""Measures the text rise of Astra's two GPT-Neo edits on stories no scored test used, at two window lengths. Makes no Astra calls.

Reads the edits from named runs, those whose folder names held "neo" and the clean-folder reruns, so it doesn't change when later runs are added. Seed 777 is the draw a verifier used for its own 256 windows, kept here to show how much the draw moves the result.
"""
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from checks.edit_demo import astra_edit
from scripts.run_write_neo import fresh
from wsp.family import disguise_family, load
from wsp.write_task import TEXT_FILE

ROOT = Path(__file__).parent.parent
RUNS = {
    "run 1": ("write-neo-1__live__20261104T195206Z", 1501),
    "run 2": ("write-neo-2__live__20261104T201804Z", 1502),
    "clean-folder run 1": ("write-neo-1__live__20261105T002705Z", 1501),
    "clean-folder run 2": ("write-neo-2__live__20261105T002705Z", 1502),
}
SKIPPED_CHARACTERS = 600_000
HELD_OUT_CHARACTERS = 600_000
# Windows, tokens per window, seed of the draw, windows per batch.
DRAWS = ((256, 48, 0, 16), (64, 256, 0, 4), (256, 48, 777, 16))


def windows(tokens: list[int], count: int, length: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.tensor([tokens[int(i) : int(i) + length] for i in rng.choice(len(tokens) - length, count, replace=False)])


@torch.no_grad()
def text_loss(model, text: torch.Tensor, batch: int) -> float:
    total, count = 0.0, 0
    for part in text.split(batch):
        logits = model(part).logits[:, :-1]
        total += float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), part[:, 1:].reshape(-1), reduction="sum"))
        count += part[:, 1:].numel()
    return total / count


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    with open(ROOT / TEXT_FILE, encoding="utf-8", errors="replace") as handle:
        handle.read(SKIPPED_CHARACTERS)
        held_out = tokenizer.encode(handle.read(HELD_OUT_CHARACTERS))
    draws = [(count, length, seed, batch, windows(held_out, count, length, seed)) for count, length, seed, batch in DRAWS]
    print(f"GPT-Neo 125M, text loss on stories after the first {SKIPPED_CHARACTERS:,} characters of {TEXT_FILE}")
    for run, (folder, seed) in RUNS.items():
        disguised = disguise_family(load("neo"), seed)
        unedited = fresh(disguised.tensors)
        edited = fresh({**disguised.tensors, **astra_edit(disguised, ROOT / "runs" / folder)})
        print(f"\nAstra, {run}, {folder}")
        for count, length, draw, batch, text in draws:
            before, after = text_loss(unedited, text, batch), text_loss(edited, text, batch)
            print(f"{count} windows of {length} tokens, seed {draw}: unedited {before:.3f}, edited {after:.3f}, rise {after - before:+.3f} ({100 * (after - before) / before:.1f}%)", flush=True)


if __name__ == "__main__":
    main()
