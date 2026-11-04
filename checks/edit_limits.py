"""Tests where Astra's weight edits stop looking clean: other word lists, repeated stories, random tokens and long windows. Makes no Astra calls."""
import json
import re

import numpy as np
import torch

from checks.edit_demo import RUNS, ROOT, astra_edit, head_list_edit
from wsp.disguise import disguise, model_from
from wsp.network import TokenShuffle
from wsp.write_task import TEXT_FILE, apply_edit, repeat_tokens

SEED = 20261105
LISTS = 200
STORY_SPAN, STORY_WINDOWS = 50, 48
LONG_WINDOW, LONG_WINDOWS = 256, 100
BATCH = 8
POOLS = {
    "other nouns": "violin pillow ocean hammer ladder mirror candle basket ribbon feather lantern pencil spoon bucket carpet helmet island jungle kettle magnet".split(),
    "names and places": "John Mary Peter Anna Paul Sarah David Laura Mark Emma Paris Rome London Berlin Madrid Tokyo Cairo Oslo Vienna Texas Ohio India China Brazil Egypt".split(),
    "verbs and adjectives": "run jump sing write think carry build watch happy angry quiet heavy bright narrow gentle silent sharp frozen".split(),
}


def single_token_words(shuffle: TokenShuffle, words: list[str]) -> list[str]:
    return [word for word in words if len(shuffle.encode(f" {word}")) == 1]


def word_lists(shuffle: TokenShuffle, pool: list[str]) -> list[tuple[list[int], list[int]]]:
    """Returns comma lists a, b, c, a, b, with the tokens of a, b and c."""
    rng = np.random.default_rng(SEED)
    lists = []
    for _ in range(LISTS):
        words = [pool[i] for i in rng.choice(len(pool), 3, replace=False)]
        lists.append((shuffle.encode(", ".join(words + words[:2]) + ","), [shuffle.encode(f" {word}")[0] for word in words]))
    return lists


@torch.no_grad()
def list_scores(model, lists) -> tuple[float, float]:
    """Returns how often the top next token is the right word, and how often it is any listed word."""
    right = listed = 0
    for prompt, tokens in lists:
        top = int(model(input_ids=torch.tensor([prompt])).logits[0, -1].argmax())
        right += top == tokens[2]
        listed += top in tokens
    return right / len(lists), listed / len(lists)


@torch.no_grad()
def random_token_scores(model, repeats: torch.Tensor) -> tuple[float, float]:
    """Returns, over the second copy, how often the top token is right and how often it appears anywhere in the first copy."""
    half = repeats.shape[1] // 2
    right = seen = count = 0
    for part in repeats.chunk(BATCH):
        top = model(part).logits[:, half:-1].argmax(-1)
        right += int((top == part[:, half + 1 :]).sum())
        seen += sum(int(token) in set(row[:half].tolist()) for row, tops in zip(part, top) for token in tops)
        count += top.numel()
    return right / count, seen / count


@torch.no_grad()
def story_copying(model, windows: torch.Tensor) -> float:
    """Returns how often the top token is right in the second copy of a repeated story window."""
    doubled = torch.cat([windows, windows], 1)
    right = count = 0
    for part in doubled.chunk(BATCH):
        top = model(part).logits[:, STORY_SPAN:-1].argmax(-1)
        right += int((top == part[:, STORY_SPAN + 1 :]).sum())
        count += top.numel()
    return right / count


@torch.no_grad()
def text_loss(model, windows: torch.Tensor) -> float:
    total, count = 0.0, 0
    for part in windows.chunk(BATCH):
        logits = model(part).logits[:, :-1]
        total += float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), part[:, 1:].reshape(-1), reduction="sum"))
        count += part[:, 1:].numel()
    return total / count


def story_tokens(shuffle: TokenShuffle) -> list[int]:
    """Returns tokens from past the part of the stories the scorer reads, so these are fresh text."""
    with open(ROOT / TEXT_FILE, encoding="utf-8", errors="replace") as handle:
        handle.read(600_000)
        return shuffle.encode(handle.read(1_500_000))


def story_windows(shuffle: TokenShuffle) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = story_tokens(shuffle)
    rng = np.random.default_rng(SEED)
    pick = lambda length, number: torch.tensor([tokens[int(start) : int(start) + length] for start in rng.choice(len(tokens) - length, number, replace=False)])
    return pick(STORY_SPAN, STORY_WINDOWS), pick(LONG_WINDOW, LONG_WINDOWS)


def main() -> None:
    shuffle = TokenShuffle()
    lists = {name: word_lists(shuffle, single_token_words(shuffle, pool)) for name, pool in POOLS.items()}
    vocabulary = [word for word in (shuffle.tokenizer.decode([i]) for i in range(len(shuffle.tokenizer))) if re.fullmatch(r" [a-z]{4,}", word)]
    lists["random vocabulary words"] = word_lists(shuffle, [word[1:] for word in vocabulary])
    repeats = repeat_tokens(shuffle)
    stories, long_windows = story_windows(shuffle)
    base_loss = None
    for label, seed, folder, kind in (
        ("unedited", RUNS["run 2"][1], None, None),
        ("Astra, run 1", RUNS["run 1"][1], RUNS["run 1"][0], "astra"),
        ("Astra, run 2", RUNS["run 2"][1], RUNS["run 2"][0], "astra"),
        ("head list, run 2's heads", RUNS["run 2"][1], RUNS["run 2"][0], "list"),
    ):
        disguised = disguise(seed, shuffle)
        model = model_from(disguised)
        if kind:
            run_dir = ROOT / "runs" / folder
            targets = [tuple(head) for head in json.loads((run_dir / "write_score.json").read_text())["touched_heads"]]
            edit = astra_edit(disguised, run_dir) if kind == "astra" else head_list_edit(disguised, targets)
            model = model_from(apply_edit(disguised, edit))
        print(f"\n[{label}]")
        for name, items in lists.items():
            right, listed = list_scores(model, items)
            print(f"3-word comma lists, {name}: right word {right:.3f}, any listed word {listed:.3f}")
        right, seen = random_token_scores(model, repeats)
        print(f"random tokens repeated: right {right:.3f}, top token anywhere in the first copy {seen:.3f}")
        print(f"stories repeated, span {STORY_SPAN}: right {story_copying(model, stories):.3f}")
        loss = text_loss(model, long_windows)
        base_loss = loss if base_loss is None else base_loss
        print(f"text loss, {LONG_WINDOWS} windows of {LONG_WINDOW}: {loss:.3f}, rise {loss - base_loss:.3f} ({100 * (loss - base_loss) / base_loss:.1f}%)", flush=True)
        del model


if __name__ == "__main__":
    main()
