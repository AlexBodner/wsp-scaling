"""Tests the simple edit on the wrong heads and whole layers, the floor under repeated stories, where the top five attend after an edit, and lists away from position 0. Makes no Astra calls."""
import json

import numpy as np
import torch

from checks.edit_demo import ROOT, RUNS, SEED as LIST_SEED, TOP_FIVE, astra_edit, continuation, copying_cases, head_list_edit, top_share
from checks.edit_limits import BATCH, SEED as STORY_SEED, STORY_SPAN, STORY_WINDOWS, story_copying, story_tokens, story_windows
from checks.edit_order import LABELS, LISTS, picks, short_lists
from checks.write_baselines import NEXT_FIVE
from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH, disguise, model_from
from wsp.head_kinds import REPEAT_LENGTH
from wsp.network import TokenShuffle
from wsp.write_task import apply_edit, repeat_tokens, scores, text_tokens

SEED = 20261106
RANDOM_SETS = 8
MIDDLE_LAYERS = (5, 6, 7)
LIST_PREFIX = "Words: "
APPLE_LIST = "apple, chair, river, apple, chair,"


def scrambled_heads(disguised, real_heads: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [(layer, int(np.where(disguised.original_head[layer] == head)[0][0])) for layer, head in real_heads]


def zeroed_keys(disguised, real_heads: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    edited: dict[str, np.ndarray] = {}
    for layer, head in scrambled_heads(disguised, real_heads):
        name = f"L{layer}.qkv.W"
        values = edited.setdefault(name, disguised.tensors[name].copy())
        values[:, WIDTH + head * HEAD_SIZE : WIDTH + (head + 1) * HEAD_SIZE] = 0
    return edited


@torch.no_grad()
def mean_ablated(disguised, real_heads: list[tuple[int, int]], windows: torch.Tensor) -> dict[str, np.ndarray]:
    """Returns an edit fixing each head's output at its mean on stories the scorer doesn't read. It needs a forward pass, so it is a bar and not something Astra could do."""
    model = model_from(disguised)
    means: dict[int, list[np.ndarray]] = {}
    hooks = [
        block.attn.c_proj.register_forward_pre_hook(lambda _module, inputs, layer=layer: means.setdefault(layer, []).append(inputs[0].mean((0, 1)).numpy()))
        for layer, block in enumerate(model.transformer.h)
    ]
    for part in windows.chunk(BATCH):
        model(part)
    for hook in hooks:
        hook.remove()
    edited: dict[str, np.ndarray] = {}
    for layer, head in scrambled_heads(disguised, real_heads):
        rows = slice(head * HEAD_SIZE, (head + 1) * HEAD_SIZE)
        weight = edited.setdefault(f"L{layer}.out.W", disguised.tensors[f"L{layer}.out.W"].copy())
        bias = edited.setdefault(f"L{layer}.out.b", disguised.tensors[f"L{layer}.out.b"].copy())
        bias += np.mean(means[layer], 0)[rows] @ weight[rows]
        weight[rows] = 0
    return edited


def stories_after_real_text(shuffle: TokenShuffle, stories: torch.Tensor) -> torch.Tensor:
    """Returns the story windows after the text that really came before them, for a floor with no repeat and the same targets as the second copy."""
    tokens = story_tokens(shuffle)
    starts = np.random.default_rng(STORY_SEED).choice(len(tokens) - STORY_SPAN, STORY_WINDOWS, replace=False)
    if (starts < STORY_SPAN).any():
        raise ValueError("a story window has no text before it")
    windows = torch.tensor([tokens[int(start) - STORY_SPAN : int(start) + STORY_SPAN] for start in starts])
    if not (windows[:, STORY_SPAN:] == stories).all():
        raise ValueError("these windows don't match the check's story windows")
    return windows


@torch.no_grad()
def right_after_context(model, windows: torch.Tensor) -> float:
    """Returns how often the top token is right over the second half of each window."""
    right = count = 0
    for part in windows.chunk(BATCH):
        top = model(part).logits[:, STORY_SPAN:-1].argmax(-1)
        right += int((top == part[:, STORY_SPAN + 1 :]).sum())
        count += top.numel()
    return right / count


@torch.no_grad()
def top_five_attention(model, disguised, repeats: torch.Tensor) -> dict[str, dict[tuple[int, int], float]]:
    """Returns each top head's mean attention to the induction target and to position 0 over the second copy, and to position 0 over the first copy, where nothing has repeated yet."""
    second_half, first_half = torch.arange(REPEAT_LENGTH, 2 * REPEAT_LENGTH - 1), torch.arange(1, REPEAT_LENGTH)
    totals = {name: dict.fromkeys(TOP_FIVE, 0.0) for name in ("target", "position 0", "first copy, position 0")}
    chunks = repeats.chunk(BATCH)
    for part in chunks:
        attentions = model(part, output_attentions=True).attentions
        for real, (layer, head) in zip(TOP_FIVE, scrambled_heads(disguised, TOP_FIVE)):
            pattern = attentions[layer][:, head].mean(0)
            totals["target"][real] += float(pattern[second_half, second_half - REPEAT_LENGTH + 1].mean()) / len(chunks)
            totals["position 0"][real] += float(pattern[second_half, 0].mean()) / len(chunks)
            totals["first copy, position 0"][real] += float(pattern[first_half, 0].mean()) / len(chunks)
    return totals


def main() -> None:
    shuffle = TokenShuffle()
    repeats, text = repeat_tokens(shuffle), text_tokens(shuffle)
    comma_lists = copying_cases(shuffle)["short comma list"]
    stories = story_windows(shuffle)[0]
    real_text = stories_after_real_text(shuffle, stories)
    prefixed = short_lists(shuffle, LIST_SEED, LIST_PREFIX)
    disguised = disguise(RUNS["run 2"][1], shuffle)
    unedited = model_from(disguised)
    base_repeat, base_loss = scores(unedited, repeats, text)
    print(f"unedited: repeat {base_repeat:.3f}, text loss {base_loss:.3f}, short lists {top_share(unedited, comma_lists):.3f}", flush=True)
    del unedited

    def score(label: str, edit: dict[str, np.ndarray]) -> tuple[float, float]:
        model = model_from(apply_edit(disguised, edit))
        repeat, loss = scores(model, repeats, text)
        print(f"{label}: repeat {repeat:.3f}, text rise {loss - base_loss:+.3f}, short lists {top_share(model, comma_lists):.3f}", flush=True)
        return repeat, loss - base_loss

    simple = lambda real_heads: head_list_edit(disguised, scrambled_heads(disguised, real_heads))
    middle = [(layer, head) for layer in MIDDLE_LAYERS for head in range(HEADS)]
    print("\nsimple edit, needing only a list of heads")
    score("top 5", simple(TOP_FIVE))
    score("next 5", simple(NEXT_FIVE))
    score("all 36 heads of layers 5 to 7", simple(middle))
    score(f"all {LAYERS * HEADS} heads", simple([(layer, head) for layer in range(LAYERS) for head in range(HEADS)]))
    rng = np.random.default_rng(SEED)
    known = TOP_FIVE + NEXT_FIVE
    pools = {"anywhere, not the top ten": [(layer, head) for layer in range(LAYERS) for head in range(HEADS) if (layer, head) not in known], "in layers 5 to 7, not the top ten": [head for head in middle if head not in known]}
    for name, pool in pools.items():
        results = np.array([score(f"random 5 {name}, set {number}", simple([pool[i] for i in rng.choice(len(pool), 5, replace=False)])) for number in range(RANDOM_SETS)])
        print(f"random 5 {name}: repeat mean {results[:, 0].mean():.3f}, lowest {results[:, 0].min():.3f}; text rise mean {results[:, 1].mean():+.3f}, highest {results[:, 1].max():+.3f}")

    print("\nother bars on the same heads")
    score("top 5 keys zeroed", zeroed_keys(disguised, TOP_FIVE))
    score("top 5 mean-ablated, uses a forward pass", mean_ablated(disguised, TOP_FIVE, stories))
    score("all of layers 5 to 7 mean-ablated, uses a forward pass", mean_ablated(disguised, middle, stories))

    print(f"\nstories, span {STORY_SPAN}, same targets; where the top five attend on random-token repeats; lists behind {LIST_PREFIX!r}")
    arms = [("unedited", "run 2", None), ("Astra, run 1", "run 1", "astra"), ("Astra, run 2", "run 2", "astra"), ("simple edit, top 5", "run 2", "top"), ("simple edit, run 1's heads", "run 1", "list"), ("simple edit, run 2's heads", "run 2", "list")]
    for label, run, kind in arms:
        folder, seed = RUNS[run]
        scrambled = disguise(seed, shuffle)
        run_dir = ROOT / "runs" / folder
        edit = {}
        if kind == "astra":
            edit = astra_edit(scrambled, run_dir)
        elif kind == "top":
            edit = head_list_edit(scrambled, scrambled_heads(scrambled, TOP_FIVE))
        elif kind == "list":
            edit = head_list_edit(scrambled, [tuple(head) for head in json.loads((run_dir / "write_score.json").read_text())["touched_heads"]])
        model = model_from(apply_edit(scrambled, edit))
        floor, second = right_after_context(model, real_text), story_copying(model, stories)
        print(f"{label}: stories after real text {floor:.3f}, second copy {second:.3f}, gain {100 * (second - floor):.1f} points")
        attention = top_five_attention(model, scrambled, repeats)
        for name, values in attention.items():
            print(f"  {name}: mean {np.mean(list(values.values())):.3f}; " + ", ".join(f"{layer}.{head} {value:.3f}" for (layer, head), value in values.items()))
        counts = picks(model, prefixed)
        print("  lists: " + " | ".join(f"{name} {counts[name] / LISTS:.2f}" for name in LABELS))
        print(f"  {LIST_PREFIX + APPLE_LIST!r} -> {continuation(model, shuffle, LIST_PREFIX + APPLE_LIST)!r}", flush=True)
        del model


if __name__ == "__main__":
    main()
