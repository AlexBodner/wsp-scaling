"""Tests Astra's GPT-Neo edits on plain English and against plain edits of the same heads, and measures its previous-token heads. Makes no Astra calls.

Takes the two run folders as arguments, in the order of run 1 and run 2, and defaults to the runs whose folder names held "neo". To check other runs, pass their folders and send the output to its own file.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from checks.other_models import cosines
from checks.edit_demo import FACTS, astra_edit, continuation, copying_cases, top_share
from scripts.run_write_neo import fresh
from scripts.run_write_medium import scores, zeroed
from wsp.disguise import HEAD_SIZE
from wsp.family import disguise_family, load

ROOT = Path(__file__).parent.parent
SEEDS = {"1": 1501, "2": 1502}
NAMED_FOLDER_RUNS = ("runs/write-neo-1__live__20261104T195206Z", "runs/write-neo-2__live__20261104T201804Z")
READER = (6, 1)
HEADS = 12
NAMED = 5
RANDOM_SETS = 8
LENGTH = 50
BATCH = 8
PROMPTS = ("Words: dog cat tree house car dog cat tree", "Words: apple chair river stone apple chair river", "apple, chair, river, apple, chair,", "The capital of France is", "Once upon a time")


class PlainTokens:
    """Stands in for the token shuffle, since family models keep their real token order."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode_token(self, token: int) -> str:
        return self.tokenizer.decode([token])


@torch.no_grad()
def previous_token_scores(model) -> np.ndarray:
    tokens = torch.randint(1000, 50000, (16, LENGTH), generator=torch.Generator().manual_seed(0))
    attentions = model(tokens, output_attentions=True).attentions
    positions = torch.arange(1, LENGTH)
    return np.stack([attention[:, :, positions, positions - 1].mean((0, 2)).numpy() for attention in attentions])


@torch.no_grad()
def target_attention(model, layer: int, head: int) -> float:
    """Returns a head's attention on the position after the earlier copy, in the second of two copies of random tokens."""
    half = torch.randint(1000, 50000, (32, LENGTH), generator=torch.Generator().manual_seed(0))
    second = torch.arange(LENGTH, 2 * LENGTH - 1)
    return float(model(torch.cat([half, half], 1), output_attentions=True).attentions[layer][:, head].mean(0)[second, second - LENGTH + 1].mean())


@torch.no_grad()
def mean_fixed(disguised, heads: list[tuple[int, int]], windows: torch.Tensor) -> dict[str, np.ndarray]:
    """Returns weights with these scrambled heads' outputs fixed at their mean on stories. It needs a forward pass, so Astra couldn't do this."""
    model, stored = fresh(disguised.tensors), {}
    hooks = [block.attn.c_proj.register_forward_pre_hook(lambda _m, inputs, layer=layer: stored.setdefault(layer, []).append(inputs[0].mean((0, 1)).numpy())) for layer, block in enumerate(model.transformer.h)]
    for part in windows.chunk(BATCH):
        model(part)
    for hook in hooks:
        hook.remove()
    tensors = dict(disguised.tensors)
    for layer, head in heads:
        rows = slice(head * HEAD_SIZE, (head + 1) * HEAD_SIZE)
        weight, bias = tensors[f"L{layer}.out.W"].copy(), tensors[f"L{layer}.out.b"].copy()
        bias += np.mean(stored[layer], 0)[rows] @ weight[rows]
        weight[rows] = 0
        tensors[f"L{layer}.out.W"], tensors[f"L{layer}.out.b"] = weight, bias
    return tensors


def report(label: str, tensors, base_loss: float, tokens: PlainTokens, cases, facts) -> None:
    model = fresh(tensors)
    repeat, loss = scores(model)
    print(f"\n[{label}]\nscored test: repeat accuracy {repeat:.3f}, rise {loss - base_loss:+.3f} ({100 * (loss - base_loss) / base_loss:.1f}%)")
    for case, items in cases.items():
        print(f"copying, {case}: {top_share(model, items):.3f}")
    print(f"facts: {sum(top_share(model, [fact]) == 1 for fact in facts)} of {len(facts)}; right: {[prompt for (prompt, _), fact in zip(FACTS, facts) if top_share(model, [fact]) == 1]}")
    for prompt in PROMPTS:
        print(f"  {prompt!r} -> {continuation(model, tokens, prompt)!r}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", default=NAMED_FOLDER_RUNS, help="the folders of run 1 and run 2")
    runs = parser.parse_args().runs
    if len(runs) != len(SEEDS):
        parser.error("give the folders of both runs, or neither")
    tokens = PlainTokens()
    cases = copying_cases(tokens)
    facts = [(tokens.encode(prompt), tokens.encode(f" {answer}")[0]) for prompt, answer in FACTS]
    previous = previous_token_scores(load("neo"))
    order = np.argsort(-previous.ravel())
    print("top previous-token heads: " + ", ".join(f"{i // HEADS}.{i % HEADS} {previous.ravel()[i]:.3f}" for i in order[:10]))
    layers = np.repeat(np.arange(12), HEADS)
    cosine = cosines(load("neo"))
    allowed = layers[:, None] > layers[None, :]
    reader_share = np.where(allowed, cosine, -np.inf)
    writers = np.argsort(-np.maximum(reader_share, 0).sum(0))[:NAMED]
    best_writer = np.bincount(np.argmax(reader_share[allowed.any(1)], axis=1), minlength=len(cosine))
    print("script, heads summing the most match with later heads: " + ", ".join(f"{i // HEADS}.{i % HEADS}" for i in writers) + f"; in the top 5 by previous token: {len(set(writers.tolist()) & set(order[:NAMED].tolist()))} of {NAMED}")
    print("script, heads most often the best match for a later head: " + ", ".join(f"{i // HEADS}.{i % HEADS} ({best_writer[i]})" for i in np.argsort(-best_writer)[:NAMED]) + f"; in the top 5 by previous token: {len(set(np.argsort(-best_writer)[:NAMED].tolist()) & set(order[:NAMED].tolist()))} of {NAMED}", flush=True)
    with open(ROOT / "data/tinystories/valid.txt", encoding="utf-8", errors="replace") as handle:
        handle.read(600_000)
        text = tokens.encode(handle.read(600_000))
    windows = torch.tensor([text[int(i) : int(i) + LENGTH] for i in np.random.default_rng(0).choice(len(text) - LENGTH, 64, replace=False)])
    for (arm, seed), folder in zip(SEEDS.items(), runs):
        run = ROOT / folder
        disguised = disguise_family(load("neo"), seed)
        base_loss = scores(fresh(disguised.tensors))[1]
        named = [tuple(head) for head in json.loads((run / "write_score.json").read_text())["named_heads"]]
        real = [f"{layer}.{int(disguised.original_head[layer][head])}" for layer, head in named]
        print(f"\n===== run {arm}: named heads, real numbers: {real}; ranks on previous token: {[int(np.where(order == layer * HEADS + int(disguised.original_head[layer][head]))[0][0]) + 1 for layer, head in named]}")
        if arm == "1":
            report("unedited", disguised.tensors, base_loss, tokens, cases, facts)
        edited = {**disguised.tensors, **astra_edit(disguised, run)}
        report(f"Astra, run {arm}", edited, base_loss, tokens, cases, facts)
        reader_slot = int(np.where(disguised.original_head[READER[0]] == READER[1])[0][0])
        print(f"reader {READER[0]}.{READER[1]}, attention on the position after the earlier copy: unedited {target_attention(fresh(disguised.tensors), READER[0], reader_slot):.3f}, edited {target_attention(fresh(edited), READER[0], reader_slot):.3f}", flush=True)
        real_named = [(layer, int(disguised.original_head[layer][head])) for layer, head in named]
        report(f"named heads' outputs zeroed, run {arm}", zeroed(disguised, real_named), base_loss, tokens, cases, facts)
        report(f"named heads fixed at their mean, needs a forward pass, run {arm}", mean_fixed(disguised, named, windows), base_loss, tokens, cases, facts)
        rng = np.random.default_rng(seed)
        random_sets = np.array([scores(fresh(zeroed(disguised, [(int(i) // HEADS, int(i) % HEADS) for i in rng.choice(HEADS * 12, NAMED, replace=False)]))) for _ in range(RANDOM_SETS)])
        print(f"\n{RANDOM_SETS} random sets of 5 zeroed: repeat accuracy {random_sets[:, 0].min():.3f} to {random_sets[:, 0].max():.3f}; rise up to {random_sets[:, 1].max() - base_loss:+.3f}", flush=True)


if __name__ == "__main__":
    main()
