"""Asks whether a weights-only edit can swap GPT-2's repeat for another specific behaviour, before any Astra run. Makes no Astra calls.

On lists "Context: a b c d a b", the model normally says c. Targets: repeat the current word b, or say the word before it, a. The edits rewire the top induction heads using other heads' weights, which any reader of the weights could do.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from checks.edit_demo import SEED, TOP_FIVE, WORDS
from checks.other_abilities import stories, text_loss

# The measured top 5 duplicate heads are 0.5, 3.0, 0.1, 1.11 and 0.10 (checks/outputs/head_types.txt). This list once held (0, 3) in place of (0, 10), and the swap card's locked baselines and the saved outputs in checks/outputs/ used that old list.
DUPLICATE_HEADS = [(0, 5), (3, 0), (0, 1), (0, 10), (1, 11)]
PREVIOUS_HEAD = (4, 11)
LISTS = 200
BATCH = 64
HEAD = 64
WIDTH = 768


def load():
    return GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()


def word_lists(tokenizer) -> list[tuple[list[int], int, int, int]]:
    rng = np.random.default_rng(SEED)
    items = []
    for _ in range(LISTS):
        a, b, c, d = [WORDS[i] for i in rng.choice(len(WORDS), 4, replace=False)]
        token = lambda word: tokenizer.encode(f" {word}")[0]
        items.append((tokenizer.encode(f"Context: {a} {b} {c} {d} {a} {b}"), token(c), token(b), token(a)))
    return items


@torch.no_grad()
def shares(model, items) -> str:
    tokens = torch.tensor([item[0] for item in items])
    top = torch.cat([model(tokens[i : i + BATCH]).logits[:, -1].argmax(-1) for i in range(0, len(tokens), BATCH)]).tolist()
    return " | ".join(f"{name} {np.mean([t == item[slot] for t, item in zip(top, items)]):.3f}" for name, slot in (("next word (today)", 1), ("current word", 2), ("word before", 3)))


def columns(model, layer: int, head: int) -> tuple[slice, slice, slice]:
    start = head * HEAD
    return tuple(slice(offset + start, offset + start + HEAD) for offset in (0, WIDTH, 2 * WIDTH))


@torch.no_grad()
def copy_search(model, targets, sources) -> None:
    """Gives each target head the query and key of a duplicate head, so it attends to the earlier copy of the current word."""
    for (layer, head), (source_layer, source_head) in zip(targets, sources):
        to_q, to_k, _ = columns(model, layer, head)
        from_q, from_k, _ = columns(model, source_layer, source_head)
        weight, bias = model.transformer.h[layer].attn.c_attn.weight, model.transformer.h[layer].attn.c_attn.bias
        source_weight, source_bias = model.transformer.h[source_layer].attn.c_attn.weight, model.transformer.h[source_layer].attn.c_attn.bias
        for to, frm in ((to_q, from_q), (to_k, from_k)):
            weight[:, to] = source_weight[:, frm]
            bias[to] = source_bias[frm]


@torch.no_grad()
def read_previous(model, targets) -> None:
    """Points each target head's value at what the previous-token head writes, so what it copies is the word before."""
    layer, head = PREVIOUS_HEAD
    start = head * HEAD
    written = model.transformer.h[layer].attn.c_proj.weight[start : start + HEAD].double()
    basis = torch.linalg.qr(written.T)[0].float()
    for target_layer, target_head in targets:
        _, _, to_v = columns(model, target_layer, target_head)
        weight = model.transformer.h[target_layer].attn.c_attn.weight
        scale = weight[:, to_v].norm() / basis.norm()
        weight[:, to_v] = basis * scale


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    items = word_lists(tokenizer)
    print(f"unedited: {shares(load(), items)}", flush=True)
    text, fitted = stories(tokenizer)
    base = text_loss(load(), text)
    model = load()
    copy_search(model, TOP_FIVE, DUPLICATE_HEADS)
    print(f"top 5 search like duplicate heads: {shares(model, items)}; text loss rise {text_loss(model, text) - base:+.3f} of {base:.3f}", flush=True)
    show(model, tokenizer)
    rng = np.random.default_rng(SEED)
    others = [(layer, head) for layer in range(12) for head in range(12) if (layer, head) not in TOP_FIVE + DUPLICATE_HEADS]
    for number in range(4):
        control = load()
        copy_search(control, [others[i] for i in rng.choice(len(others), 5, replace=False)], DUPLICATE_HEADS)
        print(f"random 5 heads search like duplicate heads, set {number}: {shares(control, items)}; text loss rise {text_loss(control, text) - base:+.3f}", flush=True)
    model = load()
    copy_search(model, TOP_FIVE, DUPLICATE_HEADS)
    read_previous(model, TOP_FIVE)
    print(f"and their values read what 4.11 writes: {shares(model, items)}; text loss rise {text_loss(model, text) - base:+.3f}", flush=True)


def show(model, tokenizer) -> None:
    for prompt in ("Context: door chair car door chair", "Context: door chair car desk door chair", "The capital of France is", "Once upon a time", "The sun rises in the", "Mary went to the store and bought"):
        ids = torch.tensor([tokenizer.encode(prompt)])
        print(" ", repr(prompt), "->", repr(tokenizer.decode(model.generate(ids, max_new_tokens=10, do_sample=False, pad_token_id=50256)[0][ids.shape[1] :])), flush=True)


if __name__ == "__main__":
    main()
