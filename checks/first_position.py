"""Tests whether attention to position 0 is a no-match state of the induction heads, or a base rate. Makes no Astra calls.

Measures all 144 heads on random tokens, on real text, with the first token
swapped, and on repeated against fresh second halves. Then asks what position 0's attention is inherited from, and what
the five top induction heads write to the logits from what they read there.
"""
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from checks.head_types import random_token_patterns
from checks.signed_ladder import TOP_INDUCTION, direct_effect, name, run_with_heads
from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH
from wsp.network import TokenShuffle, real_network

SEED = 20261104
STORIES = "data/tinystories/valid.txt"
LENGTH = 50
HALF = 25
SAMPLES = 256
CHUNK = 32
THRESHOLD = 0.5
SHARP_FALL = 0.25
MID_LAYERS = range(4, 10)
LOOKALIKES = [(7, 9), (5, 7), (7, 8)]
NAMED = TOP_INDUCTION + LOOKALIKES
SHOWN = 12
# The second half skips its first position, whose token is position 0's own, so a duplicate match lands on position 0.
SECOND_HALF = range(HALF + 1, LENGTH)
ALL_HEADS = HEADS * LAYERS


def index(head: tuple[int, int]) -> int:
    return head[0] * HEADS + head[1]


def label(position: int) -> str:
    return name((position // HEADS, position % HEADS))


def is_mid(position: int) -> bool:
    return position // HEADS in MID_LAYERS


def rank(values: np.ndarray, head: tuple[int, int]) -> int:
    return int((values > values[index(head)]).sum()) + 1


def named_line(values: np.ndarray, heads: list[tuple[int, int]] = NAMED) -> str:
    return ", ".join(f"{name(head)} {values[index(head)]:.3f} (#{rank(values, head)})" for head in heads)


def top_line(values: np.ndarray) -> str:
    return ", ".join(f"{label(int(i))} {values[i]:.3f}" for i in np.argsort(-values)[:SHOWN])


def counts_line(values: np.ndarray) -> str:
    counts = (values.reshape(LAYERS, HEADS) > THRESHOLD).sum(1)
    return " ".join(f"{count:2d}" for count in counts) + f" | all {counts.sum()}, layers 4-9 {counts[list(MID_LAYERS)].sum()}"


def fresh_sequences(vocabulary: int, count: int, generator: torch.Generator) -> torch.Tensor:
    """Returns random token sequences one longer than LENGTH, with no token repeated inside a sequence."""
    return torch.stack([torch.randperm(vocabulary, generator=generator)[: LENGTH + 1] for _ in range(count)])


def story_tokens(shuffle: TokenShuffle, rng: np.random.Generator) -> torch.Tensor:
    """Returns the openings of stories, one token longer than LENGTH, skipping the file's cut-off first story."""
    stories = [story.strip() for story in Path(STORIES).read_text().split("<|endoftext|>")][1:]
    kept = []
    for choice in rng.permutation(len(stories)):
        tokens = shuffle.encode(stories[choice][:1000])
        if len(tokens) > LENGTH:
            kept.append(tokens[: LENGTH + 1])
        if len(kept) == SAMPLES:
            break
    return torch.tensor(kept)


@torch.no_grad()
def attention_to(model, tokens: torch.Tensor, targets: torch.Tensor | None = None, position_ids: torch.Tensor | None = None) -> np.ndarray:
    """Returns attention from each position to its target position, position 0 by default, as sequence by head by position."""
    targets = torch.zeros(tokens.shape[1], dtype=torch.long) if targets is None else targets
    rows = torch.arange(tokens.shape[1])
    collected = []
    for start in range(0, len(tokens), CHUNK):
        extra = {} if position_ids is None else {"position_ids": position_ids[start : start + CHUNK]}
        output = model(tokens[start : start + CHUNK], output_attentions=True, **extra)
        by_layer = np.stack([attention[:, :, rows, targets].numpy() for attention in output.attentions], 1)
        collected.append(by_layer.reshape(len(by_layer), ALL_HEADS, tokens.shape[1]))
    return np.concatenate(collected)


def captured_blocks(model) -> tuple[dict, list]:
    """Stores each layer's query, key and value output, and returns the store and the hooks to remove afterwards."""
    store, hooks = {}, []
    for layer in range(LAYERS):
        def keep(_module, _inputs, output, layer=layer):
            store[layer] = output.detach()

        hooks.append(model.transformer.h[layer].attn.c_attn.register_forward_hook(keep))
    return store, hooks


def split_head(block: torch.Tensor, part: int) -> torch.Tensor:
    """Returns query (0), key (1) or value (2) as batch by head by position by head size."""
    batch, length, _ = block.shape
    chosen = block[..., part * WIDTH : (part + 1) * WIDTH]
    return chosen.view(batch, length, HEADS, HEAD_SIZE).transpose(1, 2)


def attention_from_scores(scores: torch.Tensor) -> torch.Tensor:
    length = scores.shape[-1]
    allowed = torch.tril(torch.ones(length, length, dtype=torch.bool))
    return scores.masked_fill(~allowed, -torch.inf).softmax(-1)


def mean_pair_cosine(vectors: torch.Tensor) -> torch.Tensor:
    """Returns the mean cosine over pairs of sequences, for vectors of batch by head by head size."""
    unit = torch.nn.functional.normalize(vectors, dim=-1)
    batch = unit.shape[0]
    return (unit.sum(0).pow(2).sum(-1) - batch) / (batch * (batch - 1))


@torch.no_grad()
def key_inheritance(model, tokens: torch.Tensor) -> dict[str, np.ndarray]:
    """Rebuilds each head's attention from its queries and keys, then changes only position 0's key norm or direction."""
    names = ("attention", "norm ratio", "norm clamped", "direction replaced", "cosine to first", "cosine to others", "first key agreement", "later key agreement")
    totals = {kind: np.zeros((LAYERS, HEADS)) for kind in names}
    chunks = range(0, len(tokens), CHUNK)
    worst = 0.0
    rows = range(1, LENGTH - 1)
    for start in chunks:
        batch = tokens[start : start + CHUNK]
        store, hooks = captured_blocks(model)
        output = model(batch, output_attentions=True)
        for hook in hooks:
            hook.remove()
        for layer in range(LAYERS):
            query, key = split_head(store[layer], 0), split_head(store[layer], 1)
            scores = query @ key.transpose(-1, -2) / HEAD_SIZE**0.5
            rebuilt = attention_from_scores(scores)
            worst = max(worst, float((rebuilt - output.attentions[layer]).abs().max()))
            norms = key.norm(dim=-1)
            later_norm = norms[..., 1:].mean(-1)

            clamped = scores.clone()
            clamped[..., 0] = scores[..., 0] * (later_norm / norms[..., 0])[..., None]
            replaced = scores.clone()
            last = key[:, :, -1]
            replaced[..., 0] = (query * last[:, :, None]).sum(-1) / HEAD_SIZE**0.5 * (norms[..., 0] / norms[..., -1])[..., None]

            unit_query = torch.nn.functional.normalize(query, dim=-1)
            unit_key = torch.nn.functional.normalize(key, dim=-1)
            cosine = unit_query @ unit_key.transpose(-1, -2)
            earlier = torch.tril(torch.ones(LENGTH, LENGTH), -1)
            earlier[:, 0] = 0
            earlier[-1] = 0
            others = (cosine * earlier).sum((-1, -2)) / earlier.sum()

            values = {
                "attention": rebuilt[:, :, rows, 0].mean((0, 2)),
                "norm ratio": norms[..., 0].mean(0) / later_norm.mean(0),
                "norm clamped": attention_from_scores(clamped)[:, :, rows, 0].mean((0, 2)),
                "direction replaced": attention_from_scores(replaced)[:, :, rows, 0].mean((0, 2)),
                "cosine to first": cosine[:, :, rows, 0].mean((0, 2)),
                "cosine to others": others.mean(0),
                "first key agreement": mean_pair_cosine(key[:, :, 0]),
                "later key agreement": mean_pair_cosine(key[:, :, HALF]),
            }
            for kind, value in values.items():
                totals[kind][layer] += value.numpy() / len(chunks)
    print(f"  queries and keys rebuild the model's attention, largest difference {worst:.1e}")
    return {kind: value.ravel() for kind, value in totals.items()}


@torch.no_grad()
def residual_by_layer(model, tokens: torch.Tensor) -> None:
    """Prints the residual's size before each layer's norm at position 0 and later, and how much it changes across sequences."""
    store, hooks = {}, []
    for layer in range(LAYERS):
        def keep(_module, inputs, layer=layer):
            store[layer] = inputs[0].detach()

        hooks.append(model.transformer.h[layer].ln_1.register_forward_pre_hook(keep))
    model(tokens)
    for hook in hooks:
        hook.remove()
    for layer in range(LAYERS):
        residual = store[layer]
        print(f"    layer {layer}: norm at position 0 {residual[:, 0].norm(dim=-1).mean():.0f}, position {HALF} {residual[:, HALF].norm(dim=-1).mean():.0f}; agreement across sequences at position 0 {mean_pair_cosine(residual[:, 0]):.4f}, position {HALF} {mean_pair_cosine(residual[:, HALF]):.4f}")


@torch.no_grad()
def read_from_first(model, output, store: dict) -> dict[int, torch.Tensor]:
    """Returns what each head writes from position 0 alone, shaped like `run_with_heads`' per-head output."""
    per_head = {}
    for layer in range(LAYERS):
        value = split_head(store[layer], 2)[:, :, 0]
        written = output.attentions[layer][:, :, :, 0, None] * value[:, :, None]
        out = model.transformer.h[layer].attn.c_proj.weight
        per_head[layer] = torch.stack([written[:, head] @ out[head * HEAD_SIZE : (head + 1) * HEAD_SIZE] for head in range(HEADS)], 1)
    return per_head


def signed_effect(model, per_head: dict, final_input: torch.Tensor, positions: range, up) -> np.ndarray:
    """Returns each head's mean direct effect over positions, on the token `up(t)` picks, against the vocabulary mean."""
    effects = []
    for t in positions:
        at = torch.full((len(final_input),), t)
        effects.append(direct_effect(model, per_head, final_input, at, up(t)))
    return np.mean(effects, axis=0).ravel()


@torch.no_grad()
def logit_size(model, per_head: dict, final_input: torch.Tensor, positions: range, heads: list[tuple[int, int]]) -> dict[tuple[int, int], tuple[float, float]]:
    """Returns each head's direct effect, per position, as the root mean square logit and the largest logit in size."""
    unembed = model.lm_head.weight
    centred = (unembed - unembed.mean(0, keepdim=True)).T
    scale = model.transformer.ln_f.weight
    sizes = {}
    for layer, head in heads:
        root, largest = [], []
        for t in positions:
            deviation = final_input[:, t].std(-1, keepdim=True)
            logits = (per_head[layer][:, head, t] / deviation * scale) @ centred
            root.append(float(logits.pow(2).mean(-1).sqrt().mean()))
            largest.append(float(logits.abs().max(-1).values.mean()))
        sizes[(layer, head)] = (float(np.mean(root)), float(np.mean(largest)))
    return sizes


def effect_section(model, title: str, tokens: torch.Tensor, positions: range) -> dict[str, np.ndarray]:
    """Prints the direct effect of what the named heads read at position 0, next to their total direct effect."""
    store, hooks = captured_blocks(model)
    output, total, final_input = run_with_heads(model, tokens)
    for hook in hooks:
        hook.remove()
    first = read_from_first(model, output, store)
    next_token = lambda t: tokens[:, t + 1]
    first_token = lambda t: tokens[:, 0]
    results = {
        "next, read": signed_effect(model, first, final_input, positions, next_token),
        "next, total": signed_effect(model, total, final_input, positions, next_token),
        "first, read": signed_effect(model, first, final_input, positions, first_token),
        "first, total": signed_effect(model, total, final_input, positions, first_token),
    }
    read_size, total_size = logit_size(model, first, final_input, positions, NAMED), logit_size(model, total, final_input, positions, NAMED)
    print(f"  {title}: read at position 0 / total")
    for head in NAMED:
        i = index(head)
        print(
            f"    {name(head)}: next token {results['next, read'][i]:+.3f}/{results['next, total'][i]:+.3f}, "
            f"first token {results['first, read'][i]:+.3f}/{results['first, total'][i]:+.3f}, "
            f"root mean square logit {read_size[head][0]:.3f}/{total_size[head][0]:.3f}, "
            f"largest logit {read_size[head][1]:.3f}/{total_size[head][1]:.3f}"
        )
    five = [index(head) for head in TOP_INDUCTION]
    print(
        f"    five summed: next token {results['next, read'][five].sum():+.3f}/{results['next, total'][five].sum():+.3f}, "
        f"first token {results['first, read'][five].sum():+.3f}/{results['first, total'][five].sum():+.3f}"
    )
    return {"read rms": np.array([read_size[head][0] for head in TOP_INDUCTION]), "read largest": np.array([read_size[head][1] for head in TOP_INDUCTION]), "next": results["next, read"][five], "first": results["first, read"][five]}


def main() -> None:
    torch.manual_seed(SEED)
    shuffle = TokenShuffle()
    model = real_network(shuffle).double()
    model.set_attn_implementation("eager")
    vocabulary = len(shuffle.tokenizer)
    generator = torch.Generator().manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    print("0. Reproductions on the earlier checks' own tokens")
    print("  head_types first position (first half of a repeat, 32 sequences): " + named_line(random_token_patterns(model, vocabulary)["first position"]))
    half = torch.randint(0, vocabulary, (CHUNK, HALF), generator=torch.Generator().manual_seed(SEED))
    ladder_tokens = torch.cat([half, half], 1)
    _, ladder_heads, ladder_final = run_with_heads(model, ladder_tokens)
    repeat_effect = signed_effect(model, ladder_heads, ladder_final, range(HALF, 2 * HALF - 1), lambda t: ladder_tokens[:, t + 1])
    del ladder_heads
    print("  signed_ladder direct effect on repeats: " + ", ".join(f"{name(head)} {repeat_effect[index(head)]:.2f}" for head in TOP_INDUCTION))

    random_long = fresh_sequences(vocabulary, SAMPLES, generator)
    random_tokens = random_long[:, :LENGTH]
    real_long = story_tokens(shuffle, rng)
    real_tokens = real_long[:, :LENGTH]
    first_tokens, first_counts = real_tokens[:, 0].unique(return_counts=True)
    top_first = int(first_counts.argmax())
    print(f"\n1. Attention to position 0, {len(random_tokens)} random and {len(real_tokens)} real sequences of {LENGTH} tokens")
    print(f"  real text's commonest first token {shuffle.decode_token(int(first_tokens[top_first]))!r} in {first_counts[top_first] / len(real_tokens):.2f} of sequences, {len(first_tokens)} different first tokens")
    print(f"  uniform attention would give {np.mean([1 / (t + 1) for t in range(1, LENGTH)]):.3f} over positions 1 to {LENGTH - 1}, {np.mean([1 / (t + 1) for t in SECOND_HALF]):.3f} over {SECOND_HALF[0]} to {SECOND_HALF[-1]}")

    swapped_random = random_tokens.clone()
    swapped_random[:, 0] = random_long[:, LENGTH]
    swapped_vocabulary = real_tokens.clone()
    other_tokens = torch.from_numpy(rng.integers(0, vocabulary, len(real_tokens)))
    swapped_vocabulary[:, 0] = other_tokens
    swapped_text = real_tokens.clone()
    borrowed = real_long[:, LENGTH].roll(1)
    borrowed[borrowed == real_tokens[:, 0]] = shuffle.encode(" the")[0]
    swapped_text[:, 0] = borrowed
    assert (swapped_vocabulary[:, 0] != real_tokens[:, 0]).all() and (swapped_text[:, 0] != real_tokens[:, 0]).all()

    runs = {
        "random": attention_to(model, random_tokens),
        "random, first token swapped": attention_to(model, swapped_random),
        "real": attention_to(model, real_tokens),
        "real, first swapped for a random token": attention_to(model, swapped_vocabulary),
        "real, first swapped for a token from another story": attention_to(model, swapped_text),
    }
    means = {kind: run[:, :, 1:].mean((0, 2)) for kind, run in runs.items()}
    print(f"  Heads above {THRESHOLD} by layer 0 to 11, mean over positions 1 to {LENGTH - 1}")
    for kind, values in means.items():
        print(f"    {kind}: {counts_line(values)}")
    print(f"  Same, mean over positions {SECOND_HALF[0]} to {SECOND_HALF[-1]} only")
    for kind, run in runs.items():
        print(f"    {kind}: {counts_line(run[:, :, SECOND_HALF[0] :].mean((0, 2)))}")
    print("  Named heads, value and rank of 144")
    for kind, values in means.items():
        print(f"    {kind}: {named_line(values)}")
    print("  Top heads")
    for kind in ("random", "real"):
        print(f"    {kind}: {top_line(means[kind])}")

    print("  Swapping the first token, sequence by sequence")
    pairs = (("random", "random, first token swapped"), ("real", "real, first swapped for a random token"), ("real", "real, first swapped for a token from another story"))
    for base, swapped in pairs:
        change = np.abs(runs[base][:, :, 1:].mean(2) - runs[swapped][:, :, 1:].mean(2)).mean(0)
        correlation = np.corrcoef(means[base], means[swapped])[0, 1]
        print(f"    {swapped}: correlation with {base} over heads {correlation:.4f}; mean absolute change " + ", ".join(f"{name(head)} {change[index(head)]:.3f}" for head in NAMED))

    five = [index(head) for head in TOP_INDUCTION]
    mid_others = [i for i in range(ALL_HEADS) if is_mid(i) and i not in five]
    print("  The five against other heads in layers 4 to 9, mean attention to position 0: " + ", ".join(f"{kind} {means[kind][five].mean():.3f} against {means[kind][mid_others].mean():.3f}" for kind in ("random", "real")))

    print("\n2. What attention to position 0 is inherited from, on random tokens")
    keys = key_inheritance(model, random_tokens[: CHUNK * 4])
    high = keys["attention"] > THRESHOLD
    print(f"  over all 144 heads, rank correlation with attention to position 0: position 0's key norm ratio {spearmanr(keys['attention'], keys['norm ratio']).statistic:.3f}; difference in query cosine {spearmanr(keys['attention'], keys['cosine to first'] - keys['cosine to others']).statistic:.3f}")
    mid = np.array([is_mid(i) for i in range(ALL_HEADS)])
    print(f"  over layers 4 to 9: key norm ratio {spearmanr(keys['attention'][mid], keys['norm ratio'][mid]).statistic:.3f}; difference in query cosine {spearmanr(keys['attention'][mid], (keys['cosine to first'] - keys['cosine to others'])[mid]).statistic:.3f}")
    print(f"  heads above {THRESHOLD}: {high.sum()}; with position 0's key clamped to the mean norm {(keys['norm clamped'] > THRESHOLD).sum()}; with its direction replaced by the last position's key {(keys['direction replaced'] > THRESHOLD).sum()}")
    print(f"  median key norm ratio over all heads {np.median(keys['norm ratio']):.3f}, over heads above {THRESHOLD} {np.median(keys['norm ratio'][high]):.3f}, over the rest {np.median(keys['norm ratio'][~high]):.3f}")
    for head in NAMED:
        i = index(head)
        print(
            f"    {name(head)}: attention {keys['attention'][i]:.3f}, key norm ratio {keys['norm ratio'][i]:.3f}, norm clamped {keys['norm clamped'][i]:.3f}, "
            f"direction replaced {keys['direction replaced'][i]:.3f}, query cosine to position 0 {keys['cosine to first'][i]:.3f} against others {keys['cosine to others'][i]:.3f}, "
            f"key agreement across sequences at position 0 {keys['first key agreement'][i]:.3f} against position {HALF} {keys['later key agreement'][i]:.3f}"
        )

    embedding = model.transformer.wpe.weight.norm(dim=-1)
    print(f"  position embedding norms: position 0 {embedding[0]:.2f}, 1 {embedding[1]:.2f}, {HALF} {embedding[HALF]:.2f}, {LENGTH} {embedding[LENGTH]:.2f}; median token embedding {model.transformer.wte.weight.norm(dim=-1).median():.2f}")
    print("  Residual before each layer's norm")
    residual_by_layer(model, random_tokens[:CHUNK])

    positions = torch.arange(LENGTH).repeat(len(random_tokens), 1)
    moved = positions.clone()
    moved[:, 0] = LENGTH
    without = attention_to(model, random_tokens, position_ids=moved)[:, :, 1:].mean((0, 2))
    print(f"  position 0 given position {LENGTH}'s embedding: heads above {THRESHOLD}: {counts_line(without)}")
    print("    " + named_line(without))
    given = positions.clone()
    given[:, HALF] = 0
    targets = torch.full((LENGTH,), HALF)
    window = SECOND_HALF[0]
    normal_later = attention_to(model, random_tokens, targets)[:, :, window:].mean((0, 2))
    given_later = attention_to(model, random_tokens, targets, position_ids=given)[:, :, window:].mean((0, 2))
    print(f"  position {HALF} given position 0's embedding, attention to position {HALF} from later positions, normal to given: " + ", ".join(f"{name(head)} {normal_later[index(head)]:.3f} to {given_later[index(head)]:.3f}" for head in NAMED))

    print("\n3. Direct effect on the logits, 32 sequences, positions up to the last with a next token")
    random_effect = effect_section(model, "random tokens", random_tokens[:CHUNK], range(1, LENGTH - 1))
    real_effect = effect_section(model, "real text", real_tokens[:CHUNK], range(1, LENGTH - 1))
    half_tokens = random_tokens[:, :HALF]
    repeats = torch.cat([half_tokens, half_tokens], 1)
    effect_section(model, "repeated random tokens, second half", repeats[:CHUNK], range(SECOND_HALF[0], LENGTH - 1))

    print(f"\n4. Attention to position 0 at positions {SECOND_HALF[0]} to {SECOND_HALF[-1]}, a match against none, same first half, {len(repeats)} sequences")
    no_match = random_tokens
    match_run = attention_to(model, repeats)[:, :, SECOND_HALF[0] :].mean((0, 2))
    none_run = attention_to(model, no_match)[:, :, SECOND_HALF[0] :].mean((0, 2))
    targets = torch.arange(LENGTH) - HALF + 1
    targets[: SECOND_HALF[0]] = 0
    induction = attention_to(model, repeats, targets)[:, :, SECOND_HALF[0] :].mean((0, 2))
    fall = none_run - match_run
    relative = fall / none_run.clip(1e-9)
    for head in NAMED:
        i = index(head)
        print(f"    {name(head)}: no match {none_run[i]:.3f}, match {match_run[i]:.3f}, fall {fall[i]:.3f} (#{rank(fall, head)} of 144), kept {match_run[i] / none_run[i]:.3f} of it, attention to the match's next token {induction[i]:.3f}")
    print("  largest falls: " + top_line(fall))
    print("  largest rises: " + top_line(-fall))
    high_none = none_run > THRESHOLD
    sharp = high_none & (fall > SHARP_FALL)
    print(f"  heads above {THRESHOLD} with no match: {high_none.sum()}; of those, falling by more than {SHARP_FALL}: {sharp.sum()}: " + ", ".join(label(int(i)) for i in np.argsort(-fall) if sharp[i]))
    print(f"  of those above {THRESHOLD} and not falling sharply: " + ", ".join(f"{label(int(i))} {none_run[i]:.2f} to {match_run[i]:.2f}" for i in np.argsort(-none_run) if high_none[i] and not sharp[i]))
    print(f"  over heads above {THRESHOLD}, median fraction lost {np.median(relative[high_none]):.3f}; five's fraction lost " + ", ".join(f"{name(head)} {relative[index(head)]:.3f}" for head in TOP_INDUCTION))
    print(f"  rank correlation of fall with attention to the match's next token: all heads {spearmanr(fall, induction).statistic:.3f}, heads above {THRESHOLD} {spearmanr(fall[high_none], induction[high_none]).statistic:.3f}")
    print(f"  heads in layers 4 to 9 whose fall reaches the smallest of the five's ({fall[five].min():.3f}): " + ", ".join(label(i) for i in range(ALL_HEADS) if fall[i] >= fall[five].min() and is_mid(i)))

    print("\n5. The card's prediction, part by part")
    mid_count = int((means["random"].reshape(LAYERS, HEADS) > THRESHOLD).sum(1)[list(MID_LAYERS)].sum())
    print(f"  heads above {THRESHOLD} in layers 4 to 9 on random tokens, at least 20 predicted: {mid_count}")
    ranks = [rank(means["random"], head) for head in TOP_INDUCTION]
    print(f"  five's ranks on random tokens, all 144: {ranks}")
    print("  five, real below random and above 0.3: " + ", ".join(f"{name(head)} {means['real'][index(head)]:.3f} against {means['random'][index(head)]:.3f}" for head in TOP_INDUCTION))
    for title, effect in (("random tokens", random_effect), ("real text", real_effect)):
        print(f"  five's direct effect from position 0 on {title}, under 0.1 predicted: largest size, next token {np.abs(effect['next']).max():.3f}, first token {np.abs(effect['first']).max():.3f}, root mean square logit {effect['read rms'].max():.3f}, largest logit {effect['read largest'].max():.3f}")


if __name__ == "__main__":
    main()
