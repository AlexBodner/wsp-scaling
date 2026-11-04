"""Tests why the runs' statistic finds induction heads, on real GPT-2. Makes no Astra calls.

The statistic is trace(QK^T (VO)^T), the score a token gets against itself, averaged over random directions, when an
earlier head writes it and a later head searches for it. This measures what every head attends to, how well the
statistic sorts all heads, whether the earlier head tracks the previous token, whether the match is token-specific
once the embeddings Astra never had are added, and what silencing the earlier head does.
"""
import numpy as np
import torch
from scipy.stats import spearmanr

from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH
from wsp.network import TokenShuffle, real_network, writable_tensors

REPEAT_LENGTH = 25
BATCH = 32
SEED = 20261104
NAMED = 5
SAMPLED_TOKENS = 2000
CONTROL_HEADS = 12
CONTROL_SETS = 8
SOURCE = (4, 11)


def name(index: int) -> str:
    return f"{index // HEADS}.{index % HEADS}"


def repeated_tokens(vocabulary: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    half = torch.randint(0, vocabulary, (BATCH, REPEAT_LENGTH), generator=generator)
    return torch.cat([half, half], 1)


def silenced(model, heads: list[tuple[int, int]]) -> list:
    """Zeroes the given heads' value vectors, and returns the hooks to remove afterwards."""
    hooks = []
    for layer in {layer for layer, _ in heads}:
        chosen = [head for current, head in heads if current == layer]

        def zero_values(_module, _inputs, output, chosen=chosen):
            output = output.clone()
            for head in chosen:
                start = 2 * WIDTH + head * HEAD_SIZE
                output[..., start : start + HEAD_SIZE] = 0
            return output

        hooks.append(model.transformer.h[layer].attn.c_attn.register_forward_hook(zero_values))
    return hooks


def measured_without(model, tokens: torch.Tensor, heads: list[tuple[int, int]]) -> tuple[np.ndarray, float]:
    """Returns each head's induction attention and the repeat accuracy with the given heads silenced."""
    hooks = silenced(model, heads)
    try:
        patterns, accuracy = attention_patterns(model, tokens)
    finally:
        for hook in hooks:
            hook.remove()
    return patterns["induction"].ravel(), accuracy


@torch.no_grad()
def attention_patterns(model, tokens: torch.Tensor) -> tuple[dict[str, np.ndarray], float]:
    """Returns each head's attention to the three positions that define a pattern, and repeat accuracy."""
    output = model(tokens, output_attentions=True)
    second_half = range(REPEAT_LENGTH, 2 * REPEAT_LENGTH - 1)
    patterns = {pattern: np.zeros((LAYERS, HEADS)) for pattern in ("induction", "duplicate", "previous")}
    for layer, attention in enumerate(output.attentions):
        mean = attention.mean(0).numpy()
        for t in second_half:
            patterns["induction"][layer] += mean[:, t, t - REPEAT_LENGTH + 1] / len(second_half)
            patterns["duplicate"][layer] += mean[:, t, t - REPEAT_LENGTH] / len(second_half)
        patterns["previous"][layer] = np.mean([mean[:, t, t - 1] for t in range(1, REPEAT_LENGTH)], axis=0)
    predicted = output.logits[:, REPEAT_LENGTH:-1].argmax(-1)
    return patterns, float((predicted == tokens[:, REPEAT_LENGTH + 1 :]).float().mean())


@torch.no_grad()
def path_patched(model, tokens: torch.Tensor, sources: list[tuple[int, int]], targets: list[int], offset: int) -> tuple[np.ndarray, float]:
    """Gives only one part of the target heads (0 query, 1 key) the residual they'd see with the sources silenced."""
    cached, hooks = {}, []
    target_layers = {head // HEADS for head in targets}
    for layer in target_layers:
        def store(_module, inputs, layer=layer):
            cached[layer] = inputs[0].detach()

        hooks.append(model.transformer.h[layer].attn.c_attn.register_forward_pre_hook(store))
    hooks += silenced(model, sources)
    model(tokens)
    for hook in hooks:
        hook.remove()

    hooks = []
    for layer in target_layers:
        chosen = [head % HEADS for head in targets if head // HEADS == layer]

        def replace_part(module, _inputs, output, layer=layer, chosen=chosen):
            output = output.clone()
            changed = cached[layer] @ module.weight + module.bias
            for head in chosen:
                start = offset * WIDTH + head * HEAD_SIZE
                output[..., start : start + HEAD_SIZE] = changed[..., start : start + HEAD_SIZE]
            return output

        hooks.append(model.transformer.h[layer].attn.c_attn.register_forward_hook(replace_part))
    patterns, accuracy = attention_patterns(model, tokens)
    for hook in hooks:
        hook.remove()
    return patterns["induction"].ravel(), accuracy


def head_matrices(tensors: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Returns QK^T and VO for every head, including the layer norm's scale."""
    query_key, value_output = [], []
    for layer in range(LAYERS):
        scale = tensors[f"L{layer}.norm1.scale"][:, None]
        weight, output = scale * tensors[f"L{layer}.qkv.W"], tensors[f"L{layer}.out.W"]
        for head in range(HEADS):
            start = head * HEAD_SIZE
            q, k, v = (weight[:, offset + start : offset + start + HEAD_SIZE] for offset in (0, WIDTH, 2 * WIDTH))
            query_key.append(q @ k.T)
            value_output.append(v @ output[start : start + HEAD_SIZE])
    return query_key, value_output


def self_match(embeddings: np.ndarray, query_key: np.ndarray, value_output: np.ndarray) -> float:
    """Returns how often a token scores itself higher than any sampled token, when one head writes and the other reads."""
    written = embeddings @ value_output.T
    scores = (embeddings @ query_key) @ written.T
    return float((scores.argmax(1) == np.arange(len(embeddings))).mean())


def ranked(values: np.ndarray, count: int = NAMED) -> list[int]:
    return [int(i) for i in np.argsort(-values.ravel())[:count]]


def main() -> None:
    shuffle = TokenShuffle()
    model = real_network(shuffle).double()
    model.set_attn_implementation("eager")
    tokens = repeated_tokens(len(shuffle.tokenizer))
    patterns, baseline = attention_patterns(model, tokens)
    tensors = {key: value.detach().numpy().astype(np.float64) for key, value in writable_tensors(model).items()}
    query_key, value_output = head_matrices(tensors)
    a = np.stack([matrix.ravel() for matrix in query_key])
    b = np.stack([matrix.ravel() for matrix in value_output])
    cosine = a @ b.T / np.outer(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
    earlier = np.arange(144)[:, None] // HEADS > np.arange(144)[None, :] // HEADS
    best = np.where(earlier, cosine, -np.inf).max(1)
    source = SOURCE[0] * HEADS + SOURCE[1]
    with_source = np.where(earlier[:, source], cosine[:, source], -np.inf)
    induction = patterns["induction"].ravel()
    top = ranked(induction, 10)
    later = np.arange(HEADS, 144)

    print(f"1. What the five heads and their source attend to, on {REPEAT_LENGTH} random tokens repeated")
    for pattern, values in patterns.items():
        print(f"  top five by {pattern}: " + ", ".join(f"{name(i)} {values.ravel()[i]:.2f}" for i in ranked(values)))
    print(f"  source {name(source)}: " + ", ".join(f"{pattern} {values.ravel()[source]:.2f} (rank {int((-values.ravel()).argsort().tolist().index(source)) + 1})" for pattern, values in patterns.items()))

    print("2. How the statistic sorts every head")
    order = (-with_source).argsort().tolist()
    print(f"  top ten by cosine with {name(source)}: " + ", ".join(f"{name(i)} {with_source[i]:.2f} (induction {induction[i]:.2f})" for i in order[:10]))
    print(f"  ranks of the top ten by induction score, cosine with {name(source)}: " + ", ".join(f"{name(i)} {order.index(i) + 1}" for i in top))
    best_order = (-best).argsort().tolist()
    print("  same, each head's best earlier pairing: " + ", ".join(f"{name(i)} {best_order.index(i) + 1}" for i in top))
    print(f"  Spearman with induction score over {len(later)} later heads: cosine with source {spearmanr(with_source[later], induction[later]).statistic:.2f}, best pairing {spearmanr(best[later], induction[later]).statistic:.2f}")

    print("3. The same idea on another pattern: a head's own trace(QK^T), which needs no earlier head")
    trace = np.array([np.trace(matrix) / np.linalg.norm(matrix) for matrix in query_key])
    for pattern, values in patterns.items():
        overlap = len(set(ranked(trace)) & set(ranked(values)))
        print(f"  {pattern}: top five by trace share {overlap} of {NAMED}, Spearman {spearmanr(trace, values.ravel()).statistic:.2f}")
    print("  top five by trace: " + ", ".join(f"{name(i)} {trace[i]:.2f}" for i in ranked(trace)))

    print("4. Token-specific match with the embeddings Astra never had")
    rng = np.random.default_rng(SEED)
    embeddings = tensors["E"][rng.choice(len(tensors["E"]), SAMPLED_TOKENS, replace=False)]
    embeddings = (embeddings - embeddings.mean(1, keepdims=True)) / embeddings.std(1, keepdims=True)
    groups = {
        "top five, written by source": top[:NAMED],
        "next five, written by source": top[NAMED:],
        "12 random later heads, written by source": [int(i) for i in rng.choice(np.arange(source // HEADS * HEADS + HEADS, 144), CONTROL_HEADS, replace=False)],
    }
    for label, heads in groups.items():
        matches = [self_match(embeddings, query_key[head], value_output[source]) for head in heads]
        print(f"  {label}: tokens matching themselves {np.mean(matches):.3f} (chance {1 / SAMPLED_TOKENS:.4f}), per head " + " ".join(f"{m:.2f}" for m in matches[:NAMED]))
    random_writer = [int(i) for i in rng.choice(np.arange(3 * HEADS, 5 * HEADS), CONTROL_HEADS, replace=False) if i != source]
    matches = [np.mean([self_match(embeddings, query_key[head], value_output[writer]) for head in top[:NAMED]]) for writer in random_writer]
    print(f"  top five, written by other heads in layers 3 and 4: {np.mean(matches):.3f}, highest {max(matches):.3f}")

    print("5. Silencing the source: does the pattern depend on it?")
    five = top[:NAMED]
    print(f"  nothing silenced: repeat accuracy {baseline:.3f}, the five's mean induction attention {induction[five].mean():.3f}")
    candidates = [int(i) for i in np.arange(1 * HEADS, 5 * HEADS)]
    previous = patterns["previous"].ravel()
    previous_heads = sorted((i for i in candidates), key=lambda i: -previous[i])
    for size in (1, 2, 3, 5, 8):
        heads = [(i // HEADS, i % HEADS) for i in previous_heads[:size]]
        attention, accuracy = measured_without(model, tokens, heads)
        print(f"  top {size} previous-token heads in layers 1 to 4 silenced ({' '.join(name(i) for i in previous_heads[:size])}): repeat accuracy {accuracy:.3f}, the five's induction attention {attention[five].mean():.3f}")
    others = [i for i in candidates if i not in previous_heads[:8]]
    for size in (1, 5, 8):
        results = []
        for _ in range(CONTROL_SETS):
            picked = rng.choice(others, size, replace=False)
            attention, accuracy = measured_without(model, tokens, [(int(i) // HEADS, int(i) % HEADS) for i in picked])
            results.append((accuracy, attention[five].mean()))
        results = np.array(results)
        print(f"  {CONTROL_SETS} random sets of {size} other heads in the same layers: repeat accuracy mean {results[:, 0].mean():.3f}, lowest {results[:, 0].min():.3f}; induction attention mean {results[:, 1].mean():.3f}, lowest {results[:, 1].min():.3f}")
    attention, accuracy = measured_without(model, tokens, [(i // HEADS, i % HEADS) for i in five])
    print(f"  the five themselves silenced: repeat accuracy {accuracy:.3f}")

    print("7. Which path carries it: silence the top 8 previous-token heads only for the five heads' keys or queries")
    sources = [(i // HEADS, i % HEADS) for i in previous_heads[:8]]
    for label, offset in (("keys only", 1), ("queries only", 0)):
        attention, accuracy = path_patched(model, tokens, sources, five, offset)
        print(f"  {label}: the five's induction attention {attention[five].mean():.3f}, per head " + " ".join(f"{name(i)} {attention[i]:.2f}" for i in five) + f", repeat accuracy {accuracy:.3f}")
    picked_sets = [[(int(i) // HEADS, int(i) % HEADS) for i in rng.choice(others, 8, replace=False)] for _ in range(CONTROL_SETS)]
    for label, offset in (("keys only", 1), ("queries only", 0)):
        results = []
        for picked in picked_sets:
            attention, accuracy = path_patched(model, tokens, picked, five, offset)
            results.append((attention[five].mean(), accuracy))
        results = np.array(results)
        print(f"  {CONTROL_SETS} random sets of 8 other heads, {label}: attention mean {results[:, 0].mean():.3f}, lowest {results[:, 0].min():.3f}; repeat accuracy mean {results[:, 1].mean():.3f}, lowest {results[:, 1].min():.3f}")

    print("6. Would another previous-token head point to the same five?")
    for writer in previous_heads[:5]:
        values = np.where(earlier[:, writer], cosine[:, writer], -np.inf)
        picks = ranked(values)
        print(f"  cosine with {name(writer)} (previous {previous[writer]:.2f}): top five {' '.join(name(i) for i in picks)}, {len(set(picks) & set(five))} of {NAMED}")
    votes = np.bincount(np.argmax(np.where(earlier, cosine, -np.inf)[HEADS:], axis=1), minlength=144)
    print("  heads most often the best earlier pairing: " + ", ".join(f"{name(i)} {votes[i]} (previous {previous[i]:.2f})" for i in ranked(votes.astype(float))))


if __name__ == "__main__":
    main()
