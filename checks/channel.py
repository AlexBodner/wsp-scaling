"""Tests whether 4.11's output subspace is the channel behind the five heads' pattern, on real GPT-2. Makes no Astra calls.

The right singular vectors of 4.11's VO are where it writes into the residual. This removes them from the residual that
only the five heads' keys or queries read, before the layer norm, and compares same-rank random subspaces. It also measures
token self-match through E and block 0's MLP. Parts marked as added after the card are the top 8 previous-token heads'
union, removal that keeps the subspace's mean, and other heads' subspaces, which test how specific the drop is.
"""
import numpy as np
import torch

from checks.how_statistic_classifies import REPEAT_LENGTH, SAMPLED_TOKENS, SEED, repeated_tokens, self_match, silenced
from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH
from wsp.network import TokenShuffle, real_network

FIVE = [(5, 5), (7, 10), (6, 9), (5, 1), (7, 2)]
FIVE_LAYERS = sorted({layer for layer, _ in FIVE})
SOURCE = (4, 11)
WRITERS = [(4, 11), (3, 7), (2, 2), (3, 2), (2, 9), (2, 5), (3, 8), (3, 6)]
RANKS = (1, 2, 4, 8, 16, 32, 64)
PER_WRITER = (1, 2, 4, 8, 16, 32, 64)
RANDOM_SETS = 8
CARD_RANK = 64
CONTROL_HEADS = 12
OTHER_HEADS = 8
PARTS = (("keys", 1), ("queries", 0))
THREADS = 4
CHUNK = 8
SECOND_HALF = torch.arange(REPEAT_LENGTH, 2 * REPEAT_LENGTH - 1)


def name(head: tuple[int, int]) -> str:
    return f"{head[0]}.{head[1]}"


def part(model, head: tuple[int, int], offset: int) -> np.ndarray:
    """Returns one part of a head (0 query, 1 key, 2 value) as WIDTH by HEAD_SIZE, including the layer norm's scale."""
    block = model.transformer.h[head[0]]
    start = offset * WIDTH + head[1] * HEAD_SIZE
    return (block.ln_1.weight[:, None] * block.attn.c_attn.weight[:, start : start + HEAD_SIZE]).detach().numpy()


def value_output(model, head: tuple[int, int]) -> np.ndarray:
    """Returns VO, which takes a row vector from the residual to what the head writes back into it."""
    output = model.transformer.h[head[0]].attn.c_proj.weight[head[1] * HEAD_SIZE : (head[1] + 1) * HEAD_SIZE].detach().numpy()
    return part(model, head, 2) @ output


def query_key(model, head: tuple[int, int]) -> np.ndarray:
    return part(model, head, 0) @ part(model, head, 1).T


def random_subspace(rng: np.random.Generator, rank: int) -> np.ndarray:
    return np.linalg.qr(rng.standard_normal((WIDTH, rank)))[0].T


def share_inside(matrix: np.ndarray, basis: np.ndarray) -> float:
    """Returns how much of a matrix's columns' energy lies inside a subspace given as orthonormal rows."""
    return float(np.square(basis @ matrix).sum() / np.square(matrix).sum())


@torch.no_grad()
def residuals(model, tokens: torch.Tensor, layers: list[int], without: list[tuple[int, int]] | None = None) -> dict[int, torch.Tensor]:
    """Returns the residual just before the layer norm in the given layers, optionally with some heads silenced."""
    cached, hooks = {layer: [] for layer in layers}, []
    for layer in layers:
        def store(_module, inputs, layer=layer):
            cached[layer].append(inputs[0].detach())

        hooks.append(model.transformer.h[layer].ln_1.register_forward_pre_hook(store))
    hooks += silenced(model, without or [])
    for chunk in tokens.split(CHUNK):
        model.transformer(chunk)
    for hook in hooks:
        hook.remove()
    return {layer: torch.cat(pieces) for layer, pieces in cached.items()}


@torch.no_grad()
def removed(model, tokens: torch.Tensor, residual: dict[int, torch.Tensor], basis: np.ndarray | None, offset: int, mode: str, means: dict[int, torch.Tensor]):
    """Runs one chunk with a subspace removed from what one part of the five heads reads, or unchanged with no subspace.

    Modes: zero removes it before the layer norm, mean does the same but keeps the subspace's average over all tokens,
    after_norm removes it after the layer norm, and live uses the current run's residual rather than the given one.
    """
    if basis is None:
        return model(tokens, output_attentions=True, logits_to_keep=REPEAT_LENGTH)
    rows = torch.from_numpy(basis)
    source, hooks = dict(residual), []
    for layer in FIVE_LAYERS:
        chosen = [head for current, head in FIVE if current == layer]
        block = model.transformer.h[layer]

        def store(_module, inputs, layer=layer):
            source[layer] = inputs[0]

        def replace_part(module, _inputs, output, layer=layer, chosen=chosen, norm=block.ln_1):
            current = source[layer]
            if mode == "after_norm":
                normed = norm(current)
                changed_input = normed - normed @ rows.T @ rows
            elif mode == "mean":
                changed_input = norm(current - (current - means[layer]) @ rows.T @ rows)
            else:
                changed_input = norm(current - current @ rows.T @ rows)
            changed = changed_input @ module.weight + module.bias
            output = output.clone()
            for head in chosen:
                start = offset * WIDTH + head * HEAD_SIZE
                output[..., start : start + HEAD_SIZE] = changed[..., start : start + HEAD_SIZE]
            return output

        if mode == "live":
            hooks.append(block.ln_1.register_forward_pre_hook(store))
        hooks.append(block.attn.c_attn.register_forward_hook(replace_part))
    output = model(tokens, output_attentions=True, logits_to_keep=REPEAT_LENGTH)
    for hook in hooks:
        hook.remove()
    return output


def scored(output, tokens: torch.Tensor) -> tuple[np.ndarray, float]:
    """Returns the five's induction attention and the repeat accuracy, as attention_patterns measures them."""
    attention = [output.attentions[layer][:, head, SECOND_HALF, SECOND_HALF - REPEAT_LENGTH + 1].mean() for layer, head in FIVE]
    predicted = output.logits[:, :-1].argmax(-1)
    return np.array([float(value) for value in attention]), float((predicted == tokens[:, REPEAT_LENGTH + 1 :]).float().mean())


def chunks(model, tokens: torch.Tensor, residual: dict[int, torch.Tensor], basis: np.ndarray | None, offset: int, mode: str):
    """Yields each chunk's tokens and output, so a forward pass never holds the whole batch's logits."""
    means = {layer: values.mean((0, 1), keepdim=True) for layer, values in residual.items()}
    for start in range(0, len(tokens), CHUNK):
        part_tokens = tokens[start : start + CHUNK]
        part_residual = {layer: values[start : start + CHUNK] for layer, values in residual.items()}
        yield part_tokens, removed(model, part_tokens, part_residual, basis, offset, mode, means)


def measured(model, tokens: torch.Tensor, residual: dict[int, torch.Tensor], basis: np.ndarray | None, offset: int = 1, mode: str = "zero") -> tuple[np.ndarray, float]:
    runs = [scored(output, chunk) for chunk, output in chunks(model, tokens, residual, basis, offset, mode)]
    return np.mean([run[0] for run in runs], axis=0), float(np.mean([run[1] for run in runs]))


def per_head(values: np.ndarray) -> str:
    return " ".join(f"{name(head)} {value:.3f}" for head, value in zip(FIVE, values))


def where_it_goes(model, tokens: torch.Tensor, residual: dict[int, torch.Tensor], basis: np.ndarray | None, offset: int = 1, mode: str = "zero") -> str:
    """Returns where the five's attention lands in the second half, averaged over the five."""
    targets = {"next after the match": SECOND_HALF - REPEAT_LENGTH + 1, "the match, which is the first position at 1 of 24 steps": SECOND_HALF - REPEAT_LENGTH, "itself": SECOND_HALF, "previous": SECOND_HALF - 1, "first position": torch.zeros_like(SECOND_HALF)}
    runs = []
    for _, output in chunks(model, tokens, residual, basis, offset, mode):
        shares = [np.mean([float(output.attentions[layer][:, head, SECOND_HALF, target].mean()) for layer, head in FIVE]) for target in targets.values()]
        largest = np.mean([float(output.attentions[layer][:, head, SECOND_HALF].max(-1).values.mean()) for layer, head in FIVE])
        runs.append(shares + [largest])
    average = np.mean(runs, axis=0)
    return ", ".join(f"{label} {share:.3f}" for label, share in zip(targets, average)) + f", largest single position {average[-1]:.3f}"


def apart_from_first(model, tokens: torch.Tensor, residual: dict[int, torch.Tensor], basis: np.ndarray | None, offset: int = 1, mode: str = "zero") -> str:
    """Returns the first position's share, and the right position's share of the attention left on all the others."""
    target = SECOND_HALF - REPEAT_LENGTH + 1
    first, others = [], []
    for _, output in chunks(model, tokens, residual, basis, offset, mode):
        attentions = [output.attentions[layer][:, head, SECOND_HALF] for layer, head in FIVE]
        first.append([float(attention[..., 0].mean()) for attention in attentions])
        others.append([float((attention[:, torch.arange(len(target)), target] / attention[..., 1:].sum(-1)).mean()) for attention in attentions])
    conditional = np.mean(others, axis=0)
    return f"first position {np.mean(first):.3f}, right position among the rest {conditional.mean():.3f}, per head {per_head(conditional)}"


def compared(model, tokens: torch.Tensor, residual: dict[int, torch.Tensor], basis: np.ndarray, rng: np.random.Generator, label: str) -> dict[str, tuple[float, np.ndarray]]:
    """Prints a subspace against same-rank random ones, for keys and queries, and returns (attention, random means) by part."""
    rank = len(basis)
    controls = [random_subspace(rng, rank) for _ in range(RANDOM_SETS)]
    results = {}
    for part_name, offset in PARTS:
        attention, accuracy = measured(model, tokens, residual, basis, offset)
        print(f"  {label}, {part_name}: mean {attention.mean():.3f}, per head {per_head(attention)}, repeat accuracy {accuracy:.3f}", flush=True)
        runs = [measured(model, tokens, residual, control, offset) for control in controls]
        random_attention = np.stack([run[0] for run in runs])
        means = random_attention.mean(1)
        print(f"    {RANDOM_SETS} random rank {rank}: mean {means.mean():.3f}, lowest {means.min():.3f}, highest {means.max():.3f}, per head mean {per_head(random_attention.mean(0))}, repeat accuracy mean {np.mean([run[1] for run in runs]):.3f}", flush=True)
        results[part_name] = (float(attention.mean()), means)
    return results


def checks(model, tokens: torch.Tensor, residual: dict[int, torch.Tensor], source_vo: np.ndarray, baseline: np.ndarray, accuracy: float) -> None:
    left, singular, right = np.linalg.svd(source_vo, full_matrices=False)
    block = model.transformer.h[SOURCE[0]]
    print("1. Checks before trusting anything")
    print(f"  shapes: c_attn weight {tuple(block.attn.c_attn.weight.shape)}, c_proj weight {tuple(block.attn.c_proj.weight.shape)}, VO {source_vo.shape}, right singular rows {right.shape}")
    numeric_rank = int((singular > 1e-10 * singular[0]).sum())
    print(f"  VO of {name(SOURCE)} has numeric rank {numeric_rank}, so rank {CARD_RANK} is its whole row space and no rank above {numeric_rank} exists for it")
    print(f"  singular values: 1st {singular[0]:.2f}, 8th {singular[7]:.2f}, 16th {singular[15]:.2f}, 32nd {singular[31]:.2f}, 64th {singular[63]:.2f}, 65th {singular[64]:.2e}")
    print(f"  basis orthonormal: largest error {np.abs(right[:CARD_RANK] @ right[:CARD_RANK].T - np.eye(CARD_RANK)).max():.1e}")
    print(f"  nothing changed: repeat accuracy {accuracy:.3f}, the five's mean induction attention {baseline.mean():.3f}, per head {per_head(baseline)}")

    empty = np.zeros((0, WIDTH))
    for part_name, offset in PARTS:
        attention, _ = measured(model, tokens, residual, empty, offset)
        print(f"  rank 0 through the recompute, {part_name}: mean {attention.mean():.3f}, largest change {np.abs(attention - baseline).max():.1e}")
    for label, sources in (("top 8 previous-token heads", WRITERS), (name(SOURCE), [SOURCE])):
        patched = residuals(model, tokens, FIVE_LAYERS, sources)
        for part_name, offset in PARTS:
            attention, patched_accuracy = measured(model, tokens, patched, empty, offset)
            print(f"  {label} silenced for {part_name} only: {attention.mean():.3f}, repeat accuracy {patched_accuracy:.3f}", flush=True)

    captured = {}

    def store(_module, _inputs, output):
        start = 2 * WIDTH + SOURCE[1] * HEAD_SIZE
        captured["values"] = output[..., start : start + HEAD_SIZE].detach()

    hook = block.attn.c_attn.register_forward_hook(store)
    with torch.no_grad():
        attentions = model.transformer(tokens[: CHUNK], output_attentions=True).attentions[SOURCE[0]][:, SOURCE[1]]
    hook.remove()
    source_input = residuals(model, tokens[: CHUNK], [SOURCE[0]])[SOURCE[0]].numpy()
    output_rows = block.attn.c_proj.weight[SOURCE[1] * HEAD_SIZE : (SOURCE[1] + 1) * HEAD_SIZE]
    written = (attentions @ captured["values"] @ output_rows).detach().numpy()
    flat_written = written.reshape(-1, WIDTH)
    energy = np.square(flat_written).sum()
    print(f"  what {name(SOURCE)} really writes on the first {CHUNK} sequences, share inside its top right singular directions: " + ", ".join(f"rank {rank} {np.square(flat_written @ right[:rank].T).sum() / energy:.4f}" for rank in RANKS))
    print(f"    same share with the left singular directions, the wrong side, rank {CARD_RANK}: {np.square(flat_written @ left[:, :CARD_RANK]).sum() / energy:.4f}; chance {CARD_RANK / WIDTH:.4f}")

    standard = (source_input - source_input.mean(-1, keepdims=True)) / source_input.std(-1, keepdims=True)
    previous, actual = standard[:, :-1].reshape(-1, WIDTH), written[:, 1:].reshape(-1, WIDTH)
    actual = actual - actual.mean(0)
    for label, matrix in (("e @ VO", source_vo), ("e @ VO^T", source_vo.T)):
        predicted = previous @ matrix
        predicted = predicted - predicted.mean(0)
        print(f"  cosine of the previous position's normed residual through {label} with what {name(SOURCE)} really writes: {(predicted * actual).sum() / np.linalg.norm(predicted) / np.linalg.norm(actual):.3f}")
    print("  layer norm scale, since the directions live before it: " + ", ".join(f"layer {layer} min {scale.min():.2f} median {np.median(scale):.2f} max {scale.max():.2f}" for layer in FIVE_LAYERS for scale in [model.transformer.h[layer].ln_1.weight.detach().numpy()]))


def token_match(model, query_keys: dict[tuple[int, int], np.ndarray], source_vo: np.ndarray, others: list[tuple[int, int]], embeddings: np.ndarray) -> dict[str, dict[str, float]]:
    with torch.no_grad():
        block = model.transformer.h[0]
        table = torch.from_numpy(embeddings)
        mlp = block.mlp(block.ln_2(table)).numpy()

    def standard(values: np.ndarray) -> np.ndarray:
        return (values - values.mean(1, keepdims=True)) / values.std(1, keepdims=True)

    def written_match(inputs: np.ndarray, reader: np.ndarray, matrix: np.ndarray) -> float:
        written = inputs @ matrix
        scores = (inputs @ reader) @ written.T
        return float((scores.argmax(1) == np.arange(len(inputs))).mean())

    inputs = {"raw E": standard(embeddings), "E through block 0's MLP alone": standard(mlp), "E plus block 0's MLP": standard(embeddings + mlp)}
    results = {"as imported": {}, "corrected": {}}
    print(f"3. Token self-match, {SAMPLED_TOKENS} sampled tokens, chance {1 / SAMPLED_TOKENS:.4f}")
    print("  self_match from how_statistic_classifies, the form behind the card's 0.020, which writes with e @ VO^T")
    for label, values in inputs.items():
        matches = [self_match(values, query_keys[head], source_vo) for head in FIVE]
        control = np.mean([self_match(values, query_keys[head], source_vo) for head in others])
        print(f"    {label}: the five {np.mean(matches):.3f}, per head {per_head(np.array(matches))}; {CONTROL_HEADS} random later heads {control:.3f}")
        results["as imported"][label] = float(np.mean(matches))
    print("  the same with the write as the model does it, e @ VO, which also matches trace(QK^T (VO)^T)")
    for label, values in inputs.items():
        matches = [written_match(values, query_keys[head], source_vo) for head in FIVE]
        control = np.mean([written_match(values, query_keys[head], source_vo) for head in others])
        print(f"    {label}: the five {np.mean(matches):.3f}, per head {per_head(np.array(matches))}; {CONTROL_HEADS} random later heads {control:.3f}")
        results["corrected"][label] = float(np.mean(matches))
    return results


def main() -> None:
    torch.set_num_threads(THREADS)
    shuffle = TokenShuffle()
    model = real_network(shuffle).double()
    model.set_attn_implementation("eager")
    tokens = repeated_tokens(len(shuffle.tokenizer))
    residual = residuals(model, tokens, FIVE_LAYERS)
    baseline, accuracy = measured(model, tokens, residual, None)
    vo = {head: value_output(model, head) for head in WRITERS}
    right = {head: np.linalg.svd(matrix, full_matrices=False)[2] for head, matrix in vo.items()}
    rng = np.random.default_rng(SEED + 1)
    checks(model, tokens, residual, vo[SOURCE], baseline, accuracy)

    print(f"2. Removing {name(SOURCE)}'s top right singular directions from what only the five's keys or queries read, on {REPEAT_LENGTH} random tokens repeated")
    results = {}
    for rank in RANKS:
        results[rank] = compared(model, tokens, residual, right[SOURCE][:rank], rng, f"rank {rank}")

    rng_tokens = np.random.default_rng(SEED)
    table = model.transformer.wte.weight.detach().numpy()
    embeddings = table[rng_tokens.choice(len(table), SAMPLED_TOKENS, replace=False)].copy()
    candidates = np.arange((SOURCE[0] + 1) * HEADS, LAYERS * HEADS)
    others = [(int(i) // HEADS, int(i) % HEADS) for i in rng_tokens.choice(candidates, CONTROL_HEADS, replace=False)]
    query_keys = {head: query_key(model, head) for head in FIVE + others}
    matched = token_match(model, query_keys, vo[SOURCE], others, embeddings)
    del query_keys

    print(f"4. Added after the card: union of top directions from the top 8 previous-token heads ({' '.join(name(head) for head in WRITERS)})")
    unions = {}
    for count in PER_WRITER:
        stacked = np.concatenate([right[head][:count] for head in WRITERS])
        _, singular, rows = np.linalg.svd(stacked, full_matrices=False)
        rank = int((singular > 1e-10 * singular[0]).sum())
        print(f"  top {count} from each head: union rank {rank} of {len(stacked)}, smallest singular value {singular[rank - 1]:.3f}")
        unions[count] = rows[:rank]
        compared(model, tokens, residual, unions[count], rng, f"union rank {rank}")

    print("5. Added after the card, because the drop overshoots: how specific is it?")
    basis = right[SOURCE][:CARD_RANK]
    control = random_subspace(rng, CARD_RANK)
    print(f"  where the five's attention goes, nothing changed: {where_it_goes(model, tokens, residual, None)}")
    for label, chosen in ((f"{name(SOURCE)} rank {CARD_RANK}", basis), (f"random rank {CARD_RANK}", control)):
        print(f"  where it goes, {label} removed from keys: {where_it_goes(model, tokens, residual, chosen)}", flush=True)
    print(f"  where it goes, {name(SOURCE)} rank {CARD_RANK} removed from queries: {where_it_goes(model, tokens, residual, basis, 0)}")
    print(f"  where it goes, {name(SOURCE)} rank {CARD_RANK} removed from keys, keeping its mean: {where_it_goes(model, tokens, residual, basis, 1, 'mean')}", flush=True)

    print(f"  apart from the first position, nothing changed: {apart_from_first(model, tokens, residual, None)}")
    for rank in RANKS:
        print(f"  apart from the first position, {name(SOURCE)} rank {rank} removed from keys: {apart_from_first(model, tokens, residual, right[SOURCE][:rank])}", flush=True)
    print(f"  apart from the first position, random rank {CARD_RANK} removed from keys: {apart_from_first(model, tokens, residual, control)}")
    print(f"  apart from the first position, union rank {len(unions[PER_WRITER[3]])} removed from keys: {apart_from_first(model, tokens, residual, unions[PER_WRITER[3]])}", flush=True)
    for layer in FIVE_LAYERS:
        values = residual[layer].numpy().reshape(-1, WIDTH)
        deviation = values - values.mean(0)
        pieces = []
        for label, chosen in ((f"{name(SOURCE)} rank {CARD_RANK}", basis), (f"random rank {CARD_RANK}", control)):
            kept = values - values @ chosen.T @ chosen
            pieces.append(f"{label}: energy {np.square(values @ chosen.T).sum() / np.square(values).sum():.3f}, energy around the mean {np.square(deviation @ chosen.T).sum() / np.square(deviation).sum():.3f}, layer norm's divisor falls to {kept.std(-1).mean() / values.std(-1).mean():.3f} of itself")
        print(f"  layer {layer} residual removed, " + "; ".join(pieces))
    for part_name, offset in PARTS:
        print(f"  share of the five's {part_name} weights inside {name(SOURCE)}'s rank {CARD_RANK}: " + per_head(np.array([share_inside(part(model, head, offset), basis) for head in FIVE])) + f"; chance {CARD_RANK / WIDTH:.3f}")
    print(f"  share of other writers' VO inside {name(SOURCE)}'s rank {CARD_RANK}: " + " ".join(f"{name(head)} {share_inside(vo[head].T, basis):.3f}" for head in WRITERS[1:]) + f"; chance {CARD_RANK / WIDTH:.3f}")

    print("  variants of removing from keys and queries, each against same-rank random where shown")
    for label, mode in (("removed after the layer norm, directions unscaled", "after_norm"), ("from the live residual, so changed heads feed later ones", "live")):
        attention, repeat = measured(model, tokens, residual, basis, 1, mode)
        print(f"    {label}, rank {CARD_RANK}, keys: mean {attention.mean():.3f}, per head {per_head(attention)}, repeat accuracy {repeat:.3f}", flush=True)
    print("    keeping the subspace's mean, so only what varies by token is removed:")
    for rank in RANKS:
        for part_name, offset in PARTS:
            attention, repeat = measured(model, tokens, residual, right[SOURCE][:rank], offset, "mean")
            print(f"      {name(SOURCE)} rank {rank}, {part_name}: mean {attention.mean():.3f}, per head {per_head(attention)}, repeat accuracy {repeat:.3f}", flush=True)
    for part_name, offset in PARTS:
        means = [measured(model, tokens, residual, random_subspace(rng, CARD_RANK), offset, "mean")[0].mean() for _ in range(RANDOM_SETS)]
        print(f"      {RANDOM_SETS} random rank {CARD_RANK}, {part_name}: mean {np.mean(means):.3f}, lowest {np.min(means):.3f}", flush=True)
    for count, rows in unions.items():
        attention, repeat = measured(model, tokens, residual, rows, 1, "mean")
        print(f"      union rank {len(rows)}, keys: mean {attention.mean():.3f}, per head {per_head(attention)}, repeat accuracy {repeat:.3f}", flush=True)
    for count in (PER_WRITER[3], PER_WRITER[-1]):
        rank = len(unions[count])
        means = [measured(model, tokens, residual, random_subspace(rng, rank), 1, "mean")[0].mean() for _ in range(RANDOM_SETS)]
        print(f"      {RANDOM_SETS} random rank {rank}, keys: mean {np.mean(means):.3f}, lowest {np.min(means):.3f}", flush=True)

    with torch.no_grad():
        plain_attentions = model.transformer(tokens[: CHUNK], output_attentions=True).attentions
    candidates = [(layer, head) for layer in range(1, 5) for head in range(HEADS) if (layer, head) not in WRITERS]
    picked = [candidates[i] for i in rng.choice(len(candidates), OTHER_HEADS, replace=False)]
    print(f"  {OTHER_HEADS} random other heads in layers 1 to 4, their own VO's row space removed, rank {CARD_RANK} (their previous-token attention in brackets)")
    for head in picked:
        rows = np.linalg.svd(value_output(model, head), full_matrices=False)[2][:CARD_RANK]
        previous = float(plain_attentions[head[0]][:, head[1], torch.arange(1, REPEAT_LENGTH), torch.arange(0, REPEAT_LENGTH - 1)].mean())
        zero, zero_accuracy = measured(model, tokens, residual, rows, 1)
        kept, kept_accuracy = measured(model, tokens, residual, rows, 1, "mean")
        query, _ = measured(model, tokens, residual, rows, 0)
        print(f"    {name(head)} ({previous:.2f}): keys {zero.mean():.3f} (repeat accuracy {zero_accuracy:.3f}), keys keeping its mean {kept.mean():.3f} (repeat accuracy {kept_accuracy:.3f}), queries {query.mean():.3f}; share of the five's key weights inside {np.mean([share_inside(part(model, five, 1), rows) for five in FIVE]):.3f}; keys, {apart_from_first(model, tokens, residual, rows)}", flush=True)

    print("6. The card's prediction")
    keys, keys_random = results[CARD_RANK]["keys"]
    queries, queries_random = results[CARD_RANK]["queries"]
    print(f"  rank {CARD_RANK} on keys under 0.5: {keys:.3f}, {'held' if keys < 0.5 else 'did not hold'}")
    print(f"  same-rank random above 0.8 on keys: mean {keys_random.mean():.3f}, {'held' if keys_random.mean() > 0.8 else 'did not hold'}; lowest {keys_random.min():.3f}, {'held' if keys_random.min() > 0.8 else 'did not hold'}")
    print(f"  queries as a control: {queries:.3f}, random mean {queries_random.mean():.3f}, lowest {queries_random.min():.3f}")
    for label, values in matched.items():
        print(f"  token self-match above 0.1, {label}: " + ", ".join(f"{key} {value:.3f}" for key, value in values.items()))


if __name__ == "__main__":
    main()
