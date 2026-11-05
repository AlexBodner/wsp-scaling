"""Tests whether zeroing the statistic's picks costs more repeat accuracy than zeroing heads from the same layers. Makes no Astra calls.

On gpt2-medium this zeroes the top heads by attention, best-pairing, sum_strong_writers and by-source picks at the top 5,
then cumulatively at 10, 20 and 40. Each pick set gets 8 random sets with the same number of heads from each layer,
excluding the picks; each size gets 8 uniformly random sets. distilgpt2 runs only if picks beat same-layer sets on medium.
Weights go through float64 and back after loading, because a freshly loaded gpt2-medium in float32 gives NaN logits.
"""
import gc
import resource
import time
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from checks.other_models import (
    ABLATION_BATCH,
    ABLATION_SEED,
    REPEAT_LENGTH,
    TOKEN_RANGE,
    check_slices,
    head_count,
    head_size,
    input_by_output,
    layer_modules,
    measured_scores,
    name,
    qkv_starts,
)

SIZES = (5, 10, 20, 40)
BATCHES = 4
BATCH = 32
CHUNK = 4
CONTROL_SETS = 8
CONTROL_SEED = 20261104
BOOTSTRAPS = 2000
STRONG_WRITERS = 8
BEAT_MARGIN = 10.0
PREDICTED_COST = 30.0
SLICE_TOLERANCE = 1e-2
MPS_LIMIT_GB = 3.0
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
STATISTIC = ("best_pairing", "sum_strong_writers", "by_source")
# Real head numbers from checks/outputs/medium_recipes.txt and checks/outputs/other_models.txt.
EXPECTED = {
    "gpt2-medium": {
        "attention": [(9, 9), (18, 5), (11, 1), (12, 1), (7, 2)],
        "best_pairing": [(6, 1), (7, 2), (5, 8), (6, 15), (9, 9)],
        "sum_strong_writers": [(9, 9), (7, 2), (12, 1), (11, 1), (10, 8)],
        "by_source": [(14, 9), (18, 8), (17, 2), (16, 15), (15, 8)],
    }
}
REPRODUCE = {"gpt2-medium": (0.989, 0.943)}
peak_driver = 0


def load(model_name: str) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, dtype=torch.float32).eval()
    model.set_attn_implementation("eager")
    for tensor in [*model.parameters(), *model.buffers()]:
        if tensor.is_floating_point():
            tensor.data = tensor.data.double().float()
    gc.collect()
    return model


def recipe_cosines(model) -> np.ndarray:
    """Returns the cosine of each head's QK^T (rows) with each earlier head's VO (columns), as medium_recipes.py computes it."""
    config, size = model.config, head_size(model.config)
    total, width = head_count(config), config.hidden_size
    q, k, v = (np.empty((total, width, size)) for _ in range(3))
    o = np.empty((total, size, width))
    for layer in range(config.num_hidden_layers):
        norm, qkv, out = layer_modules(model, layer)
        scale = norm.weight.detach().double().numpy()[:, None]
        weight = input_by_output(qkv).detach().double().numpy()
        output = input_by_output(out).detach().double().numpy()
        for head in range(config.num_attention_heads):
            index = layer * config.num_attention_heads + head
            for role, start in zip((q, k, v), qkv_starts(config, head)):
                part = scale * weight[:, start : start + size]
                role[index] = part - part.mean(0)
            o[index] = output[head * size : (head + 1) * size]
    v_q = v.transpose(0, 2, 1)
    reader_norms = np.sqrt(((q.transpose(0, 2, 1) @ q) * (k.transpose(0, 2, 1) @ k)).sum((1, 2)))
    writer_norms = np.sqrt(((v_q @ v) * (o @ o.transpose(0, 2, 1))).sum((1, 2)))
    cosine = np.zeros((total, total))
    for index in range(total):
        cosine[index] = ((v_q @ q[index]) * (o @ k[index])).sum((1, 2)) / (reader_norms[index] * writer_norms)
    layers = np.arange(total) // config.num_attention_heads
    cosine[layers[:, None] <= layers[None, :]] = -np.inf
    return cosine


def recipe_orders(cosine: np.ndarray, heads_per_layer: int) -> tuple[dict[str, np.ndarray], int]:
    """Returns each recipe's heads in order, leaving out heads the recipe cannot score, and the by-source recipe's source."""
    votes = np.bincount(cosine[heads_per_layer:].argmax(1), minlength=len(cosine))
    source = int(votes.argmax())
    strong = np.argsort(-cosine.max(0))[:STRONG_WRITERS]
    values = {
        "best_pairing": cosine.max(1),
        "sum_strong_writers": np.maximum(cosine[:, strong], 0).sum(1),
        "by_source": cosine[:, source],
    }
    orders = {}
    for recipe, value in values.items():
        order = np.argsort(-value)
        orders[recipe] = order[np.isfinite(value[order]) & (value[order] != 0)]
    return orders, source


def repeated_tokens(seed: int, batch: int) -> torch.Tensor:
    half = torch.randint(*TOKEN_RANGE, (batch, REPEAT_LENGTH), generator=torch.Generator().manual_seed(seed))
    return torch.cat([half, half], 1)


def zero_hooks(model, heads: list[int]) -> list:
    """Zeroes the given heads' values, and returns the hooks to remove afterwards."""
    config, size = model.config, head_size(model.config)
    per_layer = config.num_attention_heads
    hooks = []
    for layer in {index // per_layer for index in heads}:
        qkv = layer_modules(model, layer)[1]
        mask = torch.ones(3 * config.hidden_size, device=qkv.weight.device)
        for index in heads:
            if index // per_layer == layer:
                start = qkv_starts(config, index % per_layer)[2]
                mask[start : start + size] = 0
        hooks.append(qkv.register_forward_hook(lambda _module, _inputs, output, mask=mask: output * mask))
    return hooks


@torch.no_grad()
def accuracy_by_sequence(model, tokens: torch.Tensor, heads: list[int]) -> np.ndarray:
    """Returns each sequence's share of right top next tokens in the second copy, with the given heads zeroed."""
    global peak_driver
    device = next(model.parameters()).device
    hooks = zero_hooks(model, heads)
    right = []
    for chunk in tokens.split(CHUNK):
        chunk = chunk.to(device)
        logits = model.lm_head(model.base_model(chunk).last_hidden_state[:, REPEAT_LENGTH:-1])
        if not torch.isfinite(logits).all():
            raise ValueError(f"non-finite logits with heads {heads}")
        right.append((logits.argmax(-1) == chunk[:, REPEAT_LENGTH + 1 :]).sum(1).cpu().numpy())
        if device.type == "mps":
            peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())
            if peak_driver / 1e9 > MPS_LIMIT_GB:
                raise MemoryError(f"MPS driver at {peak_driver / 1e9:.2f} GB")
            del logits
            torch.mps.empty_cache()
    for hook in hooks:
        hook.remove()
    return np.concatenate(right) / (REPEAT_LENGTH - 1)


@torch.no_grad()
def check_hook(model, tokens: torch.Tensor, config) -> str:
    """Checks on the output map's input that exactly the chosen heads' columns are zero, and that a same-layer neighbour is untouched."""
    size, per_layer = head_size(config), config.num_attention_heads
    layer_a, layer_b = 2, 4
    chosen = [layer_a * per_layer + 1, layer_a * per_layer + per_layer - 1, layer_b * per_layer + 3]
    seen = {}
    capture = [
        layer_modules(model, layer)[2].register_forward_pre_hook(lambda _m, inputs, layer=layer: seen.__setitem__(layer, inputs[0]))
        for layer in (layer_a, layer_a + 1, layer_b)
    ]
    chunk = tokens[:2].to(next(model.parameters()).device)
    model.base_model(chunk)
    intact = dict(seen)
    hooks = zero_hooks(model, chosen)
    model.base_model(chunk)
    for hook in [*hooks, *capture]:
        hook.remove()
    columns = lambda output, head: output[..., head * size : (head + 1) * size]
    zero = lambda layer: [head for head in range(per_layer) if float(columns(seen[layer], head).abs().max()) == 0]
    if zero(layer_a) != [1, per_layer - 1] or zero(layer_a + 1) != [] or zero(layer_b) != [3]:
        raise ValueError(f"wrong heads zeroed: {zero(layer_a)}, {zero(layer_a + 1)}, {zero(layer_b)}")
    if not torch.equal(columns(seen[layer_a], 0), columns(intact[layer_a], 0)):
        raise ValueError("zeroing changed another head in the same layer")
    return f"zeroing {[name(index, config) for index in chosen]} zeroes only those heads' columns"


def interval(values: np.ndarray) -> tuple[float, float]:
    """Returns a 95% bootstrap interval for the mean, resampling sequences."""
    draws = np.random.default_rng(CONTROL_SEED).integers(0, len(values), (BOOTSTRAPS, len(values)))
    low, high = np.percentile(values[draws].mean(1), [2.5, 97.5])
    return float(low), float(high)


def same_layer_set(picks: list[int], per_layer: int, total: int, rng: np.random.Generator) -> list[int]:
    """Returns a random set with the picks' count from each layer, excluding the picks, capped by the heads left in the layer."""
    chosen = []
    counts = Counter(index // per_layer for index in picks)
    for layer, count in sorted(counts.items()):
        rest = [index for index in range(layer * per_layer, (layer + 1) * per_layer) if index not in picks]
        chosen += rng.choice(rest, min(count, len(rest)), replace=False).tolist()
    if set(chosen) & set(picks) or max(chosen, default=0) >= total:
        raise ValueError("same-layer set overlaps the picks or leaves the model")
    return chosen


def show(heads: np.ndarray, config) -> str:
    return " ".join(name(int(index), config) for index in heads)


def study(model_name: str) -> list[dict]:
    began = time.time()
    model = load(model_name)
    config = model.config
    per_layer, total = config.num_attention_heads, head_count(config)
    print(f"\n{model_name}: {config.num_hidden_layers} layers, {per_layer} heads, {total} heads in all, forward passes on {DEVICE} in float32", flush=True)

    cosine = recipe_cosines(model)
    orders, source = recipe_orders(cosine, per_layer)
    del cosine
    gc.collect()
    print(f"  recipes took {time.time() - began:.0f} s, by-source source {name(source, config)}", flush=True)
    worst = check_slices(model)
    if worst > SLICE_TOLERANCE:
        raise ValueError(f"slices do not match the model, error {worst}")
    print(f"  slices rebuild the model's attention and output in float32, worst error {worst:.2e}")
    scores, _ = measured_scores(model)
    orders = {"attention": np.argsort(-scores), **orders}
    print(f"  attention scores measured, {time.time() - began:.0f} s", flush=True)
    for recipe, order in orders.items():
        print(f"  {recipe} order: {show(order[:max(SIZES)], config)}")
        expected = EXPECTED.get(model_name, {}).get(recipe)
        if expected is not None:
            found = [(int(i) // per_layer, int(i) % per_layer) for i in order[:SIZES[0]]]
            print(f"    top 5 {'match the saved output' if found == expected else f'DIFFER from the saved output {expected}'}", flush=True)
    print("  scores of the top attention heads: " + ", ".join(f"{name(int(i), config)} {scores[i]:.3f}" for i in orders["attention"][:SIZES[0]]))

    tokens = torch.cat([repeated_tokens(ABLATION_SEED + 1 + batch, BATCH) for batch in range(BATCHES)])
    probe = orders["best_pairing"][:SIZES[0]].tolist()
    on_cpu = accuracy_by_sequence(model, tokens[:CHUNK], probe)
    print("  " + check_hook(model, tokens, config))
    model.to(DEVICE)
    gap = np.abs(on_cpu - accuracy_by_sequence(model, tokens[:CHUNK], probe)).max()
    print(f"  cpu against {DEVICE} on {CHUNK} sequences with best-pairing picks zeroed: worst per-sequence gap {gap:.4f}", flush=True)

    if model_name in REPRODUCE:
        earlier = repeated_tokens(ABLATION_SEED, ABLATION_BATCH)
        intact_then = accuracy_by_sequence(model, earlier, []).mean()
        without_then = accuracy_by_sequence(model, earlier, orders["attention"][:SIZES[0]].tolist()).mean()
        print(f"  reproducing other_models.py on its {ABLATION_BATCH} sequences: intact {intact_then:.3f}, top 5 zeroed {without_then:.3f}, "
              f"saved {REPRODUCE[model_name][0]:.3f}, {REPRODUCE[model_name][1]:.3f}", flush=True)

    intact = accuracy_by_sequence(model, tokens, [])
    batches = intact.reshape(BATCHES, BATCH).mean(1)
    print(f"  intact repeat accuracy on {len(tokens)} sequences in {BATCHES} batches: {intact.mean():.4f}, "
          f"95% interval {interval(intact)[0]:.4f} to {interval(intact)[1]:.4f}, by batch " + " ".join(f"{a:.4f}" for a in batches), flush=True)

    uniform_costs = {}
    for size in SIZES:
        rng = np.random.default_rng([CONTROL_SEED, size])
        sets = [rng.choice(total, size, replace=False).tolist() for _ in range(CONTROL_SETS)]
        uniform_costs[size] = [100 * (intact - accuracy_by_sequence(model, tokens, chosen)) for chosen in sets]
        means = [float(cost.mean()) for cost in uniform_costs[size]]
        print(f"  {size} uniformly random heads, {CONTROL_SETS} sets: cost mean {np.mean(means):.1f}, lowest {min(means):.1f}, highest {max(means):.1f}", flush=True)

    print("  costs below are points of repeat accuracy, intact minus zeroed; intervals resample sequences and hold the control sets fixed")
    rows = []
    for size in SIZES:
        for recipe_index, (recipe, order) in enumerate(orders.items()):
            if len(order) < size:
                print(f"  {recipe} top {size}: skipped, the recipe scores only {len(order)} heads")
                continue
            picks = order[:size].tolist()
            cost = 100 * (intact - accuracy_by_sequence(model, tokens, picks))
            rng = np.random.default_rng([CONTROL_SEED, recipe_index + 1, size])
            controls = [same_layer_set(picks, per_layer, total, rng) for _ in range(CONTROL_SETS)]
            same_layer = [100 * (intact - accuracy_by_sequence(model, tokens, chosen)) for chosen in controls]
            means = [float(entry.mean()) for entry in same_layer]
            excess = cost - np.mean(same_layer, axis=0)
            uniform = np.mean([float(entry.mean()) for entry in uniform_costs[size]])
            row = {
                "recipe": recipe, "size": size, "cost": float(cost.mean()), "same_layer": float(np.mean(means)),
                "highest": max(means), "excess": float(excess.mean()), "excess_interval": interval(excess),
            }
            rows.append(row)
            print(f"  {recipe} top {size}: cost {cost.mean():.1f} ({interval(cost)[0]:.1f} to {interval(cost)[1]:.1f}), by batch "
                  + " ".join(f"{value:.1f}" for value in cost.reshape(BATCHES, BATCH).mean(1))
                  + f"; same-layer mean {np.mean(means):.1f}, lowest {min(means):.1f}, highest {max(means):.1f}, "
                  f"heads per set {len(controls[0])} of {size}; uniform mean {uniform:.1f}; "
                  f"picks minus same-layer {excess.mean():.1f} ({row['excess_interval'][0]:.1f} to {row['excess_interval'][1]:.1f})", flush=True)

    print("  each best-pairing pick zeroed alone:")
    for index in orders["best_pairing"][:SIZES[0]].tolist():
        cost = 100 * (intact - accuracy_by_sequence(model, tokens, [index]))
        print(f"    {name(index, config)}: cost {cost.mean():.1f} ({interval(cost)[0]:.1f} to {interval(cost)[1]:.1f}), attention score {scores[index]:.3f}", flush=True)

    first = next(row for row in rows if row["recipe"] == "best_pairing" and row["size"] == SIZES[0])
    print(f"  card's prediction: best-pairing top 5 costs at least {PREDICTED_COST:.0f} ({first['cost']:.1f}), "
          f"same-layer within {BEAT_MARGIN:.0f} ({first['excess']:.1f}): {'held' if first['cost'] >= PREDICTED_COST and abs(first['excess']) <= BEAT_MARGIN else 'failed'}")
    print(f"  took {time.time() - began:.0f} s", flush=True)
    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()
    return rows


def main() -> None:
    if DEVICE == "mps":
        torch.mps.set_per_process_memory_fraction(MPS_LIMIT_GB * 1e9 / torch.mps.recommended_max_memory())
    rows = study("gpt2-medium")
    # Picks beat same-layer sets when they cost over 10 points more than the mean and more than every set.
    beat = [row for row in rows if row["recipe"] in STATISTIC and row["excess"] > BEAT_MARGIN and row["cost"] > row["highest"]]
    print(f"\npick sets beating same-layer sets by over {BEAT_MARGIN:.0f} points and every set: " + (", ".join(f"{row['recipe']} top {row['size']}" for row in beat) or "none"))
    if beat:
        study("distilbert/distilgpt2")
    else:
        print("distilgpt2 not run: no pick set beat its same-layer sets on medium")
    print(f"\npeak resident memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB, peak MPS driver memory {peak_driver / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
