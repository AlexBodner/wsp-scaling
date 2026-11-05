"""Tests whether gpt2-medium's statistic picks feed its late high-attention heads through their queries. Makes no Astra calls.

On repeated random tokens, a first pass silences the source heads and stores each layer's input. A second pass gives only one
part (query or key) of the target heads the residual from the first pass, with everything else left on. A layer's input holds
nothing from that layer or later, so a source's silencing reaches only layers after it. Targets are the named late heads, or
every head in a layer after the first source. "Alone" rows rebuild one target's attention from the slices, with its other part
exactly intact, so nothing leaks between targets. Each row gives the late heads' attention by head and repeat accuracy.
"""
import resource
import time

import numpy as np
import torch

from checks.other_models import ABLATION_SEED, REPEAT_LENGTH, check_slices, head_count, head_size, input_by_output, layer_modules, name, qkv_starts
from checks.zero_order import (
    BATCH,
    BATCHES,
    CHUNK,
    CONTROL_SEED,
    CONTROL_SETS,
    DEVICE,
    MPS_LIMIT_GB,
    SLICE_TOLERANCE,
    check_hook,
    interval,
    load,
    repeated_tokens,
    same_layer_set,
    zero_hooks,
)

MODEL = "gpt2-medium"
WIDTH = 1024
HEADS = 16
HEAD_SIZE = 64
PICKS = [(6, 1), (7, 2), (5, 8), (6, 15), (9, 9)]
ATTENTION_TOP = [(9, 9), (18, 5), (11, 1), (12, 1), (7, 2)]
LATE = [(18, 5), (20, 12), (13, 14), (19, 15), (20, 7), (11, 1), (12, 1)]
# Saved numbers from checks/outputs/zero_order.txt and checks/outputs/medium_recipes.txt, to reproduce.
SAVED_SCORES = (0.950, 0.871, 0.812, 0.783, 0.721, 0.916, 0.895)
SAVED_INTACT = 0.9844
SAVED_COSTS = {"picks": 41.2, "attention top 5": 3.9}
SAVED_SAME_LAYER = (0.1, -0.2, 0.3)
# Zero_order's seed path for the best-pairing recipe at size 5, so these are the same control sets.
CONTROL_PATH = (CONTROL_SEED, 2, 5)
QUERY_CUT_AT_LEAST = 0.5
KEY_CUT_UNDER = 0.25
RANDOM_MOVE_UNDER = 0.1
MUCH_LESS_RATIO = 0.5
CHECK_TOLERANCE = 1e-5
ROLES = ("queries", "keys")
PATCHED = ("queries", "keys", "all later queries")
ALONE = ("queries alone", "keys alone")
peak_driver = 0


def flat(head: tuple[int, int]) -> int:
    return head[0] * HEADS + head[1]


def late_attention(attentions: tuple) -> np.ndarray:
    """Returns each late head's attention to the token after the earlier copy, by sequence, averaged over the second copy."""
    times = torch.arange(REPEAT_LENGTH, 2 * REPEAT_LENGTH - 1, device=attentions[0].device)
    rows = [attentions[layer][:, head][:, times, times - REPEAT_LENGTH + 1].mean(1) for layer, head in LATE]
    return torch.stack(rows, 1).cpu().double().numpy()


def check_memory() -> None:
    global peak_driver
    if DEVICE == "mps":
        peak_driver = max(peak_driver, torch.mps.driver_allocated_memory())
        torch.mps.empty_cache()
    if max(peak_driver, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1e9 > MPS_LIMIT_GB:
        raise MemoryError(f"over {MPS_LIMIT_GB} GB: driver {peak_driver / 1e9:.2f}, resident {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f}")


@torch.no_grad()
def forward(model, chunk: torch.Tensor, hooks: list) -> tuple[np.ndarray, np.ndarray]:
    """Runs the model with the given hooks, removes them, and returns late heads' attention and each sequence's repeat accuracy."""
    output = model.base_model(chunk, output_attentions=True, use_cache=False)
    for hook in hooks:
        hook.remove()
    logits = model.lm_head(output.last_hidden_state[:, REPEAT_LENGTH:-1])
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite logits")
    right = (logits.argmax(-1) == chunk[:, REPEAT_LENGTH + 1 :]).sum(1).cpu().numpy() / (REPEAT_LENGTH - 1)
    attention = late_attention(output.attentions)
    del output, logits
    check_memory()
    return attention, right


def input_stores(model, store: dict, layers) -> list:
    """Stores the given layers' inputs to the query-key-value map."""
    def save(_module, inputs, layer):
        store[layer] = inputs[0].detach()

    return [layer_modules(model, layer)[1].register_forward_pre_hook(lambda m, i, layer=layer: save(m, i, layer)) for layer in layers]


def patch_hooks(model, cached: dict, targets: list[int], role: int) -> list:
    """Gives only one part (0 query, 1 key) of the target heads the map of the cached input."""
    config, hooks = model.config, []
    for layer in sorted({index // HEADS for index in targets}):
        qkv = layer_modules(model, layer)[1]
        mask = torch.zeros(3 * WIDTH, dtype=torch.bool, device=qkv.weight.device)
        for index in targets:
            if index // HEADS == layer:
                start = qkv_starts(config, index % HEADS)[role]
                mask[start : start + HEAD_SIZE] = True

        def replace_part(module, _inputs, output, layer=layer, mask=mask):
            return torch.where(mask, cached[layer] @ input_by_output(module) + module.bias, output)

        hooks.append(qkv.register_forward_hook(replace_part))
    return hooks


@torch.no_grad()
def alone(model, clean: dict, cached: dict, head: tuple[int, int], role: int) -> np.ndarray:
    """Rebuilds one head's attention, by sequence, with one part from the cached input and the other from the clean one."""
    layer, index = head
    qkv = layer_modules(model, layer)[1]
    weight, bias = input_by_output(qkv), qkv.bias
    inputs = (cached, clean) if role == 0 else (clean, cached)
    q, k = (
        source[layer] @ weight[:, start : start + HEAD_SIZE] + bias[start : start + HEAD_SIZE]
        for source, start in zip(inputs, qkv_starts(model.config, index))
    )
    length = q.shape[1]
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=q.device))
    attention = (q @ k.transpose(-1, -2) / HEAD_SIZE**0.5).masked_fill(~causal, -torch.inf).softmax(-1)
    times = torch.arange(REPEAT_LENGTH, 2 * REPEAT_LENGTH - 1, device=q.device)
    return attention[:, times, times - REPEAT_LENGTH + 1].mean(1).cpu().double().numpy()


@torch.no_grad()
def check_patch(model, chunk: torch.Tensor) -> list[str]:
    """Checks the patch itself: no change with no sources, only later layers change, only the chosen columns change, and slices agree."""
    layers = model.config.num_hidden_layers
    picks, every = [flat(head) for head in PICKS], list(range(HEADS, head_count(model.config)))
    clean, cached, partial = {}, {}, {}
    base, base_right = forward(model, chunk, input_stores(model, clean, range(layers)))
    same, same_right = forward(model, chunk, patch_hooks(model, clean, every, 0))
    noop = float(np.abs(same - base).max())
    if noop > CHECK_TOLERANCE or not np.array_equal(same_right, base_right):
        raise ValueError(f"patching from the clean input changed attention by {noop}")
    forward(model, chunk, input_stores(model, cached, range(layers)) + zero_hooks(model, picks))
    if not all(torch.equal(cached[layer], clean[layer]) for layer in range(min(layer for layer, _ in PICKS) + 1)):
        raise ValueError("silencing changed a layer at or before the first source")
    forward(model, chunk, input_stores(model, partial, [8]) + zero_hooks(model, [flat(head) for head in PICKS if head[0] < 8]))
    if not torch.equal(cached[8], partial[8]) or torch.equal(cached[8], clean[8]):
        raise ValueError("layer 8's input depends on a source at or after layer 8, or on none")

    seen = {}
    first = LATE[np.argmin([layer for layer, _ in LATE])]
    worst_alone = 0.0
    position = LATE.index(first)
    for role in range(2):
        qkv = layer_modules(model, first[0])[1]
        hooks = [qkv.register_forward_hook(lambda _m, _i, output: seen.__setitem__("intact", output))]
        hooks += patch_hooks(model, cached, [flat(first)], role)
        hooks.append(qkv.register_forward_hook(lambda _m, _i, output: seen.__setitem__("patched", output)))
        hooks.append(qkv.register_forward_pre_hook(lambda _m, inputs: seen.__setitem__("input", inputs[0])))
        patched, _ = forward(model, chunk, hooks)
        if not torch.equal(seen["input"], clean[first[0]]):
            raise ValueError("the first target's input changed")
        changed = (seen["patched"] != seen["intact"]).any(0).any(0).nonzero().ravel().tolist()
        start = qkv_starts(model.config, first[1])[role]
        if not changed or min(changed) < start or max(changed) >= start + HEAD_SIZE:
            raise ValueError(f"patch changed columns {changed[:5]} to {changed[-5:]}, expected {start} to {start + HEAD_SIZE - 1}")
        worst_alone = max(worst_alone, float(np.abs(patched[:, position] - alone(model, clean, cached, first, role)).max()))
    if worst_alone > CHECK_TOLERANCE:
        raise ValueError(f"the slices and the model disagree on {name(flat(first), model.config)} by {worst_alone}")
    return [
        f"patching every later head from the clean input changes attention by at most {noop:.1e} and no predictions",
        "silencing leaves layers up to the first source exactly unchanged, and layer 8 sees only sources before it",
        f"patching {name(flat(first), model.config)} changes only its query or key columns; slices match the model's attention within {worst_alone:.1e}",
    ]


def row(attention: np.ndarray, right: np.ndarray, base: np.ndarray, base_right: np.ndarray) -> dict:
    fall = base.mean(1) - attention.mean(1)
    cost = 100 * (base_right - right) if right is not None else None
    return {
        "mean": float(attention.mean()), "by_head": attention.mean(0), "fall": float(fall.mean()), "fall_interval": interval(fall),
        "share": float(fall.mean() / base.mean()), "cost": None if cost is None else float(cost.mean()),
        "cost_interval": None if cost is None else interval(cost), "by_batch": None if cost is None else cost.reshape(BATCHES, BATCH).mean(1),
        "accuracy": None if right is None else float(right.mean()),
    }


def show(label: str, result: dict, base: np.ndarray, config) -> str:
    heads = ", ".join(f"{name(flat(head), config)} {base[i]:.3f} to {result['by_head'][i]:.3f}" for i, head in enumerate(LATE))
    low, high = result["fall_interval"]
    line = f"  {label}: mean attention {result['mean']:.3f}, fall {result['fall']:.3f} ({low:.3f} to {high:.3f}), {100 * result['share']:.1f}% of intact; {heads}"
    if result["cost"] is not None:
        line += (f"; repeat accuracy {result['accuracy']:.4f}, cost {result['cost']:.1f} ({result['cost_interval'][0]:.1f} to {result['cost_interval'][1]:.1f}), "
                 "by batch " + " ".join(f"{value:.1f}" for value in result["by_batch"]))
    return line


def main() -> None:
    began = time.time()
    if DEVICE == "mps":
        torch.mps.set_per_process_memory_fraction(MPS_LIMIT_GB * 1e9 / torch.mps.recommended_max_memory())
    model = load(MODEL)
    config = model.config
    if (config.hidden_size, config.num_attention_heads, head_size(config)) != (WIDTH, HEADS, HEAD_SIZE):
        raise ValueError(f"unexpected shape {config.hidden_size}, {config.num_attention_heads}, {head_size(config)}")
    layers, total = config.num_hidden_layers, head_count(config)
    print(f"{MODEL}: {layers} layers, {HEADS} heads of size {HEAD_SIZE}, width {WIDTH}, forward passes on {DEVICE} in float32", flush=True)

    worst = check_slices(model)
    if worst > SLICE_TOLERANCE:
        raise ValueError(f"slices do not match the model, error {worst}")
    print(f"  slices rebuild the model's attention and output in float32, worst error {worst:.2e}")
    tokens = torch.cat([repeated_tokens(ABLATION_SEED + 1 + batch, BATCH) for batch in range(BATCHES)])
    print("  " + check_hook(model, tokens, config))
    for line in check_patch(model, tokens[:CHUNK]):
        print("  " + line, flush=True)

    picks, attention_top = [flat(head) for head in PICKS], [flat(head) for head in ATTENTION_TOP]
    rng = np.random.default_rng(CONTROL_PATH)
    controls = [same_layer_set(picks, HEADS, total, rng) for _ in range(CONTROL_SETS)]
    print(f"  picks: {' '.join(name(i, config) for i in picks)}; attention top 5: {' '.join(name(i, config) for i in attention_top)}")
    print("  same-layer sets: " + "; ".join(" ".join(name(i, config) for i in chosen) for chosen in controls), flush=True)
    sets = {"picks": picks, "attention top 5": attention_top, **{f"same-layer {i + 1}": chosen for i, chosen in enumerate(controls)}}

    late_flat = [flat(head) for head in LATE]
    later = {label: [i for i in range(total) if i // HEADS > min(j // HEADS for j in sources)] for label, sources in sets.items()}
    named = {label: [i for i in late_flat if i in later[label]] for label in sets}
    if any(named[label] != late_flat for label in sets):
        raise ValueError("a named late head is not later than the sets' first source")
    late_layers = sorted({layer for layer, _ in LATE})

    probe = tokens[:CHUNK]
    cached = {}
    first = min(i // HEADS for i in picks)
    forward(model, probe, input_stores(model, cached, range(first + 1, layers)) + zero_hooks(model, picks))
    on_cpu, on_cpu_right = forward(model, probe, patch_hooks(model, cached, late_flat, 0))
    model.to(DEVICE)
    cached = {}
    chunk = probe.to(DEVICE)
    forward(model, chunk, input_stores(model, cached, range(first + 1, layers)) + zero_hooks(model, picks))
    on_device, on_device_right = forward(model, chunk, patch_hooks(model, cached, late_flat, 0))
    print(f"  cpu against {DEVICE} on {CHUNK} sequences with the picks patched into queries: worst attention gap {np.abs(on_cpu - on_device).max():.1e}, "
          f"worst accuracy gap {np.abs(on_cpu_right - on_device_right).max():.4f}", flush=True)

    conditions = [("silenced everywhere", None, None)] + [(label, part, role) for label, part, role in (("queries", "named", 0), ("keys", "named", 1), ("all later queries", "later", 0))]
    attention_by = {(label, kind): [] for label in sets for kind in [*[c[0] for c in conditions], *ALONE]}
    right_by = {(label, kind): [] for label in sets for kind, _, _ in conditions}
    base_attention, base_right = [], []
    for step, chunk in enumerate(tokens.split(CHUNK)):
        chunk = chunk.to(DEVICE)
        clean = {}
        attention, right = forward(model, chunk, input_stores(model, clean, late_layers))
        base_attention.append(attention)
        base_right.append(right)
        for label, sources in sets.items():
            cached = {}
            first = min(i // HEADS for i in sources)
            for kind, part, role in conditions:
                if part is None:
                    result = forward(model, chunk, input_stores(model, cached, range(first + 1, layers)) + zero_hooks(model, sources))
                else:
                    result = forward(model, chunk, patch_hooks(model, cached, (named if part == "named" else later)[label], role))
                attention_by[label, kind].append(result[0])
                right_by[label, kind].append(result[1])
            for role, kind in enumerate(ALONE):
                attention_by[label, kind].append(np.stack([alone(model, clean, cached, head, role) for head in LATE], 1))
            del cached
        del clean
        check_memory()
        if (step + 1) % (BATCH // CHUNK) == 0:
            print(f"  {(step + 1) * CHUNK} sequences done, {time.time() - began:.0f} s", flush=True)

    base, base_right = np.concatenate(base_attention), np.concatenate(base_right)
    base_heads = base.mean(0)
    print(f"\n  intact on {len(tokens)} sequences in {BATCHES} batches: repeat accuracy {base_right.mean():.4f} (saved {SAVED_INTACT:.4f}), late heads' mean attention {base.mean():.3f}")
    print("  intact attention by head against the saved score: " + ", ".join(f"{name(flat(head), config)} {base_heads[i]:.3f}/{SAVED_SCORES[i]:.3f}" for i, head in enumerate(LATE)))

    results = {}
    for label in sets:
        for kind in attention_by:
            if kind[0] != label:
                continue
            attention = np.concatenate(attention_by[kind])
            right = np.concatenate(right_by[kind]) if kind in right_by else None
            results[kind] = row(attention, right, base, base_right)

    print("\n  sources silenced, per intervention; cost is points of repeat accuracy, intact minus intervention; intervals resample sequences")
    for label in ("picks", "attention top 5"):
        for kind in ["silenced everywhere", *PATCHED, *ALONE]:
            print(show(f"{label}, {kind}", results[label, kind], base_heads, config))

    print(f"\n  {CONTROL_SETS} same-layer sets of the picks, fixed before the run")
    labels = [label for label in sets if label.startswith("same-layer")]
    for kind in ["silenced everywhere", *PATCHED, *ALONE]:
        rows = [results[label, kind] for label in labels]
        falls = [entry["fall"] for entry in rows]
        heads = np.stack([entry["by_head"] for entry in rows])
        line = (f"  {kind}: fall by set " + " ".join(f"{value:.3f}" for value in falls) + f"; mean {np.mean(falls):.3f}, largest move {max(map(abs, falls)):.3f}; "
                + ", ".join(f"{name(flat(head), config)} mean {heads[:, i].mean():.3f} lowest {heads[:, i].min():.3f}" for i, head in enumerate(LATE)))
        if rows[0]["cost"] is not None:
            costs = [entry["cost"] for entry in rows]
            line += "; cost by set " + " ".join(f"{value:.1f}" for value in costs) + f"; mean {np.mean(costs):.1f}, lowest {min(costs):.1f}, highest {max(costs):.1f}"
        print(line)

    print("\n  reproducing zero_order.py: silenced everywhere costs " + ", ".join(f"{label} {results[label, 'silenced everywhere']['cost']:.1f} (saved {SAVED_COSTS[label]})" for label in SAVED_COSTS)
          + f"; same-layer mean {np.mean([results[label, 'silenced everywhere']['cost'] for label in labels]):.1f} (saved {SAVED_SAME_LAYER[0]})"
          + f", lowest {min(results[label, 'silenced everywhere']['cost'] for label in labels):.1f} (saved {SAVED_SAME_LAYER[1]})"
          + f", highest {max(results[label, 'silenced everywhere']['cost'] for label in labels):.1f} (saved {SAVED_SAME_LAYER[2]})")

    query, key, attention_query = results["picks", "queries"], results["picks", "keys"], results["attention top 5", "queries"]
    moves = [abs(results[label, "queries"]["fall"]) for label in labels]
    tests = {
        f"picks through queries cut mean attention by at least {100 * QUERY_CUT_AT_LEAST:.0f}% ({100 * query['share']:.1f}%)": query["share"] >= QUERY_CUT_AT_LEAST,
        f"picks through keys cut it by under {100 * KEY_CUT_UNDER:.0f}% ({100 * key['share']:.1f}%)": key["share"] < KEY_CUT_UNDER,
        f"every same-layer set through queries moves it under {RANDOM_MOVE_UNDER} (largest {max(moves):.3f})": max(moves) < RANDOM_MOVE_UNDER,
        f"attention's top 5 through queries cut it much less, read as under {MUCH_LESS_RATIO} of the picks' fall ({attention_query['fall']:.3f} against {query['fall']:.3f})": attention_query["fall"] < MUCH_LESS_RATIO * query["fall"],
    }
    print("\n  card's prediction:")
    for text, held in tests.items():
        print(f"    {text}: {'held' if held else 'failed'}")
    print(f"  all four: {'held' if all(tests.values()) else 'failed'}")
    print(f"\n  took {time.time() - began:.0f} s, peak resident memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB, peak MPS driver memory {peak_driver / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
