"""Tests which routes carry the cost of silencing gpt2-medium's best-pairing picks. Makes no Astra calls.

On repeated random tokens, a first pass silences the source heads and stores what each layer's attention and MLP read. Then:
"final residual": later layers see the sources as usual, and the sources' own output, as computed in that run, is subtracted from
what reaches the final layer norm. "Later values, queries, keys": every head in a layer after the first source gets only that part
from the first pass, as path_patched and medium_ladder.py do, so the head's changed output still reaches later layers' other parts,
later MLPs and the final norm. "Later MLPs" reads the first pass's MLP input from the first source's layer on. Later heads include
any sources in later layers. Combined routes apply several of these together; all of them together must give the silenced model
back, which is checked. One further route gives later heads the first pass's values with queries and keys held at the intact run's,
so changed values reach only MLPs and the final norm. Controls are medium_ladder.py's 8 same-layer sets through each route.
Pass a sequence count to debug.
"""
import resource
import sys
import time

import numpy as np
import torch

from checks import medium_ladder
from checks.medium_ladder import (
    CONTROL_PATH,
    HEAD_SIZE,
    HEADS,
    LATE,
    MODEL,
    PICKS,
    SAVED_INTACT,
    WIDTH,
    check_patch,
    flat,
    forward,
    row,
    show,
)
from checks.other_models import ABLATION_SEED, REPEAT_LENGTH, check_slices, head_count, head_size, input_by_output, layer_modules, name, qkv_starts
from checks.token_negative import load
from checks.zero_order import BATCH, BATCHES, CHUNK, CONTROL_SETS, DEVICE, MPS_LIMIT_GB, SLICE_TOLERANCE, check_hook, repeated_tokens, same_layer_set, zero_hooks

ROLE = {"values": 2, "queries": 0, "keys": 1}
ROUTES = {
    "silenced everywhere": None,
    "final residual": ("final",),
    "later values": ("values",),
    "later values, queries and keys held intact": ("values", "intact queries", "intact keys"),
    "later queries": ("queries",),
    "later keys": ("keys",),
    "later MLPs": ("mlps",),
    "queries and keys": ("queries", "keys"),
    "queries, keys and values": ("queries", "keys", "values"),
    "final and values": ("final", "values"),
    "final and queries": ("final", "queries"),
    "final, queries and values": ("final", "queries", "values"),
    "final, queries, keys and values": ("final", "queries", "keys", "values"),
    "queries, keys, values and MLPs": ("queries", "keys", "values", "mlps"),
    "all routes": ("final", "queries", "keys", "values", "mlps"),
}
# Saved numbers from checks/outputs/medium_ladder.txt, to reproduce.
SAVED_COSTS = {"silenced everywhere": 41.2, "later queries": 10.3}
SAVED_SAME_LAYER = (0.1, -0.2, 0.3)
FINAL_AT_LEAST = 20.0
VALUES_UNDER = 10.0
LAST_LAYER_HEAD = (23, 3)
STATE_TOLERANCE = 2e-2
FINAL_TOLERANCE = 2e-3


def input_stores(model, store: dict, layers) -> list:
    """Stores on the cpu what the given layers' query-key-value maps read: a pass needs near 1.5 GB of MPS room above the model."""
    def save(_module, inputs, layer):
        store[layer] = inputs[0].detach().cpu()

    return [layer_modules(model, layer)[1].register_forward_pre_hook(lambda m, i, layer=layer: save(m, i, layer)) for layer in layers]


def patch_hooks(model, cached: dict, targets: list[int], role: int) -> list:
    """Gives only one part (0 query, 1 key, 2 value) of the target heads the map of the stored input, as medium_ladder.py does."""
    hooks = []
    for layer in sorted({index // HEADS for index in targets}):
        qkv = layer_modules(model, layer)[1]
        mask = torch.zeros(3 * WIDTH, dtype=torch.bool, device=qkv.weight.device)
        for index in targets:
            if index // HEADS == layer:
                start = qkv_starts(model.config, index % HEADS)[role]
                mask[start : start + HEAD_SIZE] = True

        def replace_part(module, _inputs, output, layer=layer, mask=mask):
            return torch.where(mask, cached[layer].to(output.device) @ input_by_output(module) + module.bias, output)

        hooks.append(qkv.register_forward_hook(replace_part))
    return hooks


def mlp_stores(model, store: dict, layers) -> list:
    """Stores on the cpu what the given layers' MLPs read."""
    def save(_module, inputs, layer):
        store[layer] = inputs[0].detach().cpu()

    return [model.base_model.h[layer].mlp.register_forward_pre_hook(lambda m, i, layer=layer: save(m, i, layer)) for layer in layers]


def mlp_patches(model, cached: dict) -> list:
    """Gives the stored layers' MLPs the stored input."""
    return [model.base_model.h[layer].mlp.register_forward_pre_hook(lambda _m, inputs, layer=layer: (cached[layer].to(inputs[0].device),)) for layer in cached]


def final_hooks(model, sources: list[int]) -> list:
    """Adds up the sources' output as each layer computes it, and subtracts the total from what reaches the final layer norm."""
    total, hooks = {}, []
    for layer in sorted({index // HEADS for index in sources}):
        out = layer_modules(model, layer)[2]
        chosen = [index % HEADS for index in sources if index // HEADS == layer]

        def add(module, inputs, chosen=chosen):
            for head in chosen:
                part = head * HEAD_SIZE, (head + 1) * HEAD_SIZE
                written = inputs[0][..., part[0] : part[1]] @ input_by_output(module)[part[0] : part[1]]
                total["x"] = written if "x" not in total else total["x"] + written

        hooks.append(out.register_forward_pre_hook(add))
    hooks.append(model.base_model.ln_f.register_forward_pre_hook(lambda _m, inputs: (inputs[0] - total["x"],)))
    return hooks


def route_hooks(model, parts: tuple, sources: list[int], later: list[int], inputs: dict, mlps: dict, intact: dict | None = None) -> list:
    hooks = []
    for part in parts:
        if part == "final":
            hooks += final_hooks(model, sources)
        elif part == "mlps":
            hooks += mlp_patches(model, mlps)
        elif part.startswith("intact "):
            hooks += patch_hooks(model, intact, later, ROLE[part.split()[1]])
        else:
            hooks += patch_hooks(model, inputs, later, ROLE[part])
    return hooks


def first_pass(model, chunk: torch.Tensor, sources: list[int]) -> tuple:
    """Runs the model with the sources silenced, and returns its late attention and accuracy, and what later parts read."""
    first = min(index // HEADS for index in sources)
    inputs, mlps = {}, {}
    layers = model.config.num_hidden_layers
    hooks = input_stores(model, inputs, range(first + 1, layers)) + mlp_stores(model, mlps, range(first, layers)) + zero_hooks(model, sources)
    attention, right = forward(model, chunk, hooks)
    return attention, right, inputs, mlps


@torch.no_grad()
def final_state(model, chunk: torch.Tensor, hooks: list, seen: dict | None = None) -> torch.Tensor:
    """Runs the model with the given hooks, removes them, and returns the final hidden state, storing the final norm's input in `seen`."""
    if seen is not None:
        hooks = hooks + [model.base_model.ln_f.register_forward_pre_hook(lambda _m, inputs: seen.__setitem__("final", inputs[0].detach()))]
    state = model.base_model(chunk, use_cache=False).last_hidden_state
    for hook in hooks:
        hook.remove()
    if DEVICE == "mps":
        # MPS runs lazily, and a pass nobody reads holds all its intermediates until something forces it.
        torch.mps.synchronize()
    medium_ladder.check_memory()
    return state


@torch.no_grad()
def check_routes(model, chunk: torch.Tensor, control: list[int]) -> list[str]:
    """Checks each route: no change with no sources, only what it should touch changes, and all routes together give the silenced model."""
    config = model.config
    layers = config.num_hidden_layers
    picks = [flat(head) for head in PICKS]
    first = min(index // HEADS for index in picks)
    every = list(range((first + 1) * HEADS, head_count(config)))
    lines = []

    clean, clean_mlps, seen = {}, {}, {}
    intact = final_state(model, chunk, input_stores(model, clean, range(layers)) + mlp_stores(model, clean_mlps, range(layers)), seen)
    intact_final = seen["final"]
    no_change = max(
        float((final_state(model, chunk, make()) - intact).abs().max())
        for make in (lambda: patch_hooks(model, clean, every, ROLE["values"]), lambda: mlp_patches(model, clean_mlps))
    )
    if no_change > STATE_TOLERANCE:
        raise ValueError(f"patching values or MLPs from the clean run moved the final state by {no_change}")
    lines.append(f"patching later values or MLPs from the clean run moves the final state by at most {no_change:.1e}")

    layer, head = LAST_LAYER_HEAD
    out = layer_modules(model, layer)[2]
    output = {}
    capture = lambda key: out.register_forward_hook(lambda _m, _i, result: output.__setitem__(key, result.detach()))
    final_state(model, chunk, [capture("intact")])
    final_state(model, chunk, [capture("zeroed")] + zero_hooks(model, [layer * HEADS + head]))
    final_state(model, chunk, final_hooks(model, [layer * HEADS + head]), seen)
    gap = float(((intact_final - seen["final"]) - (output["intact"] - output["zeroed"])).abs().max())
    if gap > FINAL_TOLERANCE:
        raise ValueError(f"the subtracted output misses the head's output by {gap}")
    lines.append(f"on {layer}.{head}, what the final route subtracts matches the output map's change when the head is zeroed, within {gap:.1e}")

    streams = {}
    stream_hooks = input_stores(model, streams, range(layers))
    final_state(model, chunk, stream_hooks + final_hooks(model, picks), seen)
    if not all(torch.equal(streams[index], clean[index]) for index in range(layers)):
        raise ValueError("the final route changed what a layer reads")
    if torch.equal(seen["final"], intact_final):
        raise ValueError("the final route changed nothing at the final norm")
    lines.append("the final route leaves every layer's attention input exactly unchanged, and changes the final norm's input")
    routed = final_state(model, chunk, final_hooks(model, picks), seen)
    change = (intact_final - seen["final"]).norm(dim=-1).mean() / intact_final.norm(dim=-1).mean()
    right = chunk[:, REPEAT_LENGTH + 1 :, None]
    logit = lambda state: model.lm_head(state[:, REPEAT_LENGTH:-1])
    before, after = logit(intact), logit(routed)
    margin = lambda values: (values.gather(-1, right)[..., 0] - values.scatter(-1, right, -torch.inf).max(-1).values).mean()
    lines.append(f"on {len(chunk)} sequences, the picks' output is {float(change):.3f} of the final norm's input by size; subtracting it moves the right token's logit from {float(before.gather(-1, right).mean()):.2f} to {float(after.gather(-1, right).mean()):.2f}, and its lead over the best other token from {float(margin(before)):.2f} to {float(margin(after)):.2f}")

    for label, sources in (("picks", picks), ("first same-layer set", control)):
        attention, _, inputs, mlps = first_pass(model, chunk, sources)
        seen = {}
        silenced = final_state(model, chunk, zero_hooks(model, sources), seen)
        later = list(range((min(index // HEADS for index in sources) + 1) * HEADS, head_count(config)))
        rebuilt = final_state(model, chunk, route_hooks(model, ROUTES["all routes"], sources, later, inputs, mlps))
        miss = float((rebuilt - silenced).abs().max())
        if miss > STATE_TOLERANCE:
            raise ValueError(f"all routes together miss the silenced model by {miss} on {label}")
        agree = float((model.lm_head(rebuilt).argmax(-1) == model.lm_head(silenced).argmax(-1)).float().mean())
        lines.append(f"{label}: all routes together rebuild the silenced model's final state within {miss:.1e}, with {100 * agree:.1f}% of top tokens the same")

    target = (first + 1) * HEADS + 2
    qkv = layer_modules(model, first + 1)[1]
    seen = {}
    hooks = [qkv.register_forward_hook(lambda _m, _i, result: seen.__setitem__("intact", result))]
    hooks += patch_hooks(model, inputs, [target], ROLE["values"])
    hooks.append(qkv.register_forward_hook(lambda _m, _i, result: seen.__setitem__("patched", result)))
    final_state(model, chunk, hooks)
    changed = (seen["patched"] != seen["intact"]).any(0).any(0).nonzero().ravel().tolist()
    start = qkv_starts(config, target % HEADS)[2]
    if not changed or min(changed) < start or max(changed) >= start + HEAD_SIZE:
        raise ValueError(f"the value patch changed columns {changed[:5]}, expected {start} to {start + HEAD_SIZE - 1}")
    lines.append(f"patching {name(target, config)}'s values changes only its value columns")
    return lines


def main() -> None:
    began = time.time()
    small = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if DEVICE == "mps":
        torch.mps.set_per_process_memory_fraction(MPS_LIMIT_GB * 1e9 / torch.mps.recommended_max_memory())
    model = load(MODEL)
    config = model.config
    if (config.hidden_size, config.num_attention_heads, head_size(config)) != (WIDTH, HEADS, HEAD_SIZE):
        raise ValueError(f"unexpected shape {config.hidden_size}, {config.num_attention_heads}, {head_size(config)}")
    layers, total = config.num_hidden_layers, head_count(config)
    print(f"{MODEL}: {layers} layers, {HEADS} heads of size {HEAD_SIZE}, width {WIDTH}, forward passes on {DEVICE} in float32", flush=True)
    if small:
        print(f"DEBUG SAMPLE of {small} sequences: not a result")
        medium_ladder.BATCHES, medium_ladder.BATCH = 1, small

    worst = check_slices(model)
    if worst > SLICE_TOLERANCE:
        raise ValueError(f"slices do not match the model, error {worst}")
    print(f"  slices rebuild the model's attention and output in float32, worst error {worst:.2e}")
    tokens = torch.cat([repeated_tokens(ABLATION_SEED + 1 + batch, BATCH) for batch in range(BATCHES)])
    if small:
        tokens = tokens[:small]
    print("  " + check_hook(model, tokens, config))
    for line in check_patch(model, tokens[:CHUNK]):
        print("  " + line, flush=True)

    picks = [flat(head) for head in PICKS]
    rng = np.random.default_rng(CONTROL_PATH)
    controls = [same_layer_set(picks, HEADS, total, rng) for _ in range(CONTROL_SETS)]
    print(f"  picks: {' '.join(name(i, config) for i in picks)}")
    print("  same-layer sets: " + "; ".join(" ".join(name(i, config) for i in chosen) for chosen in controls), flush=True)
    sets = {"picks": picks, **{f"same-layer {i + 1}": chosen for i, chosen in enumerate(controls)}}
    later = {label: list(range((min(i // HEADS for i in sources) + 1) * HEADS, total)) for label, sources in sets.items()}

    probe = tokens[:CHUNK]
    _, _, inputs, mlps = first_pass(model, probe, picks)
    on_cpu = {label: forward(model, probe, route_hooks(model, ROUTES[label], picks, later["picks"], inputs, mlps))[1] for label in ("final residual", "later values", "all routes")}
    model.to(DEVICE)
    chunk = probe.to(DEVICE)
    _, _, inputs, mlps = first_pass(model, chunk, picks)
    gaps = {label: float(np.abs(right - forward(model, chunk, route_hooks(model, ROUTES[label], picks, later["picks"], inputs, mlps))[1]).max()) for label, right in on_cpu.items()}
    print(f"  cpu against {DEVICE} on {len(probe)} sequences, worst per-sequence accuracy gap: " + ", ".join(f"{label} {gap:.4f}" for label, gap in gaps.items()), flush=True)
    for line in check_routes(model, chunk, controls[0]):
        print("  " + line, flush=True)
    del inputs, mlps

    attention_by = {(label, route): [] for label in sets for route in ROUTES}
    right_by = {(label, route): [] for label in sets for route in ROUTES}
    base_attention, base_right = [], []
    for step, chunk in enumerate(tokens.split(CHUNK)):
        chunk = chunk.to(DEVICE)
        intact = {}
        attention, right = forward(model, chunk, input_stores(model, intact, range(layers)))
        base_attention.append(attention)
        base_right.append(right)
        for label, sources in sets.items():
            attention, right, inputs, mlps = first_pass(model, chunk, sources)
            attention_by[label, "silenced everywhere"].append(attention)
            right_by[label, "silenced everywhere"].append(right)
            for route, parts in ROUTES.items():
                if parts is None:
                    continue
                attention, right = forward(model, chunk, route_hooks(model, parts, sources, later[label], inputs, mlps, intact))
                attention_by[label, route].append(attention)
                right_by[label, route].append(right)
            del inputs, mlps
        if (step + 1) % (BATCH // CHUNK) == 0:
            print(f"  {(step + 1) * CHUNK} sequences done, {time.time() - began:.0f} s", flush=True)

    base, base_right = np.concatenate(base_attention), np.concatenate(base_right)
    base_heads = base.mean(0)
    print(f"\n  intact on {len(tokens)} sequences: repeat accuracy {base_right.mean():.4f} (saved {SAVED_INTACT:.4f}), late heads' mean attention {base.mean():.3f}; late heads are {' '.join(name(flat(head), config) for head in LATE)}")
    results = {kind: row(np.concatenate(attention_by[kind]), np.concatenate(right_by[kind]), base, base_right) for kind in attention_by}

    print("\n  the picks through each route; cost is points of repeat accuracy, intact minus intervention; intervals resample sequences")
    for route in ROUTES:
        print(show(route, results["picks", route], base_heads, config))

    labels = [label for label in sets if label.startswith("same-layer")]
    print(f"\n  {CONTROL_SETS} same-layer sets of the picks through each route, fixed before the run")
    for route in ROUTES:
        costs = [results[label, route]["cost"] for label in labels]
        print(f"  {route}: cost by set " + " ".join(f"{value:.1f}" for value in costs) + f"; mean {np.mean(costs):.1f}, lowest {min(costs):.1f}, highest {max(costs):.1f}; picks minus mean {results['picks', route]['cost'] - np.mean(costs):.1f}")

    print("\n  reproducing medium_ladder.py: " + ", ".join(f"picks, {route} {results['picks', route]['cost']:.1f} (saved {saved})" for route, saved in SAVED_COSTS.items())
          + f"; same-layer sets silenced everywhere mean {np.mean([results[label, 'silenced everywhere']['cost'] for label in labels]):.1f} (saved {SAVED_SAME_LAYER[0]})"
          + f", lowest {min(results[label, 'silenced everywhere']['cost'] for label in labels):.1f} (saved {SAVED_SAME_LAYER[1]})"
          + f", highest {max(results[label, 'silenced everywhere']['cost'] for label in labels):.1f} (saved {SAVED_SAME_LAYER[2]})")
    closure = float(np.abs(np.concatenate(right_by["picks", "all routes"]) - np.concatenate(right_by["picks", "silenced everywhere"])).max())
    print(f"  all routes together against silenced everywhere, worst per-sequence accuracy gap {closure:.4f}")

    final, values = results["picks", "final residual"]["cost"], results["picks", "later values"]["cost"]
    tests = {
        f"silencing the picks only in the final residual costs at least {FINAL_AT_LEAST:.0f} points ({final:.1f})": final >= FINAL_AT_LEAST,
        f"silencing them only for later heads' values costs under {VALUES_UNDER:.0f} points ({values:.1f})": values < VALUES_UNDER,
    }
    print("\n  card's prediction:")
    for text, held in tests.items():
        print(f"    {text}: {'held' if held else 'failed'}")
    print(f"  both: {'held' if all(tests.values()) else 'failed'}")
    print(f"\n  took {time.time() - began:.0f} s, peak resident memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB, peak MPS driver memory {medium_ladder.peak_driver / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
