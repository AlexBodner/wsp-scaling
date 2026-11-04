"""Tests whether a copy score in token space finds negative heads on gpt2-medium and distilgpt2. Makes no Astra calls.

Each head's score is a token's own logit, through its embedding, block 0's MLP, the layer norm's mean and scale removal, V, O,
the final scale and the centred unembedding, averaged over 8000 tokens and divided by VO's norm, as in
a local verifier run that isn't public. The behaviour is each head's direct effect on the logit of the token it attends to
most, on the second copy of repeated random tokens and, as a second reading off the prediction, on stories. GPT-2 small is run
last and has to reproduce the saved numbers. Controls are a within-layer shuffle of the score, and the same score with the embedding
rows permuted relative to the unembedding. Pass a sequence count to debug on a small sample: it skips the reproduction checks.
Weights load one tensor at a time and go through float64 and back: zero_order.py's load keeps every mapped page resident, 3.4 GB.
"""
import gc
import glob
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from scipy.stats import rankdata, spearmanr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from checks.other_models import check_slices, head_count, head_size, input_by_output, layer_modules, name, qkv_starts

MODELS = ("gpt2-medium", "distilbert/distilgpt2", "gpt2")
PREDICTED = ("gpt2-medium", "distilbert/distilgpt2")
SEED = 20261104
SHUFFLE_SEED = 20261102
TOKEN_SEED = 1
VOCABULARY = 50257
REPEAT_LENGTH = 25
BATCH = 32
CHUNK = 4
STORIES = 128
STORY_TOKENS = 64
TEXT = Path("data/tinystories/valid.txt")
SEPARATOR = "<|endoftext|>"
TOKEN_RANGE = (1000, 50000)
TOKENS = 8000
TOP_SCORE = 3
TOP_EFFECT = 10
SHOWN = 10
PREDICTED_SPEARMAN = 0.5
DRAWS = 2000
PERMUTATIONS = 20
SLICE_TOLERANCE = 1e-2
LIMIT_GB = 3.4
# GPT-2 small from a local verifier run that isn't public.
SAVED = {"10.7": (-1.745, -1.011, -0.298), "11.10": (-1.184, -0.568, -0.168)}
SAVED_SPEARMAN = {"repeats": 0.795, "stories": 0.434}
SAVED_TOLERANCE = 0.0015


def check_memory(where: str) -> float:
    used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    if used > LIMIT_GB:
        raise MemoryError(f"over {LIMIT_GB} GB at {where}: {used:.2f}")
    return used


@torch.no_grad()
def load(model_name: str) -> torch.nn.Module:
    """Builds the model in float32, and fills it one tensor at a time, each through float64 and back, unmapping the file after each."""
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_name), dtype=torch.float32).eval()
    model.set_attn_implementation("eager")
    path = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_name.replace('/', '--')}/snapshots/*/model.safetensors"))[0]
    with safe_open(path, "pt") as handle:
        stored = set(handle.keys())
    for parameter_name, parameter in model.named_parameters():
        key = parameter_name if parameter_name in stored else parameter_name.removeprefix("transformer.")
        with safe_open(path, "pt") as handle:
            parameter.copy_(handle.get_tensor(key).double().float())
    gc.collect()
    return model


@torch.no_grad()
def token_tables(model) -> tuple[np.ndarray, np.ndarray]:
    """Returns the normalised embeddings through block 0's MLP, and the scaled centred unembedding, on the same tokens, in float64."""
    base = model.base_model
    block = base.h[0]
    tokens = torch.from_numpy(np.random.default_rng(TOKEN_SEED).choice(np.arange(*TOKEN_RANGE), TOKENS, replace=False))
    embedded = base.wte.weight[tokens]
    through = embedded + block.mlp(block.ln_2(embedded))
    through = through - through.mean(-1, keepdim=True)
    through = through / through.std(-1, keepdim=True)
    unembed = model.lm_head.weight
    target = (unembed - unembed.mean(0))[tokens] * base.ln_f.weight
    return through.double().numpy(), target.double().numpy()


def value_output(model):
    """Yields each head's V and O in float64, with the layer norm's scale and mean removal included, without holding them all."""
    config, size = model.config, head_size(model.config)
    for layer in range(config.num_hidden_layers):
        norm, qkv, out = layer_modules(model, layer)
        scale = norm.weight.detach().double().numpy()[:, None]
        weight, output = (input_by_output(module).detach().double().numpy() for module in (qkv, out))
        for head in range(config.num_attention_heads):
            start = qkv_starts(config, head)[2]
            v = scale * weight[:, start : start + size]
            o = output[head * size : (head + 1) * size]
            yield v - v.mean(0), o - o.mean(1, keepdims=True)


def copy_scores(model, table: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Returns each head's mean over tokens of table V O target, over VO's norm: the trace of O (target^T table / N) V."""
    between = target.T @ table / len(table)
    return np.array([np.einsum("ij,ji->", o @ between, v) / np.sqrt(((v.T @ v) * (o @ o.T)).sum()) for v, o in value_output(model)])


@torch.no_grad()
def attended_effect(model, tokens: torch.Tensor, positions: torch.Tensor) -> dict[str, np.ndarray]:
    """Returns each head's direct effect on the logit of the token it attends to most, and its largest attention and share on itself."""
    config = model.config
    per_layer, size, layers = config.num_attention_heads, head_size(config), config.num_hidden_layers
    base = model.base_model
    unembed, scale = model.lm_head.weight, base.ln_f.weight
    centre = unembed.mean(0)
    effect, largest, itself = (np.zeros((layers, per_layer)) for _ in range(3))
    for chunk in tokens.split(CHUNK):
        attended, final, hooks = {}, {}, []
        for layer in range(layers):
            def store(_module, inputs, layer=layer):
                attended[layer] = inputs[0][:, positions].detach()

            hooks.append(layer_modules(model, layer)[2].register_forward_pre_hook(store))
        hooks.append(base.ln_f.register_forward_pre_hook(lambda _m, inputs: final.__setitem__("x", inputs[0][:, positions].detach())))
        attentions = base(chunk, output_attentions=True, use_cache=False).attentions
        for hook in hooks:
            hook.remove()
        deviation = final["x"].std(-1, keepdim=True)
        rows = torch.arange(len(chunk))[:, None, None]
        for layer in range(layers):
            peak, picked = attentions[layer][:, :, positions].max(-1)
            direction = unembed[chunk[rows, picked]] - centre
            out = layer_modules(model, layer)[2].weight
            for head in range(per_layer):
                part = head * size, (head + 1) * size
                written = attended[layer][..., part[0] : part[1]] @ out[part[0] : part[1]] / deviation * scale
                effect[layer, head] += float((written * direction[:, head]).sum())
            largest[layer] += peak.sum((0, 2)).double().numpy()
            itself[layer] += (picked == positions[None, None]).sum((0, 2)).double().numpy()
        del attended, final, attentions
        check_memory("direct effect")
    count = len(tokens) * len(positions)
    return {"effect": effect.ravel() / count, "largest": largest.ravel() / count, "itself": itself.ravel() / count}


def repeat_tokens() -> torch.Tensor:
    """Returns the repeats signed_ladder.py uses on the shuffled model, mapped back to the original token numbers, as the local verifier run does."""
    half = torch.randint(0, VOCABULARY, (BATCH, REPEAT_LENGTH), generator=torch.Generator().manual_seed(SEED))
    to_original = np.argsort(np.random.default_rng(SHUFFLE_SEED).permutation(VOCABULARY))
    return torch.from_numpy(to_original[torch.cat([half, half], 1).numpy()])


def story_tokens() -> torch.Tensor:
    """Returns the opening tokens of random whole stories, leaving out the fragment the file starts with."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    stories = TEXT.read_text().split(SEPARATOR)[1:]
    chosen = []
    for index in np.random.default_rng(SEED).permutation(len(stories)):
        tokens = tokenizer(stories[index].strip(), add_special_tokens=False).input_ids
        if len(tokens) >= STORY_TOKENS:
            chosen.append(tokens[:STORY_TOKENS])
        if len(chosen) == STORIES:
            return torch.tensor(chosen)
    raise ValueError("too few long stories")


def top_test(score: np.ndarray, effect: np.ndarray) -> tuple[bool, int]:
    """Returns whether the score's most negative heads are all among the most negative effects, and how many are."""
    lowest = set(np.argsort(effect)[:TOP_EFFECT].tolist())
    found = sum(int(head) in lowest for head in np.argsort(score)[:TOP_SCORE])
    return found == TOP_SCORE, found


def within_layer_chance(score: np.ndarray, effect: np.ndarray, per_layer: int, rng: np.random.Generator) -> dict:
    """Shuffles the score within layers, and returns each draw's Spearman with the effect and whether the top test passes."""
    layers = len(score) // per_layer
    effect_ranks = rankdata(effect)
    spearman, passed = np.zeros(DRAWS), np.zeros(DRAWS, dtype=bool)
    for draw in range(DRAWS):
        permutation = np.concatenate([layer * per_layer + rng.permutation(per_layer) for layer in range(layers)])
        spearman[draw] = np.corrcoef(rankdata(score[permutation]), effect_ranks)[0, 1]
        passed[draw] = top_test(score[permutation], effect)[0]
    return {"spearman": spearman, "passed": passed}


def within_layer_spearman(score: np.ndarray, effect: np.ndarray, per_layer: int) -> float:
    """Returns the Spearman after taking out each layer's mean from both, so layer depth alone can't produce it."""
    centred = lambda values: (values.reshape(-1, per_layer) - values.reshape(-1, per_layer).mean(1, keepdims=True)).ravel()
    return float(spearmanr(centred(score), centred(effect)).statistic)


def study(model_name: str, repeats: torch.Tensor, stories: torch.Tensor, small: bool) -> dict:
    began = time.time()
    model = load(model_name)
    config = model.config
    per_layer, total = config.num_attention_heads, head_count(config)
    print(f"\n{model_name}: {config.num_hidden_layers} layers, {per_layer} heads of size {head_size(config)}, width {config.hidden_size}, {total} heads, on cpu in {next(model.parameters()).dtype}", flush=True)
    if not torch.equal(model.lm_head.weight, model.base_model.wte.weight):
        raise ValueError("the unembedding is not the embedding")
    worst = check_slices(model)
    print(f"  slices rebuild the model's attention and output, worst error {worst:.2e}")
    if not worst < SLICE_TOLERANCE:
        raise ValueError(f"slices do not match the model, error {worst}")

    table, target = token_tables(model)
    score = copy_scores(model, table, target)
    spot = total // 2
    v, o = next(part for index, part in enumerate(value_output(model)) if index == spot)
    direct = ((table[:200] @ v) @ o * target[:200]).sum(1).mean() / np.linalg.norm(v @ o)
    subset = copy_scores(model, table[:200], target[:200])[spot]
    if abs(direct - subset) > 1e-8:
        raise ValueError(f"the trace form of the score misses the direct product: {direct} against {subset}")
    print(f"  trace form of the score matches the direct product on {name(spot, config)} on 200 tokens within {abs(direct - subset):.1e}; {time.time() - began:.0f} s", flush=True)

    positions = torch.arange(REPEAT_LENGTH, 2 * REPEAT_LENGTH - 1)
    measured = {"repeats": attended_effect(model, repeats, positions), "stories": attended_effect(model, stories, torch.arange(1, stories.shape[1]))}
    print(f"  direct effects measured on {len(repeats)} repeats, {len(positions)} positions, and {len(stories)} stories, {stories.shape[1] - 1} positions; {time.time() - began:.0f} s", flush=True)

    order = np.argsort(score)
    rank_by = {label: np.argsort(np.argsort(values["effect"])) + 1 for label, values in measured.items()}
    print(f"  most negative by score, with score and on repeats / stories: direct effect (effect rank from the bottom), largest attention, share of positions on itself")
    for head in order[:SHOWN]:
        rep, sto = measured["repeats"], measured["stories"]
        print(f"    {name(int(head), config)}: score {score[head]:.3f}, effect {rep['effect'][head]:.3f} ({rank_by['repeats'][head]})/{sto['effect'][head]:.3f} ({rank_by['stories'][head]}), attention {rep['largest'][head]:.2f}/{sto['largest'][head]:.2f}, on itself {rep['itself'][head]:.2f}/{sto['itself'][head]:.2f}")
    score_rank = np.argsort(order) + 1
    for label, values in measured.items():
        lowest = np.argsort(values["effect"])[:SHOWN]
        print(f"  most negative effects on {label}: " + ", ".join(f"{name(int(head), config)} {values['effect'][head]:.3f} (score rank {score_rank[head]})" for head in lowest))

    rng = np.random.default_rng((SEED, total))
    results = {"score": score, "measured": measured, "config": config}
    print(f"  Spearman between score and direct effect on all {total} heads, with {DRAWS} within-layer shuffles of the score")
    for label, values in measured.items():
        observed = float(spearmanr(score, values["effect"]).statistic)
        chance = within_layer_chance(score, values["effect"], per_layer, rng)
        held, found = top_test(score, values["effect"])
        low, high = np.percentile(chance["spearman"], [2.5, 97.5])
        print(f"    {label}: Spearman {observed:.3f}, shuffles mean {chance['spearman'].mean():.3f}, 2.5th to 97.5th percentile {low:.3f} to {high:.3f}, chance of this or higher {np.mean(chance['spearman'] >= observed):.4f}; "
              f"within-layer Spearman {within_layer_spearman(score, values['effect'], per_layer):.3f}; top {TOP_SCORE} by score among {TOP_EFFECT} most negative effects: {found} of {TOP_SCORE}, shuffles passing {np.mean(chance['passed']):.4f}")
        results[label] = {"spearman": observed, "found": found, "held": held}

    permuted = np.zeros((PERMUTATIONS, 2))
    tops = []
    for draw in range(PERMUTATIONS):
        shuffled = copy_scores(model, table, target[rng.permutation(len(target))])
        permuted[draw] = spearmanr(shuffled, measured["repeats"]["effect"]).statistic, top_test(shuffled, measured["repeats"]["effect"])[1]
        tops.append(" ".join(name(int(head), config) for head in np.argsort(shuffled)[:TOP_SCORE]))
        check_memory("permuted score")
    print(f"  {PERMUTATIONS} permutations of the embedding rows against the unembedding, on repeats: Spearman mean {permuted[:, 0].mean():.3f}, lowest {permuted[:, 0].min():.3f}, highest {permuted[:, 0].max():.3f}; "
          f"top {TOP_SCORE} among the {TOP_EFFECT} most negative effects: at most {int(permuted[:, 1].max())} of {TOP_SCORE}, {int((permuted[:, 1] == TOP_SCORE).sum())} permutations pass; first draw's top {TOP_SCORE}: {tops[0]}; last draw's: {tops[-1]}")
    print(f"  took {time.time() - began:.0f} s, peak resident memory {check_memory('end'):.2f} GB", flush=True)
    if model_name == "gpt2" and not small:
        reproduce(score, measured, config)
    del model, table, target
    gc.collect()
    return results


def reproduce(score: np.ndarray, measured: dict, config) -> None:
    """Checks GPT-2 small against the saved off-card numbers, and stops the run if it misses."""
    index = {label: int(label.split(".")[0]) * config.num_attention_heads + int(label.split(".")[1]) for label in SAVED}
    order = np.argsort(score)
    lines = []
    for label, (saved_score, saved_repeats, saved_stories) in SAVED.items():
        now = score[index[label]], measured["repeats"]["effect"][index[label]], measured["stories"]["effect"][index[label]]
        lines.append(f"{label} score {now[0]:.3f} (saved {saved_score}), effect {now[1]:.3f}/{now[2]:.3f} (saved {saved_repeats}/{saved_stories})")
        if max(abs(now[0] - saved_score), abs(now[1] - saved_repeats), abs(now[2] - saved_stories)) > SAVED_TOLERANCE:
            raise ValueError(f"{lines[-1]} does not reproduce")
    spearman = {label: float(spearmanr(score, measured[label]["effect"]).statistic) for label in SAVED_SPEARMAN}
    lines.append(", ".join(f"Spearman {label} {spearman[label]:.3f} (saved {SAVED_SPEARMAN[label]})" for label in SAVED_SPEARMAN))
    if any(abs(spearman[label] - SAVED_SPEARMAN[label]) > SAVED_TOLERANCE for label in SAVED_SPEARMAN):
        raise ValueError(f"{lines[-1]} does not reproduce")
    if [name(int(head), config) for head in order[:2]] != list(SAVED):
        raise ValueError("10.7 and 11.10 are not the two most negative heads")
    print("  reproduces the local verifier run: " + "; ".join(lines), flush=True)


def main() -> None:
    small = int(sys.argv[1]) if len(sys.argv) > 1 else None
    repeats, stories = repeat_tokens(), story_tokens()
    if small:
        repeats, stories = repeats[:small], stories[:small]
        print(f"DEBUG SAMPLE of {small} sequences: not a result")
    results = {model_name: study(model_name, repeats, stories, bool(small)) for model_name in MODELS}

    print("\ncard's prediction, on repeats:")
    tests = {}
    for model_name in PREDICTED:
        result = results[model_name]["repeats"]
        tests[f"{model_name}: the {TOP_SCORE} most negative by score are all among the {TOP_EFFECT} most negative direct effects ({result['found']} of {TOP_SCORE})"] = result["held"]
        tests[f"{model_name}: Spearman with direct effect at least {PREDICTED_SPEARMAN} ({result['spearman']:.3f})"] = result["spearman"] >= PREDICTED_SPEARMAN
    for text, held in tests.items():
        print(f"  {text}: {'held' if held else 'failed'}")
    print(f"all four: {'held' if all(tests.values()) else 'failed'}")


if __name__ == "__main__":
    main()
