"""Tests whether summing a head's cosine over later readers finds previous-token heads from blocks alone. Makes no Astra calls.

For gpt2, gpt2-medium and distilgpt2, each head's statistic is the sum of cosine[reader, head] over readers in later
layers, where cosine[i, j] has reader i's QK^T and writer j's VO. Last-layer heads have no later reader and are not ranked.
Previous-token heads are measured by running each model on repeated random tokens, as in other_models.py. Controls are a
within-layer shuffle of the statistic, the roles swapped (each head's sum as a reader of earlier writers), and the same
sum over earlier readers, the direction other_models.py calls swapped.
"""
import gc
import resource

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from transformers.utils import cached_file

from checks.other_models import (
    FLOAT32_FORWARD,
    check_slices,
    head_count,
    head_size,
    input_by_output,
    layer_modules,
    measured_scores,
    name,
    qkv_starts,
)

MODELS = ("gpt2", "gpt2-medium", "distilbert/distilgpt2")
# The top previous-token heads the card rests on, from checks/outputs/other_models.txt.
EXPECTED_PREVIOUS = {
    "gpt2": ["4.11", "3.7", "2.2", "3.2", "2.9"],
    "gpt2-medium": ["5.11", "4.13", "6.7", "8.1", "13.5"],
    "distilbert/distilgpt2": ["2.11", "1.2", "1.9", "3.0", "2.3"],
}
PREDICTED_HITS_ON = ("gpt2", "gpt2-medium")
PREDICTED_HITS = 2
TOP = 5
TOP_RANK = 10
DRAWS = 2000
SEED = 20261104
SLICE_TOLERANCE = 1e-2


def load(model_name: str) -> torch.nn.Module:
    """Loads a model for forward passes: gpt2-medium through float64 and back to avoid NaN logits, the others in float64.

    Medium's weights are read one tensor at a time, because converting them through from_pretrained's single mapping of
    the file keeps every page resident next to the converted copy, 3.4 GB at peak.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, dtype=torch.float32).eval()
    model.set_attn_implementation("eager")
    if model_name in FLOAT32_FORWARD:
        path = cached_file(model_name, "model.safetensors")
        for key, parameter in model.named_parameters():
            with safe_open(path, framework="pt") as weights:
                raw = weights.get_tensor(key.removeprefix("transformer."))
            parameter.data = raw.double().float()
    else:
        model.double()
    gc.collect()
    return model


def cosines(model) -> np.ndarray:
    """Returns other_models.cosines, reader rows and writer columns, filling arrays in place so medium stays under 3 GB."""
    config, size = model.config, head_size(model.config)
    total, width = head_count(config), config.hidden_size
    q, k, v = (np.empty((total, width, size)) for _ in range(3))
    o = np.empty((total, size, width))
    for layer in range(config.num_hidden_layers):
        norm, qkv, out = layer_modules(model, layer)
        scale = norm.weight.detach().double().numpy()[:, None]
        weight, output = (input_by_output(module).detach().double().numpy() for module in (qkv, out))
        for head in range(config.num_attention_heads):
            index = layer * config.num_attention_heads + head
            for role, start in zip((q, k, v), qkv_starts(config, head)):
                part = scale * weight[:, start : start + size]
                role[index] = part - part.mean(0)
            part = output[head * size : (head + 1) * size]
            o[index] = part - part.mean(1, keepdims=True)
    v_q = v.transpose(0, 2, 1)
    reader_norms = np.sqrt(((q.transpose(0, 2, 1) @ q) * (k.transpose(0, 2, 1) @ k)).sum((1, 2)))
    writer_norms = np.sqrt(((v_q @ v) * (o @ o.transpose(0, 2, 1))).sum((1, 2)))
    cosine = np.zeros((total, total))
    for index in range(total):
        cosine[index] = ((v_q @ q[index]) * (o @ k[index])).sum((1, 2)) / (reader_norms[index] * writer_norms)
    return cosine


def statistics(cosine: np.ndarray, layers: np.ndarray) -> dict[str, np.ndarray]:
    """Returns each statistic per head, at minus infinity for a head with nothing to sum over."""
    later_reader = layers[:, None] > layers[None, :]
    earlier_reader = layers[:, None] < layers[None, :]

    def scored(values: np.ndarray, has_partners: np.ndarray) -> np.ndarray:
        return np.where(has_partners, values, -np.inf)

    return {
        "sum as writer for later heads": scored((cosine * later_reader).sum(0), later_reader.any(0)),
        "best as writer for a later head": np.where(later_reader, cosine, -np.inf).max(0),
        "roles swapped, sum as reader of earlier heads": scored((cosine * later_reader).sum(1), later_reader.any(1)),
        "direction swapped, sum as writer for earlier heads": scored((cosine * earlier_reader).sum(0), earlier_reader.any(0)),
    }


def shuffle_chance(statistic: np.ndarray, layers: np.ndarray, top_previous: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns the top-5 hits and the top head's rank, for each of the within-layer shuffles of a statistic."""
    hits, ranks = np.zeros(DRAWS, dtype=int), np.zeros(DRAWS, dtype=int)
    groups = [np.flatnonzero(layers == layer) for layer in np.unique(layers)]
    for draw in range(DRAWS):
        shuffled = statistic.copy()
        for group in groups:
            shuffled[group] = rng.permutation(statistic[group])
        order = np.argsort(-shuffled, kind="stable")
        hits[draw] = len(set(order[:TOP].tolist()) & set(top_previous.tolist()))
        ranks[draw] = int(np.flatnonzero(order == top_previous[0])[0]) + 1
    return hits, ranks


def study(model_name: str, rng: np.random.Generator) -> dict:
    model = load(model_name)
    config = model.config
    total = head_count(config)
    layers = np.arange(total) // config.num_attention_heads
    print(f"\n{model_name}: {config.num_hidden_layers} layers, {config.num_attention_heads} heads, {total} heads in all, forward passes in {next(model.parameters()).dtype}")

    worst = check_slices(model)
    print(f"  slices rebuild the model's attention and output, worst error {worst:.2e}")
    if not worst < SLICE_TOLERANCE:
        raise ValueError(f"{model_name}: slices do not match the model, error {worst}")

    cosine = cosines(model)
    gc.collect()
    _, previous = measured_scores(model)
    if not np.isfinite(previous).all():
        raise ValueError(f"{model_name}: attention is not finite")
    del model
    gc.collect()

    previous_order = np.argsort(-previous)
    previous_rank = np.argsort(previous_order) + 1
    top_previous = previous_order[:TOP]
    measured_names = [name(int(i), config) for i in top_previous]
    print("  top heads on the previous token: " + ", ".join(f"{name(int(i), config)} {previous[i]:.3f}" for i in previous_order[:TOP_RANK]))
    print(f"  top 5 match the card's list from other_models.txt: {measured_names == EXPECTED_PREVIOUS[model_name]}")

    results = {}
    for label, statistic in statistics(cosine, layers).items():
        order = np.argsort(-statistic, kind="stable")
        ranked = int(np.isfinite(statistic).sum())
        hits = sorted(set(order[:TOP].tolist()) & set(top_previous.tolist()))
        hits_ten = len(set(order[:TOP_RANK].tolist()) & set(top_previous.tolist()))
        top_rank = int(np.flatnonzero(order == top_previous[0])[0]) + 1
        results[label] = {"hits": len(hits), "rank": top_rank}
        print(f"  {label} ({ranked} heads ranked): {len(hits)} of {TOP} in its top {TOP} [{', '.join(name(int(i), config) for i in hits)}], "
              f"{hits_ten} of {TOP} in its top {TOP_RANK}, top previous-token head {measured_names[0]} ranks {top_rank}")
        print("    its top 10, with value and rank on the previous token: " + ", ".join(
            f"{name(int(i), config)} {statistic[i]:.2f} (prev {previous[i]:.3f}, rank {previous_rank[i]})" for i in order[:TOP_RANK]))

    statistic = statistics(cosine, layers)["sum as writer for later heads"]
    print("  mean sum as writer for later heads, by layer: " + ", ".join(
        f"{layer} {statistic[layers == layer].mean():.2f}" for layer in range(config.num_hidden_layers - 1)))
    hits_drawn, ranks_drawn = shuffle_chance(statistic, layers, top_previous, rng)
    observed = results["sum as writer for later heads"]
    print(f"  within-layer shuffle, {DRAWS} draws: mean {hits_drawn.mean():.3f} of {TOP} in the top {TOP}, chance of {observed['hits']} or more {np.mean(hits_drawn >= observed['hits']):.4f}, "
          f"chance the top head ranks in the top {TOP_RANK} {np.mean(ranks_drawn <= TOP_RANK):.4f}, median rank {np.median(ranks_drawn):.0f}")
    return results


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = {model_name: study(model_name, rng) for model_name in MODELS}
    label, best = "sum as writer for later heads", "best as writer for a later head"
    print("\nmodel | sum hits | best hits | roles swapped hits | direction swapped hits | top head rank by sum | by best")
    for model_name, row in rows.items():
        print(f"{model_name} | {row[label]['hits']} | {row[best]['hits']} | {row['roles swapped, sum as reader of earlier heads']['hits']} | "
              f"{row['direction swapped, sum as writer for earlier heads']['hits']} | {row[label]['rank']} | {row[best]['rank']}")
    enough = all(rows[model_name][label]["hits"] >= PREDICTED_HITS for model_name in PREDICTED_HITS_ON)
    close = all(row[label]["rank"] <= TOP_RANK for row in rows.values())
    print(f"\nprediction: at least {PREDICTED_HITS} of {TOP} on gpt2 and medium: {enough}; top head within the top {TOP_RANK} on all three: {close}")
    print(f"peak resident memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
