"""Tests whether the cosine statistic finds induction heads on other models and sizes. Makes no Astra calls.

For each model this measures each head's attention to the token after the earlier copy on repeated random tokens,
then how many of the top 5 the statistic finds, against chance from the head count and against the statistic with
earlier and later roles swapped, then how far zeroing the top 5 cuts repeat accuracy. Token IDs are not shuffled here.
"""
import gc
import time
from math import comb

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

MODELS = ("gpt2", "gpt2-medium", "distilbert/distilgpt2", "EleutherAI/pythia-14m")
TRUTH_SEEDS = (0, 1, 2)
TRUTH_BATCH = 32
REPEAT_LENGTH = 50
TOKEN_RANGE = (1000, 50000)
CHUNK = 8
ABLATION_BATCH = 64
ABLATION_SEED = 20261103
RANDOM_SETS = 8
TOP = 5
SHOWN = 10
SLICE_TOLERANCE = 1e-4
FLOAT32_FORWARD = ("gpt2-medium",)


def head_size(config) -> int:
    return config.hidden_size // config.num_attention_heads


def head_count(config) -> int:
    return config.num_hidden_layers * config.num_attention_heads


def name(index: int, config) -> str:
    return f"{index // config.num_attention_heads}.{index % config.num_attention_heads}"


def qkv_starts(config, head: int) -> tuple[int, int, int]:
    """Returns where a head's query, key and value columns start: GPT-2 groups by role, GPTNeoX by head."""
    size = head_size(config)
    if config.model_type == "gpt_neox":
        return tuple(3 * size * head + role * size for role in range(3))
    return tuple(role * config.hidden_size + head * size for role in range(3))


def layer_modules(model, layer: int) -> tuple:
    """Returns a layer's input norm, query-key-value map and output map."""
    block = model.base_model.layers[layer] if model.config.model_type == "gpt_neox" else model.base_model.h[layer]
    if model.config.model_type == "gpt_neox":
        return block.input_layernorm, block.attention.query_key_value, block.attention.dense
    return block.ln_1, block.attn.c_attn, block.attn.c_proj


def input_by_output(module) -> torch.Tensor:
    """Returns a map's weight as input x output: Conv1D already is, Linear is the other way round."""
    return module.weight.T if isinstance(module, torch.nn.Linear) else module.weight


def head_parts(model) -> list[tuple[np.ndarray, ...]]:
    """Returns each head's Q, K, V and O in float64, with the layer norm's scale and mean removal included."""
    config, size = model.config, head_size(model.config)
    parts = []
    for layer in range(config.num_hidden_layers):
        norm, qkv, out = layer_modules(model, layer)
        scale = norm.weight.detach().double().numpy()[:, None]
        weight, output = (input_by_output(module).detach().double().numpy() for module in (qkv, out))
        for head in range(config.num_attention_heads):
            q, k, v = (scale * weight[:, start : start + size] for start in qkv_starts(config, head))
            o = output[head * size : (head + 1) * size]
            parts.append((q - q.mean(0), k - k.mean(0), v - v.mean(0), o - o.mean(1, keepdims=True)))
    return parts


def cosines(model) -> np.ndarray:
    """Returns the cosine of every head's QK^T (rows) with every head's VO (columns), without forming the large products."""
    q, k, v, o = (np.stack(role) for role in zip(*head_parts(model)))
    v_q = v.transpose(0, 2, 1)
    query_key_norms = np.sqrt(((q.transpose(0, 2, 1) @ q) * (k.transpose(0, 2, 1) @ k)).sum((1, 2)))
    value_output_norms = np.sqrt(((v_q @ v) * (o @ o.transpose(0, 2, 1))).sum((1, 2)))
    cosine = np.zeros((len(q), len(q)))
    for index in range(len(q)):
        cosine[index] = ((v_q @ q[index]) * (o @ k[index])).sum((1, 2)) / (query_key_norms[index] * value_output_norms)
    return cosine


def statistic_picks(cosine: np.ndarray, allowed: np.ndarray) -> dict:
    """Returns the source, with its votes, and the top heads by that source and by each head's best pairing."""
    masked = np.where(allowed, cosine, -np.inf)
    votes = np.bincount(np.argmax(masked[allowed.any(1)], axis=1), minlength=len(cosine))
    source = int(np.argmax(votes))
    return {
        "source": source,
        "votes": int(votes[source]),
        "voters": int(allowed.any(1).sum()),
        "by_source": np.argsort(-masked[:, source])[:TOP].tolist(),
        "best_pairing": np.argsort(-masked.max(axis=1))[:TOP].tolist(),
    }


@torch.no_grad()
def check_slices(model) -> float:
    """Rebuilds every head's attention and each layer's output from the slices, and returns the worst error against the model."""
    config, size = model.config, head_size(model.config)
    length = 20
    tokens = torch.randint(*TOKEN_RANGE, (2, length), generator=torch.Generator().manual_seed(0))
    seen, hooks = {}, []
    for layer in range(config.num_hidden_layers):
        _, qkv, out = layer_modules(model, layer)
        hooks.append(qkv.register_forward_hook(lambda _m, inputs, _o, layer=layer: seen.__setitem__(("in", layer), inputs[0])))
        hooks.append(out.register_forward_hook(lambda _m, _i, output, layer=layer: seen.__setitem__(("out", layer), output)))
    attentions = model.base_model(tokens, output_attentions=True).attentions
    for hook in hooks:
        hook.remove()
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool))
    worst = 0.0
    for layer in range(config.num_hidden_layers):
        _, qkv, out = layer_modules(model, layer)
        projected = seen["in", layer] @ input_by_output(qkv) + qkv.bias
        output = out.bias.expand_as(seen["out", layer]).clone()
        for head in range(config.num_attention_heads):
            q, k, v = (projected[..., start : start + size].unsqueeze(1) for start in qkv_starts(config, head))
            if config.model_type == "gpt_neox":
                position = torch.arange(length)[None]
                q, k = apply_rotary_pos_emb(q, k, *model.base_model.rotary_emb(q[:, 0], position_ids=position))
            logits = (q @ k.transpose(-1, -2) / size**0.5).masked_fill(~causal, -torch.inf)
            # GPTNeoX's own eager attention takes its softmax in float32.
            attention = logits.softmax(-1, dtype=torch.float32 if config.model_type == "gpt_neox" else None).to(q.dtype)
            worst = max(worst, float((attention[:, 0] - attentions[layer][:, head]).abs().max()))
            output = output + (attention @ v)[:, 0] @ input_by_output(out)[head * size : (head + 1) * size]
        worst = max(worst, float((output - seen["out", layer]).abs().max()))
    return worst


@torch.no_grad()
def measured_scores(model) -> tuple[np.ndarray, np.ndarray]:
    """Returns each head's attention to the token after the earlier copy, and to the previous token in the first copy."""
    config = model.config
    induction = np.zeros((config.num_hidden_layers, config.num_attention_heads))
    previous = np.zeros_like(induction)
    for seed in TRUTH_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        half = torch.randint(*TOKEN_RANGE, (TRUTH_BATCH, REPEAT_LENGTH), generator=generator)
        for chunk in torch.cat([half, half], 1).split(CHUNK):
            attentions = model.base_model(chunk, output_attentions=True).attentions
            for layer, attention in enumerate(attentions):
                for t in range(REPEAT_LENGTH, 2 * REPEAT_LENGTH - 1):
                    induction[layer] += attention[:, :, t, t - REPEAT_LENGTH + 1].double().sum(0).numpy()
                for t in range(1, REPEAT_LENGTH):
                    previous[layer] += attention[:, :, t, t - 1].double().sum(0).numpy()
    total = len(TRUTH_SEEDS) * TRUTH_BATCH * (REPEAT_LENGTH - 1)
    return (induction / total).ravel(), (previous / total).ravel()


@torch.no_grad()
def repeat_accuracy(model, tokens: torch.Tensor, heads: list[int]) -> float:
    """Returns how often the top next token in the second copy is right, with the given heads' values zeroed."""
    config, size = model.config, head_size(model.config)
    hooks = []
    for layer in {index // config.num_attention_heads for index in heads}:
        chosen = [index % config.num_attention_heads for index in heads if index // config.num_attention_heads == layer]

        def zero_values(_module, _inputs, output, chosen=chosen):
            output = output.clone()
            for head in chosen:
                start = qkv_starts(config, head)[2]
                output[..., start : start + size] = 0
            return output

        hooks.append(layer_modules(model, layer)[1].register_forward_hook(zero_values))
    right = 0
    for chunk in tokens.split(CHUNK):
        predicted = model(chunk).logits[:, REPEAT_LENGTH:-1].argmax(-1)
        right += int((predicted == chunk[:, REPEAT_LENGTH + 1 :]).sum())
    for hook in hooks:
        hook.remove()
    return right / (len(tokens) * (REPEAT_LENGTH - 1))


def chance(heads: int) -> tuple[float, float]:
    """Returns the expected overlap of two random sets of 5, and the chance of 3 or more."""
    at_least_three = sum(comb(TOP, k) * comb(heads - TOP, TOP - k) for k in range(3, TOP + 1)) / comb(heads, TOP)
    return TOP * TOP / heads, at_least_three


def study(model_name: str) -> dict:
    began = time.time()
    model = AutoModelForCausalLM.from_pretrained(model_name, use_safetensors=True, dtype=torch.float32).eval()
    model.set_attn_implementation("eager")
    config = model.config
    heads = head_count(config)
    total = sum(parameter.numel() for parameter in model.parameters())
    used = sum(
        module.weight.numel() for layer in range(config.num_hidden_layers) for module in layer_modules(model, layer)
    )
    print(f"\n{model_name}: {config.model_type}, {config.num_hidden_layers} layers, {config.num_attention_heads} heads, "
          f"width {config.hidden_size}, head size {head_size(config)}, {heads} heads in all")
    print(f"  weights: {total:,} in all, the statistic uses {used:,} ({used / total:.1%})")

    cosine = cosines(model)
    layers = np.arange(heads) // config.num_attention_heads
    forward = statistic_picks(cosine, layers[:, None] > layers[None, :])
    swapped = statistic_picks(cosine, layers[:, None] < layers[None, :])
    gc.collect()

    # The check always runs in float64, where a wrong slice cannot pass for rounding.
    model.double()
    worst = check_slices(model)
    print(f"  slices rebuild the model's attention and output in float64, worst error {worst:.2e}")
    if worst > SLICE_TOLERANCE:
        raise ValueError(f"{model_name}: slices do not match the model, error {worst}")
    if model_name in FLOAT32_FORWARD:
        model.float()
    print(f"  forward passes in {next(model.parameters()).dtype}, statistic in float64")
    scores, previous = measured_scores(model)
    model.float()

    order = np.argsort(-scores)
    top = set(order[:TOP].tolist())
    rank = np.argsort(order)
    print("  top heads by score: " + ", ".join(f"{name(int(i), config)} {scores[i]:.3f}" for i in order[:SHOWN]))
    print(f"  5th/6th {scores[order[TOP - 1]]:.3f}/{scores[order[TOP]]:.3f}, gap {scores[order[TOP - 1]] - scores[order[TOP]]:.3f}")
    expected, three = chance(heads)
    print(f"  chance overlap from {heads} heads: expected {expected:.3f} of 5, chance of 3 or more {three:.2e}")

    results = {}
    for label, picks in (("statistic", forward), ("swapped", swapped)):
        source = picks["source"]
        print(f"  {label}: source {name(source, config)}, best pairing of {picks['votes']} of {picks['voters']} heads, "
              f"{previous[source]:.3f} on the previous token, rank {int(np.argsort(np.argsort(-previous))[source]) + 1} of {heads} on it")
        for variant in ("by_source", "best_pairing"):
            hits = len(set(picks[variant]) & top)
            results[label, variant] = hits
            shown = ", ".join(f"{name(int(i), config)} (score {scores[i]:.3f}, rank {rank[i] + 1})" for i in picks[variant])
            print(f"    {variant}: {hits} of {TOP}: {shown}")
    top_previous = np.argsort(-previous)[:SHOWN]
    print("  top heads on the previous token: " + ", ".join(f"{name(int(i), config)} {previous[i]:.3f}" for i in top_previous))
    # Not on the card and not weights-only: the source is chosen by measured attention, to explain a low by_source.
    measured_source = int(top_previous[0])
    allowed = np.where(layers > layers[measured_source], cosine[:, measured_source], -np.inf)
    print(f"  diagnostic, source set to the top previous-token head {name(measured_source, config)}: "
          f"{len(set(np.argsort(-allowed)[:TOP].tolist()) & top)} of {TOP}")

    generator = torch.Generator().manual_seed(ABLATION_SEED)
    half = torch.randint(*TOKEN_RANGE, (ABLATION_BATCH, REPEAT_LENGTH), generator=generator)
    tokens = torch.cat([half, half], 1)
    intact = repeat_accuracy(model, tokens, [])
    without_top = repeat_accuracy(model, tokens, order[:TOP].tolist())
    rng = np.random.default_rng(ABLATION_SEED)
    others = order[TOP:]
    without_random = [repeat_accuracy(model, tokens, rng.choice(others, TOP, replace=False).tolist()) for _ in range(RANDOM_SETS)]
    print(f"  repeat accuracy: intact {intact:.3f}, top 5 zeroed {without_top:.3f} ({100 * (intact - without_top):.1f} points), "
          f"{RANDOM_SETS} random sets of 5 from the other heads: mean {np.mean(without_random):.3f}, "
          f"lowest {min(without_random):.3f}, highest {max(without_random):.3f}")
    print(f"  took {time.time() - began:.0f} s", flush=True)
    return {
        "name": model_name, "total": total, "share": used / total, "heads": heads, "top": scores[order[TOP - 1]],
        "gap": scores[order[TOP - 1]] - scores[order[TOP]], "results": results, "chance": expected,
        "intact": intact, "without_top": without_top, "without_random": float(np.mean(without_random)),
        "lowest_random": min(without_random),
    }


def main() -> None:
    print("TinyStories-8M dropped: it has no safetensors file on the Hub, only pytorch_model.bin.")
    rows = []
    for model_name in MODELS:
        rows.append(study(model_name))
        gc.collect()
    print("\nmodel | weights | heads | 5th score | gap | by source | best pairing | chance | swapped by source | swapped best pairing | intact | top 5 zeroed | random mean | random lowest")
    for row in rows:
        results = row["results"]
        print(f"{row['name']} | {row['total']:,} | {row['heads']} | {row['top']:.3f} | {row['gap']:.3f} | "
              f"{results['statistic', 'by_source']} | {results['statistic', 'best_pairing']} | {row['chance']:.3f} | "
              f"{results['swapped', 'by_source']} | {results['swapped', 'best_pairing']} | {row['intact']:.3f} | "
              f"{row['without_top']:.3f} | {row['without_random']:.3f} | {row['lowest_random']:.3f}")


if __name__ == "__main__":
    main()
