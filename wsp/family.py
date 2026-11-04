"""Scrambles other GPT-2 family models and builds their prompts, for reading runs beyond GPT-2 small.

The scramble repeats `wsp.disguise` with each model's own shape, and leaves that frozen code alone. These models keep
their real token order, since the reader gets no token table.
"""

from dataclasses import dataclass

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from wsp.disguise import _random_invertible
from wsp.network import writable_tensors
from wsp.reading import python_prompt

HEAD_SIZE = 64
REPEAT_LENGTH = 50
TRUTH_SEEDS = (11, 12, 13, 14)
TRUTH_BATCH = 16
VOCABULARY = 50257
MODELS = {"medium": "gpt2-medium", "distil": "distilbert/distilgpt2", "small": "gpt2"}


@dataclass
class FamilyDisguise:
    """The disguised weights by neutral name, and where each head came from."""

    tensors: dict[str, np.ndarray]
    original_head: np.ndarray


def load(short_name: str) -> GPT2LMHeadModel:
    return GPT2LMHeadModel.from_pretrained(MODELS[short_name], attn_implementation="eager", use_safetensors=True).eval()


def disguise_family(model: GPT2LMHeadModel, seed: int) -> FamilyDisguise:
    """Reorders each layer's heads and changes their internal coordinates, as `wsp.disguise.disguise` does."""
    layers, heads, width = model.config.n_layer, model.config.n_head, model.config.n_embd
    rng = np.random.default_rng(seed)
    tensors = {name: tensor.detach().numpy().astype(np.float64).copy() for name, tensor in writable_tensors(model).items()}
    original_head = np.zeros((layers, heads), dtype=int)
    for layer in range(layers):
        weight, bias, output = tensors[f"L{layer}.qkv.W"].copy(), tensors[f"L{layer}.qkv.b"].copy(), tensors[f"L{layer}.out.W"].copy()
        order = rng.permutation(heads)
        original_head[layer] = order
        for new, old in enumerate(order):
            query_map, query_inverse = _random_invertible(rng)
            value_map, value_inverse = _random_invertible(rng)
            q_old, k_old, v_old = old * HEAD_SIZE, width + old * HEAD_SIZE, 2 * width + old * HEAD_SIZE
            q_new, k_new, v_new = new * HEAD_SIZE, width + new * HEAD_SIZE, 2 * width + new * HEAD_SIZE
            target = tensors[f"L{layer}.qkv.W"]
            target[:, q_new : q_new + HEAD_SIZE] = weight[:, q_old : q_old + HEAD_SIZE] @ query_map
            target[:, k_new : k_new + HEAD_SIZE] = weight[:, k_old : k_old + HEAD_SIZE] @ query_inverse.T
            target[:, v_new : v_new + HEAD_SIZE] = weight[:, v_old : v_old + HEAD_SIZE] @ value_map
            query_bias = bias[q_old : q_old + HEAD_SIZE] @ query_map
            bias_target = tensors[f"L{layer}.qkv.b"]
            bias_target[q_new : q_new + HEAD_SIZE] = query_bias
            bias_target[k_new : k_new + HEAD_SIZE] = rng.normal(0, query_bias.std(), HEAD_SIZE)
            bias_target[v_new : v_new + HEAD_SIZE] = bias[v_old : v_old + HEAD_SIZE] @ value_map
            tensors[f"L{layer}.out.W"][q_new : q_new + HEAD_SIZE] = value_inverse @ output[q_old : q_old + HEAD_SIZE]
    return FamilyDisguise(tensors, original_head)


def check_products(model: GPT2LMHeadModel, disguised: FamilyDisguise) -> float:
    """Returns the largest relative change in QK^T and VO over sampled heads, which should be near zero."""
    width = model.config.n_embd
    real = {name: tensor.detach().numpy().astype(np.float64) for name, tensor in writable_tensors(model).items()}
    worst = 0.0
    for layer in (0, model.config.n_layer // 2, model.config.n_layer - 1):
        for new in (0, model.config.n_head - 1):
            old = int(disguised.original_head[layer][new])

            def products(tensors, head):
                w, o = tensors[f"L{layer}.qkv.W"], tensors[f"L{layer}.out.W"]
                start = head * HEAD_SIZE
                q, k, v = (w[:, offset + start : offset + start + HEAD_SIZE] for offset in (0, width, 2 * width))
                return q @ k.T, v @ o[start : start + HEAD_SIZE]

            for a, b in zip(products(disguised.tensors, new), products(real, old)):
                worst = max(worst, float(np.linalg.norm(a - b) / np.linalg.norm(b)))
    return worst


def family_prompt(model: GPT2LMHeadModel) -> str:
    """Returns the Python prompt with this model's shape and no example that names a layer."""
    layers, heads, width = model.config.n_layer, model.config.n_head, model.config.n_embd
    weights = sum(parameter.numel() for parameter in model.parameters())
    prompt = python_prompt()
    pairs = [
        ("124,439,808", f"{weights:,}"),
        ("12 layers of width 768, 12 attention heads", f"{layers} layers of width {width}, {heads} attention heads"),
        ("out of all 144", f"out of all {layers * heads}"),
        ("the weights of the 12 blocks", f"the weights of the {layers} blocks"),
        ("from 0 to 11:", f"from 0 to {layers - 1}:"),
        ("For each head h from 0 to 11", f"For each head h from 0 to {heads - 1}"),
        ("the 12 heads' o_j joined in order (768 entries)", f"the {heads} heads' o_j joined in order ({width} entries)"),
        ("(entries 0-767), k_j (768-1535), v_j (1536-2303)", f"(entries 0-{width - 1}), k_j ({width}-{2 * width - 1}), v_j ({2 * width}-{3 * width - 1})"),
        ("over the 768 entries", f"over the {width} entries"),
        ("E: 50257 x 768", f"E: 50257 x {width}"),
        ("P: 1024 x 768", f"P: 1024 x {width}"),
        ("F.scale, F.shift: 768", f"F.scale, F.shift: {width}"),
        ("Li.norm1.scale, Li.norm1.shift: 768", f"Li.norm1.scale, Li.norm1.shift: {width}"),
        ("Li.qkv.W: 768 x 2304, Li.qkv.b: 2304", f"Li.qkv.W: {width} x {3 * width}, Li.qkv.b: {3 * width}"),
        ("Li.out.W: 768 x 768, Li.out.b: 768", f"Li.out.W: {width} x {width}, Li.out.b: {width}"),
        ("Li.norm2.scale, Li.norm2.shift: 768", f"Li.norm2.scale, Li.norm2.shift: {width}"),
        ("Li.up.W: 768 x 3072, Li.up.b: 3072", f"Li.up.W: {width} x {4 * width}, Li.up.b: {4 * width}"),
        ("Li.down.W: 3072 x 768, Li.down.b: 768", f"Li.down.W: {4 * width} x {width}, Li.down.b: {width}"),
        ("W['L5.qkv.W'] is 768 x 2304, W['L5.up.W'] is 768 x 3072", f"W['L0.qkv.W'] is {width} x {3 * width}, W['L0.up.W'] is {width} x {4 * width}"),
    ]
    for old, new in pairs:
        if old not in prompt:
            raise ValueError(f"expected to replace {old!r}")
        prompt = prompt.replace(old, new)
    prompt = prompt.replace("```answer\n3.7\n```", "```answer\n0.0\n```")
    if width != 768 and any(token in prompt for token in (" 768", "2304", "3072", "1535", "124,439")):
        raise ValueError("GPT-2 small's shape survived in the prompt")
    if "L5" in prompt or "3.7" in prompt:
        raise ValueError("a layer-naming example survived in the prompt")
    return prompt


@torch.no_grad()
def induction_scores(model: GPT2LMHeadModel) -> np.ndarray:
    """Measures each head's attention to the token after the earlier copy, as the prompt defines it."""
    layers, heads = model.config.n_layer, model.config.n_head
    total = np.zeros((layers, heads))
    second = range(REPEAT_LENGTH, 2 * REPEAT_LENGTH - 1)
    for seed in TRUTH_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        half = torch.randint(0, VOCABULARY, (TRUTH_BATCH, REPEAT_LENGTH), generator=generator)
        attentions = model(torch.cat([half, half], 1), output_attentions=True).attentions
        for layer, attention in enumerate(attentions):
            mean = attention.mean(0)
            for t in second:
                total[layer] += mean[:, t, t - REPEAT_LENGTH + 1].numpy()
    return total / (len(TRUTH_SEEDS) * len(second))
