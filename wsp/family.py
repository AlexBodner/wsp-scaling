"""Scrambles other GPT-2 family models and builds their prompts, for reading runs beyond GPT-2 small.

The scramble repeats `wsp.disguise` with each model's own shape, and leaves that frozen code alone. These models keep
their real token order, since the reader gets no token table.
"""

from dataclasses import dataclass

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPTNeoForCausalLM

from wsp.disguise import _random_invertible
from wsp.network import writable_tensors
from wsp.reading import _REPEAT_LENGTH, _TRUTH_BATCH, _TRUTH_SEEDS, _TRUTH_TOKEN_RANGE, python_prompt

HEAD_SIZE = 64
MODELS = {"medium": "gpt2-medium", "distil": "distilbert/distilgpt2", "small": "gpt2", "neo": "EleutherAI/gpt-neo-125M"}


@dataclass
class FamilyDisguise:
    """The disguised weights by neutral name, and where each head came from."""

    tensors: dict[str, np.ndarray]
    original_head: np.ndarray


def _gpt2_from_neo(name: str) -> GPT2LMHeadModel:
    """Returns GPT-Neo's weights in GPT-2's layout. Neo has no attention scale and no query, key or value bias, and its local windows don't reach past 256 tokens."""
    neo = GPTNeoForCausalLM.from_pretrained(name, use_safetensors=True).eval()
    config = GPT2Config(n_layer=neo.config.num_layers, n_head=neo.config.num_heads, n_embd=neo.config.hidden_size, n_positions=neo.config.max_position_embeddings, scale_attn_weights=False, activation_function="gelu_new", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
    model = GPT2LMHeadModel(config).eval()
    with torch.no_grad():
        model.transformer.wte.weight.copy_(neo.transformer.wte.weight)
        model.transformer.wpe.weight.copy_(neo.transformer.wpe.weight)
        model.transformer.ln_f.load_state_dict(neo.transformer.ln_f.state_dict())
        for block, source in zip(model.transformer.h, neo.transformer.h):
            attention = source.attn.attention
            block.ln_1.load_state_dict(source.ln_1.state_dict())
            block.ln_2.load_state_dict(source.ln_2.state_dict())
            block.attn.c_attn.weight.copy_(torch.cat([attention.q_proj.weight.T, attention.k_proj.weight.T, attention.v_proj.weight.T], 1))
            block.attn.c_attn.bias.zero_()
            block.attn.c_proj.weight.copy_(attention.out_proj.weight.T)
            block.attn.c_proj.bias.copy_(attention.out_proj.bias)
            block.mlp.c_fc.weight.copy_(source.mlp.c_fc.weight.T)
            block.mlp.c_fc.bias.copy_(source.mlp.c_fc.bias)
            block.mlp.c_proj.weight.copy_(source.mlp.c_proj.weight.T)
            block.mlp.c_proj.bias.copy_(source.mlp.c_proj.bias)
    model.set_attn_implementation("eager")
    return model


def load(short_name: str) -> GPT2LMHeadModel:
    if short_name == "neo":
        return _gpt2_from_neo(MODELS[short_name])
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
    if width != 768 and any(token in prompt for token in (" 768", "2304", "1535", "124,439")):
        raise ValueError("GPT-2 small's shape survived in the prompt")
    if "L5" in prompt or "3.7" in prompt:
        raise ValueError("a layer-naming example survived in the prompt")
    return prompt


@torch.no_grad()
def induction_scores(model: GPT2LMHeadModel) -> np.ndarray:
    """Measures each head's attention to the token after the earlier copy, with the settings GPT-2 small's truth uses."""
    return np.mean([seed_scores(model, seed) for seed in _TRUTH_SEEDS], axis=0)


@torch.no_grad()
def seed_scores(model: GPT2LMHeadModel, seed: int) -> np.ndarray:
    """Measures the same score on one seed's sequences, to see how stable near-ties are."""
    scores = np.zeros((model.config.n_layer, model.config.n_head))
    second = range(_REPEAT_LENGTH, 2 * _REPEAT_LENGTH - 1)
    generator = torch.Generator().manual_seed(seed)
    half = torch.randint(*_TRUTH_TOKEN_RANGE, (_TRUTH_BATCH, _REPEAT_LENGTH), generator=generator)
    attentions = model(torch.cat([half, half], 1), output_attentions=True).attentions
    for layer, attention in enumerate(attentions):
        mean = attention.mean(0)
        for t in second:
            scores[layer] += mean[:, t, t - _REPEAT_LENGTH + 1].numpy()
    return scores / len(second)
