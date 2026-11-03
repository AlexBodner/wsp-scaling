"""Real GPT-2 small with the same behaviour but different numbers, so memory of it does not help.

Each layer's heads are reordered, and each head's internal coordinates are changed by random
invertible matrices: Q to QA, K to KA^-T, V to VC, and the output's matching rows to C^-1 O. The key
bias is replaced by noise, because the softmax over source positions ignores it: it only adds the
same amount to every position. Left in place, its dot with the query bias reveals copying heads.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import ortho_group
from transformers import GPT2LMHeadModel

from wsp.network import TokenShuffle, real_network, writable_tensors

LAYERS = 12
HEADS = 12
WIDTH = 768
HEAD_SIZE = 64
_MAX_CONDITION = 8.0


@dataclass
class DisguisedModel:
    """The disguised weights by neutral name, and where each head came from."""

    tensors: dict[str, np.ndarray]
    original_head: np.ndarray  # layer x new head -> original head


def _random_invertible(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    left = ortho_group.rvs(HEAD_SIZE, random_state=int(rng.integers(1 << 31)))
    right = ortho_group.rvs(HEAD_SIZE, random_state=int(rng.integers(1 << 31)))
    condition = np.exp(rng.uniform(0, np.log(_MAX_CONDITION)))
    scales = np.exp(rng.uniform(-np.log(condition), np.log(condition), HEAD_SIZE)) * np.exp(rng.uniform(-1, 1))
    return left @ np.diag(scales) @ right, right.T @ np.diag(1 / scales) @ left.T


def disguise(seed: int, shuffle: TokenShuffle) -> DisguisedModel:
    """Returns the disguised model for one seed. The same seed always gives the same numbers."""
    rng = np.random.default_rng(seed)
    source = writable_tensors(real_network(shuffle))
    tensors = {name: tensor.detach().numpy().astype(np.float64).copy() for name, tensor in source.items()}
    original_head = np.zeros((LAYERS, HEADS), dtype=int)
    size = HEAD_SIZE
    for layer in range(LAYERS):
        weight = tensors[f"L{layer}.qkv.W"].copy()
        bias = tensors[f"L{layer}.qkv.b"].copy()
        output = tensors[f"L{layer}.out.W"].copy()
        order = rng.permutation(HEADS)
        original_head[layer] = order
        for new, old in enumerate(order):
            query_map, query_inverse = _random_invertible(rng)
            value_map, value_inverse = _random_invertible(rng)
            q_old, k_old, v_old = old * size, WIDTH + old * size, 2 * WIDTH + old * size
            q_new, k_new, v_new = new * size, WIDTH + new * size, 2 * WIDTH + new * size
            target = tensors[f"L{layer}.qkv.W"]
            target[:, q_new : q_new + size] = weight[:, q_old : q_old + size] @ query_map
            target[:, k_new : k_new + size] = weight[:, k_old : k_old + size] @ query_inverse.T
            target[:, v_new : v_new + size] = weight[:, v_old : v_old + size] @ value_map
            new_query_bias = bias[q_old : q_old + size] @ query_map
            bias_target = tensors[f"L{layer}.qkv.b"]
            bias_target[q_new : q_new + size] = new_query_bias
            bias_target[k_new : k_new + size] = rng.normal(0, new_query_bias.std(), size)
            bias_target[v_new : v_new + size] = bias[v_old : v_old + size] @ value_map
            tensors[f"L{layer}.out.W"][q_new : q_new + size] = value_inverse @ output[q_old : q_old + size]
    return DisguisedModel(tensors, original_head)


def model_from(disguised: DisguisedModel) -> GPT2LMHeadModel:
    """Returns a runnable double-precision model holding the disguised weights."""
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").double().eval()
    target = writable_tensors(model)
    with torch.no_grad():
        for name, values in disguised.tensors.items():
            target[name].copy_(torch.from_numpy(values))
    return model
