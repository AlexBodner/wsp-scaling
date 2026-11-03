"""Looks for cheap statistics that would solve the reading test without the real calculation."""
import json

import numpy as np

from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH, disguise
from wsp.network import TokenShuffle, real_network
from wsp.reading import disguised_scores, head_scores, positive_heads

SEEDS = range(3000, 3020)
LATER_LAYERS_FROM = LAYERS // 3


def statistics(tensors):
    out = {}
    for layer in range(LAYERS):
        weight, bias = tensors[f"L{layer}.qkv.W"], tensors[f"L{layer}.qkv.b"]
        output = tensors[f"L{layer}.out.W"]
        for head in range(HEADS):
            q, k, v = (weight[:, offset + head * HEAD_SIZE : offset + (head + 1) * HEAD_SIZE] for offset in (0, WIDTH, 2 * WIDTH))
            bq, bk, bv = (bias[offset + head * HEAD_SIZE : offset + (head + 1) * HEAD_SIZE] for offset in (0, WIDTH, 2 * WIDTH))
            o = output[head * HEAD_SIZE : (head + 1) * HEAD_SIZE]
            values = {
                "norm Q": np.linalg.norm(q),
                "norm K": np.linalg.norm(k),
                "norm Q * norm K": np.linalg.norm(q) * np.linalg.norm(k),
                "norm V": np.linalg.norm(v),
                "norm O": np.linalg.norm(o),
                "norm V * norm O": np.linalg.norm(v) * np.linalg.norm(o),
                "norm query bias": np.linalg.norm(bq),
                "norm key bias": np.linalg.norm(bk),
                "query bias . key bias": bq @ bk,
                "norm value bias": np.linalg.norm(bv),
                "Q[r] . K[r], all rows": (q * k).sum(),
            }
            for name, value in values.items():
                out.setdefault(name, np.zeros((LAYERS, HEADS)))[layer, head] = value
    return out


def precision(scores, positives):
    masked = scores.copy()
    masked[: LATER_LAYERS_FROM] = -np.inf
    order = np.argsort(-masked.ravel())[:5]
    return np.mean([(int(i) // HEADS, int(i) % HEADS) in positives for i in order])


shuffle = TokenShuffle()
real = real_network(shuffle).double()
real.set_attn_implementation("eager")
real_scores = head_scores(real)
table = {}
for seed in SEEDS:
    disguised = disguise(seed, shuffle)
    positives = positive_heads(disguised_scores(real_scores, disguised))
    for name, scores in statistics(disguised.tensors).items():
        table.setdefault(name, []).append(precision(scores, positives))
density = [len([h for h in positive_heads(real_scores) if h[0] == layer]) / HEADS for layer in range(LAYERS)]
print("chance", 10 / 144, "best layer density", max(density), "density", np.round(density, 2).tolist())
print("true top 10", sorted(positive_heads(real_scores)))
for name, values in sorted(table.items(), key=lambda item: -np.mean(item[1])):
    print(f"{name:26s} {np.mean(values):.3f} sd {np.std(values):.3f}")
