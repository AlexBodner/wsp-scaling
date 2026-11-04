"""Scores simple weight statistics on the disguised blocks, with no layer rule unless stated, as baselines for the Python run."""
from math import comb

import numpy as np

from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH, disguise
from wsp.network import TokenShuffle, real_network
from wsp.reading import disguised_scores, head_scores

SEEDS = (1004, 1005, 1006, 1007, 1008)
INDUCTION_HEADS = 5
NAMED = 5
LATER_LAYERS_FROM = 4


def statistics(tensors) -> dict[str, np.ndarray]:
    """Returns each statistic per layer and head: traces and eigenvalue shares of QK^T and of VO."""
    out = {name: np.zeros((LAYERS, HEADS)) for name in ("trace QK", "positive eigenvalue share QK", "trace VO", "positive eigenvalue share VO")}
    for layer in range(LAYERS):
        weight, output = tensors[f"L{layer}.qkv.W"], tensors[f"L{layer}.out.W"]
        for head in range(HEADS):
            start = head * HEAD_SIZE
            q = weight[:, start : start + HEAD_SIZE]
            k = weight[:, WIDTH + start : WIDTH + start + HEAD_SIZE]
            v = weight[:, 2 * WIDTH + start : 2 * WIDTH + start + HEAD_SIZE]
            o = output[start : start + HEAD_SIZE]
            for label, matrix in (("QK", q.T @ k), ("VO", o @ v)):
                values = np.linalg.eigvals(matrix)
                out[f"trace {label}"][layer, head] = values.real.sum()
                out[f"positive eigenvalue share {label}"][layer, head] = values.real.sum() / np.abs(values).sum()
    return out


def hits(scores: np.ndarray, truth: np.ndarray, count: int, first_layer: int = 0) -> int:
    ranked = scores.copy()
    ranked[:first_layer] = -np.inf
    picks = set(np.argsort(-ranked.ravel())[:NAMED].tolist())
    return len(picks & set(np.argsort(-truth.ravel())[:count].tolist()))


shuffle = TokenShuffle()
real = real_network(shuffle).double()
real.set_attn_implementation("eager")
real_scores = head_scores(real)
table: dict[str, list[tuple[int, int]]] = {}
for seed in SEEDS:
    disguised = disguise(seed, shuffle)
    truth = disguised_scores(real_scores, disguised)
    for name, scores in statistics(disguised.tensors).items():
        table.setdefault(f"{name}, all layers", []).append((hits(scores, truth, 5), hits(scores, truth, 10)))
        table.setdefault(f"{name}, layers 4 to 11", []).append((hits(scores, truth, 5, LATER_LAYERS_FROM), hits(scores, truth, 10, LATER_LAYERS_FROM)))
for name, values in table.items():
    five, ten = zip(*values)
    print(f"{name:48s} on the five {five} on the top 10 {ten}")


def at_least(heads: int, wanted: int) -> float:
    """Returns the chance that NAMED uniform picks from `heads` include at least `wanted` of the five."""
    return sum(comb(INDUCTION_HEADS, k) * comb(heads - INDUCTION_HEADS, NAMED - k) for k in range(wanted, NAMED + 1)) / comb(heads, NAMED)


for heads, label in ((144, "all heads"), (36, "layers 5 to 7")):
    print(f"guessing, {label}: expected {NAMED * INDUCTION_HEADS / heads:.2f} of 5, at least 3 of 5 {at_least(heads, 3):.1e}")
