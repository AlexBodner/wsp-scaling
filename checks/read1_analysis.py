"""Checks the first read: how good its picks were, whether its arithmetic held, and what its statistic can reach."""
import glob
import json

import numpy as np

from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH, disguise
from wsp.network import TokenShuffle, real_network
from wsp.reading import BLIND_SEED, READ_SEEDS, disguised_scores, head_scores, positive_heads

shuffle = TokenShuffle()
real = real_network(shuffle).double()
real.set_attn_implementation("eager")
real_scores = head_scores(real)
disguised = disguise(READ_SEEDS[0], shuffle)
scores = disguised_scores(real_scores, disguised)
positives = positive_heads(scores)
ranks = np.empty(144, int)
ranks[np.argsort(-scores.ravel())] = np.arange(144)
print("true top 10 in this disguise", sorted(positives))
named = [(5, 1), (5, 7), (6, 7), (7, 6), (8, 2)]
for layer, head in named:
    print(f"read named {layer}.{head}: score {scores[layer, head]:.4f}, rank {ranks[layer * HEADS + head] + 1} of 144, original head {disguised.original_head[layer, head]}")
blind = [(5, 0), (5, 1), (6, 0), (7, 0), (7, 1)]
print("blind ranks", [int(ranks[l * HEADS + h] + 1) for l, h in blind], "hits on read-1 positives", sum(h in positives for h in blind))
print("10th score", np.sort(scores.ravel())[-10], "median", np.median(scores))

# Value-to-output alignment, the statistic it used, on every row and on its first sample.
full, sampled = np.zeros((LAYERS, HEADS)), np.full((LAYERS, HEADS), np.nan)
for layer in range(LAYERS):
    weight, output = disguised.tensors[f"L{layer}.qkv.W"], disguised.tensors[f"L{layer}.out.W"]
    for head in range(HEADS):
        v = weight[:, 2 * WIDTH + head * HEAD_SIZE : 2 * WIDTH + (head + 1) * HEAD_SIZE]
        o = output[head * HEAD_SIZE : (head + 1) * HEAD_SIZE]
        full[layer, head] = np.trace(v @ o)
        rows = np.array([0, 384])
        columns = np.arange(0, HEAD_SIZE, 4)
        sampled[layer, head] = sum((v[row, column] * o[column, row]) for row in rows for column in columns)


def top(values, first=4):
    masked = values.copy()
    masked[: first] = -np.inf
    return [(int(i) // HEADS, int(i) % HEADS) for i in np.argsort(-np.nan_to_num(masked, nan=-np.inf).ravel())[:5]]


picks = top(full)
print("value-output trace, all rows, top 5:", picks, "hits", sum(p in positives for p in picks))
sample_picks = top(np.where(np.arange(LAYERS)[:, None] <= 9, sampled, -np.inf))
print("its first sample, recomputed in code, top 5:", sample_picks, "hits", sum(p in positives for p in sample_picks))
print("recomputed first sample for the heads it called strongest:", {f"{l}.{h}": round(float(sampled[l, h]), 3) for l, h in [(5, 7), (6, 7), (7, 6), (8, 2), (9, 9), (6, 3), (9, 0)]})
print("rank of those in its first sample:", {f"{l}.{h}": int((np.nan_to_num(sampled[4:10]) > sampled[l, h]).sum() + 1) for l, h in [(5, 7), (6, 7), (7, 6), (8, 2), (9, 9), (6, 3), (9, 0)]})
qk = np.zeros((LAYERS, HEADS))
for layer in range(LAYERS):
    weight = disguised.tensors[f"L{layer}.qkv.W"]
    for head in range(HEADS):
        qk[layer, head] = (weight[:, head * HEAD_SIZE : (head + 1) * HEAD_SIZE] * weight[:, WIDTH + head * HEAD_SIZE : WIDTH + (head + 1) * HEAD_SIZE]).sum()
picks = top(qk)
print("query-key statistic, top 5:", picks, "hits", sum(p in positives for p in picks))

for path in sorted(glob.glob("runs/reading-*__live__*")):
    total_input = cached = output = 0
    for call in sorted(glob.glob(path + "/calls/0*.json")):
        usage = json.load(open(call))["usage"]
        total_input += usage["input_tokens"]
        cached += usage["cached_input_tokens"]
        output += usage["output_tokens"]
    price = ((total_input - cached) * 10 + cached * 1 + output * 50) / 1e6
    print(path.split("/")[-1], "input", total_input, "cached", cached, "output", output, f"list price ${price:.2f}")
