"""Audits one finished reading run: seals, cost, where its picks rank, and what two statistics reach on its disguise.

Usage: read_audit.py <run_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH, disguise
from wsp.network import TokenShuffle, real_network
from wsp.reading import disguised_scores, head_scores, positive_heads

INPUT_PER_MILLION, CACHED_PER_MILLION, OUTPUT_PER_MILLION = 10, 1, 50

run_dir = Path(sys.argv[1])
result = json.loads((run_dir / "result.json").read_text())
calls = [json.loads(path.read_text()) for path in sorted((run_dir / "calls").glob("[0-9]*.json"))]

usage = [call["usage"] for call in calls]
fresh = sum(u["input_tokens"] - u["cached_input_tokens"] for u in usage)
cached = sum(u["cached_input_tokens"] for u in usage)
output = sum(u["output_tokens"] for u in usage)
cost = (fresh * INPUT_PER_MILLION + cached * CACHED_PER_MILLION + output * OUTPUT_PER_MILLION) / 1e6
print(f"calls {len(calls)}, input {fresh + cached} ({cached} cached), output {output}, list price ${cost:.2f}")
print("latency minutes", [round(call["latency_s"] / 60, 1) for call in calls], "stops", {call["stop_reason"] for call in calls})
for call in calls:
    seal = call["seal"]
    print(call["call_id"], {key: value for key, value in seal.items() if key not in ("call_dir", "thread_id", "rollout", "turn_context")})

shuffle = TokenShuffle()
real = real_network(shuffle).double()
real.set_attn_implementation("eager")
disguised = disguise(result["seed"], shuffle)
scores = disguised_scores(head_scores(real), disguised)
positives = positive_heads(scores)
ranks = np.empty(LAYERS * HEADS, int)
ranks[np.argsort(-scores.ravel())] = np.arange(LAYERS * HEADS)
print("true top 10", sorted(positives))
print("hits recomputed", sum(tuple(pick) in positives for pick in result["answer"]), "stored", result["hits"])
for layer, head in result["answer"]:
    print(f"named {layer}.{head}: score {scores[layer, head]:.4f}, rank {ranks[layer * HEADS + head] + 1} of 144")
print("10th score", round(float(np.sort(scores.ravel())[-10]), 4), "median", round(float(np.median(scores)), 4))

value_output, query_key = np.zeros((LAYERS, HEADS)), np.zeros((LAYERS, HEADS))
for layer in range(LAYERS):
    weight, out = disguised.tensors[f"L{layer}.qkv.W"], disguised.tensors[f"L{layer}.out.W"]
    for head in range(HEADS):
        start, end = head * HEAD_SIZE, (head + 1) * HEAD_SIZE
        value_output[layer, head] = np.trace(weight[:, 2 * WIDTH + start : 2 * WIDTH + end] @ out[start:end])
        query_key[layer, head] = (weight[:, start:end] * weight[:, WIDTH + start : WIDTH + end]).sum()


def top_five_from_layer_4(statistic):
    masked = statistic.copy()
    masked[:4] = -np.inf
    return [(int(i) // HEADS, int(i) % HEADS) for i in np.argsort(-masked.ravel())[:5]]


for name, statistic in (("value-output, all rows", value_output), ("query-key, all rows", query_key)):
    picks = top_five_from_layer_4(statistic)
    print(name, "layers 4-11, top 5:", picks, "hits", sum(pick in positives for pick in picks))
    print(f"  rank of its picks on this statistic:", [int((statistic[4:] > statistic[l, h]).sum() + 1) for l, h in result["answer"]])
