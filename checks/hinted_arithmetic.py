"""Checks the hinted read's hand arithmetic against the same query-key dot products computed in code.

Usage: hinted_arithmetic.py <run_dir>
"""
import json
import re
import sys
from math import comb
from pathlib import Path

import numpy as np

from wsp.disguise import HEAD_SIZE, HEADS, WIDTH, disguise
from wsp.network import TokenShuffle
from wsp.reading import HINTED_SEED, query_key_dot, stated_dot_products

ROW = 137
FIRST_LAYER, LAST_LAYER = 4, 11

run_dir = Path(sys.argv[1])
turns = [json.loads(line) for line in (run_dir / "transcript.jsonl").read_text().splitlines()]
disguised = disguise(HINTED_SEED, TokenShuffle())


def dot(layer, head, row):
    return query_key_dot(disguised, layer, head, row)


stated = stated_dot_products([turn["reply"] for turn in turns[1:3]])
if len(stated) != (LAST_LAYER - FIRST_LAYER + 1) * HEADS:
    raise ValueError(f"expected every head in layers {FIRST_LAYER} to {LAST_LAYER}, parsed {len(stated)}")

keys = sorted(stated)
hand = np.array([stated[key] for key in keys])
true = np.array([dot(layer, head, ROW) for layer, head in keys])
errors = np.abs(hand - true)
print(f"row {ROW}, {len(keys)} heads: correlation {np.corrcoef(hand, true)[0, 1]:.4f}, mean error {errors.mean():.3f}, median {np.median(errors):.3f}, max {errors.max():.3f}, within 0.05: {(errors < 0.05).mean():.0%}")
worst = np.argsort(-errors)[:5]
print("worst", [(f"{keys[i][0]}.{keys[i][1]}", float(hand[i]), round(float(true[i]), 3)) for i in worst])
print(f"same-sign share: {(np.sign(hand) == np.sign(true)).mean():.0%}, true std {true.std():.3f}")

three_rows = (ROW, 491, 719)
stated_means = {}
for match in re.finditer(r"^\| (\d+)\.(\d+) \| ([\d.]+) \|$", turns[4]["reply"], re.M):
    stated_means[int(match.group(1)), int(match.group(2))] = float(match.group(3))
for key, value in stated_means.items():
    print(f"three-row mean {key[0]}.{key[1]}: stated {value:.2f}, code {np.mean([dot(*key, row) for row in three_rows]):.3f}")

answer = [tuple(pick) for pick in json.loads((run_dir / "result.json").read_text())["answer"]]
rows_by_head = {key: [ROW] for key in keys}
for turn in turns:
    for layer, row_text, column in re.findall(r"L(\d+)\.qkv\.W\[([\d:]+), (\d+):\d+\]", turn["reply"]):
        column = int(column)
        if column >= WIDTH:
            continue
        head = column // HEAD_SIZE
        parts = [int(part) for part in row_text.split(":")]
        rows = range(parts[0], parts[1], parts[2]) if len(parts) == 3 else parts
        for row in rows:
            if row not in rows_by_head[int(layer), head]:
                rows_by_head[int(layer), head].append(row)
code_means = {key: np.mean([dot(*key, row) for row in rows]) for key, rows in rows_by_head.items()}
print("rows read per head:", {count: sum(len(rows) == count for rows in rows_by_head.values()) for count in sorted({len(rows) for rows in rows_by_head.values()})})
ranked = sorted(code_means, key=code_means.get, reverse=True)[:8]
print("same rows, same statistic, in code, top 8 by mean:", [(f"{l}.{h}", len(rows_by_head[l, h]), round(float(code_means[l, h]), 3)) for l, h in ranked])
print("its answer:", answer)

for name, heads in (("all 144 heads", 144), ("layers 4 to 11", 96)):
    chance = sum(comb(10, hits) * comb(heads - 10, 5 - hits) for hits in (4, 5)) / comb(heads, 5)
    print(f"chance of at least 4 of 5, {name}: {chance:.1e}")
for heads in (144, 96):
    print(f"chance of 0 of 10 over two reads, {heads} heads: {(comb(heads - 10, 5) / comb(heads, 5)) ** 2:.2f}")

calls = [json.loads(path.read_text()) for path in sorted((run_dir / "calls").glob("[0-9]*.json"))]
print("per turn: output tokens, of which reasoning, visible reply characters")
for call, turn in zip(calls, turns):
    print(call["call_id"], call["usage"]["output_tokens"], call["usage"]["reasoning_output_tokens"], len(turn["reply"]))
    print("  event types", sorted({event.get("item", {}).get("type", event["type"]) for event in call["response"]["events"]}))
for first, last, call in ((4, 7, calls[1]), (8, 11, calls[2])):
    chosen = [i for i, key in enumerate(keys) if first <= key[0] <= last]
    products = len(chosen) * HEAD_SIZE
    print(f"layers {first}-{last}: {products} products, {call['usage']['reasoning_output_tokens']} reasoning tokens, {call['usage']['reasoning_output_tokens'] / products:.1f} per product, correlation {np.corrcoef(hand[chosen], true[chosen])[0, 1]:.4f}, median error {np.median(errors[chosen]):.3f}, max {errors[chosen].max():.3f}")
