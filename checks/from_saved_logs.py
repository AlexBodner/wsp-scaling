"""Recounts the post's headline numbers from saved files. Needs no GPT-2 download and makes no model calls."""
import json
from glob import glob

import numpy as np

HEADS = 12
INDUCTION_HEADS = 5


def run_result(arm: str) -> dict:
    return json.load(open(glob(f"runs/reading-{arm}__live__*/result.json")[0]))


heads = json.load(open("figures/reading_heads_v8.json"))
scores = np.array(heads["scores"])
top_ten = {tuple(head) for head in heads["positives"]}
top_five = {(int(i) // HEADS, int(i) % HEADS) for i in np.argsort(-scores.ravel())[:INDUCTION_HEADS]}
for title, panel in heads["panels"].items():
    picks = [tuple(pick) for pick in panel["picks"]]
    print(f"{title}: {sum(p in top_five for p in picks)} of {len(picks)} on the five, {sum(p in top_ten for p in picks)} on the top 10")
print("layers named with no weights:", sorted(layer for layer, _ in run_result("blind")["answer"]))
print("layers of the five:", sorted(layer for layer, _ in top_five))

arithmetic = json.load(open("figures/arithmetic_v7.json"))
pairs = arithmetic["pairs"]
matched = sum(round(p["typed"], 2) == round(p["in_code"], 2) for p in pairs)
print(f"sums matching code to two decimals: {matched} of {len(pairs)}")
print(f"largest miss: {max(abs(p['typed'] - p['in_code']) for p in pairs):.3f}")
print("script mean on the top 10, of 5:", json.load(open("runs/reading-solver/summary.json"))["mean_precision_at_5"] * 5)
