"""Rescores every reader on the five heads that score above 0.8, which are GPT-2's induction heads.

The frozen test counts a pick if it is in the top 10, but the scores drop from 0.83 to 0.51 after fifth.
Run scripts/reading_solver.py first, because its per-seed picks are not kept in the repo.
"""
import json
from glob import glob
from math import comb

import numpy as np

from wsp.disguise import HEADS, disguise
from wsp.network import TokenShuffle, real_network
from wsp.reading import head_scores

INDUCTION_HEADS = 5
ASTRA_RUNS = ("blind", "read-1", "read-2", "hinted")


def original_picks(result: dict, shuffle: TokenShuffle) -> list[tuple[int, int]]:
    disguised = disguise(result["seed"], shuffle)
    return [(layer, int(disguised.original_head[layer][head])) for layer, head in result["answer"]]


shuffle = TokenShuffle()
real = real_network(shuffle).double()
real.set_attn_implementation("eager")
scores = head_scores(real)
order = np.argsort(-scores.ravel())
top_five = {(int(i) // HEADS, int(i) % HEADS) for i in order[:INDUCTION_HEADS]}
print("top 10 with scores:", [(f"{int(i) // HEADS}.{int(i) % HEADS}", round(float(scores.ravel()[i]), 3)) for i in order[:10]])
for arm in ASTRA_RUNS:
    result = json.load(open(glob(f"runs/reading-{arm}__live__*/result.json")[0]))
    picks = original_picks(result, shuffle)
    print(arm, picks, "in the top 10:", result["hits"], "in the top 5:", sum(pick in top_five for pick in picks))
by_layer = np.bincount([layer for layer, _ in top_five], minlength=scores.shape[0])
print("best share from knowing only the layers:", by_layer.max() / HEADS)
for heads in (HEADS * 12, HEADS * 8):
    print(f"chance of 0 of 10 on these 5, {heads} heads: {(comb(heads - INDUCTION_HEADS, 5) / comb(heads, 5)) ** 2:.2f}")
hits = []
for path in sorted(glob("runs/reading-solver/seed*/result.json")):
    hits.append(sum(pick in top_five for pick in original_picks(json.load(open(path)), shuffle)))
if not hits:
    raise ValueError("no scripted runs found: run scripts/reading_solver.py first")
print("script, 30 seeds, in the top 5: mean", np.mean(hits), "sd", round(float(np.std(hits)), 2), "counts", sorted(hits))
