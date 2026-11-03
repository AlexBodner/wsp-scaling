"""Tests the statistic named in the hint, with and without skipping early layers, across row budgets."""
import numpy as np

from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH, disguise
from wsp.network import TokenShuffle, real_network
from wsp.reading import disguised_scores, head_scores, positive_heads

shuffle = TokenShuffle()
real = real_network(shuffle).double()
real.set_attn_implementation("eager")
real_scores = head_scores(real)
rows_by_budget = {"all rows": None, "20k values": 156, "10k values": 78}
results = {}
for seed in range(4000, 4020):
    rng = np.random.default_rng(seed)
    disguised = disguise(seed, shuffle)
    positives = positive_heads(disguised_scores(real_scores, disguised))
    for label, total_rows in rows_by_budget.items():
        for first_layer in (0, LAYERS // 3):
            count = (LAYERS - first_layer) * HEADS
            per_head = None if total_rows is None else total_rows // count
            extra = 0 if total_rows is None else total_rows % count
            scores = np.full((LAYERS, HEADS), -np.inf)
            position = 0
            for layer in range(first_layer, LAYERS):
                weight = disguised.tensors[f"L{layer}.qkv.W"]
                for head in range(HEADS):
                    q = weight[:, head * HEAD_SIZE : (head + 1) * HEAD_SIZE]
                    k = weight[:, WIDTH + head * HEAD_SIZE : WIDTH + (head + 1) * HEAD_SIZE]
                    if per_head is None:
                        rows = np.arange(WIDTH)
                    else:
                        rows = rng.choice(WIDTH, per_head + (position < extra), replace=False)
                    position += 1
                    if len(rows):
                        scores[layer, head] = (q[rows] * k[rows]).sum() / len(rows)
            top = np.argsort(-scores.ravel())[:5]
            hits = np.mean([(int(i) // HEADS, int(i) % HEADS) in positives for i in top])
            results.setdefault((label, first_layer), []).append(hits)
for (label, first_layer), values in sorted(results.items()):
    print(f"{label:12s} from layer {first_layer}: {np.mean(values):.3f} sd {np.std(values):.3f}")
