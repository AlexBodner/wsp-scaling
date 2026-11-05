"""Runs known recipes for finding induction heads on scrambled GPT-2 medium, through the reader's own sandbox. Makes no Astra calls.

This sets the bar for Astra on this model, and shows whether a reader has the memory and time to compute them. Scores are
the picks' overlap with the true top 5 and the picks' mean true score, because the top scores are close together.
"""
import json
import time
from pathlib import Path

import numpy as np

from wsp.code_reader import run_code, save_blocks
from wsp.family import disguise_family, induction_scores, load, seed_scores
from wsp.reading import _TRUTH_SEEDS

SEED = 1299
WEIGHTS = Path(__file__).parent.parent / "data" / "family" / f"blocks_{SEED}.npz"
NAMED = 5
TEMPLATE = """
import numpy as np, resource, time
start = time.time()
FLOAT = {dtype}
LAYERS, HEADS, WIDTH, SIZE = 24, 16, 1024, 64
a, b = [], []
for layer in range(LAYERS):
    scale = np.asarray(W[f"L{{layer}}.norm1.scale"], dtype=FLOAT)[:, None]
    weight = np.asarray(W[f"L{{layer}}.qkv.W"], dtype=FLOAT)
    if {scaled}:
        weight = scale * weight
    output = np.asarray(W[f"L{{layer}}.out.W"], dtype=FLOAT)
    for head in range(HEADS):
        s = head * SIZE
        q, k, v = (weight[:, o + s : o + s + SIZE] for o in (0, WIDTH, 2 * WIDTH))
        if {centered}:
            q, k, v = q - q.mean(0), k - k.mean(0), v - v.mean(0)
        a.append((q @ k.T).ravel())
        b.append((v @ output[s : s + SIZE]).ravel())
a, b = np.stack(a), np.stack(b)
a /= np.linalg.norm(a, axis=1, keepdims=True)
b /= np.linalg.norm(b, axis=1, keepdims=True)
cosine = a @ b.T
total = LAYERS * HEADS
layer_of = np.arange(total) // HEADS
cosine[layer_of[:, None] <= layer_of[None, :]] = -np.inf
name = lambda i: f"{{i // HEADS}}.{{i % HEADS}}"
best = cosine.max(1)
votes = np.bincount(cosine[HEADS:].argmax(1), minlength=total)
source = int(votes.argmax())
finite = np.where(np.isfinite(cosine), cosine, 0)
top_k = lambda k: np.sort(finite, axis=1)[:, -k:].sum(1)
writers = np.where(np.isfinite(cosine), cosine, -np.inf).max(0)
strong = np.argsort(-writers)[:8]
print("RECIPE best_pairing", " ".join(name(int(i)) for i in np.argsort(-best)[:5]))
print("RECIPE by_source", " ".join(name(int(i)) for i in np.argsort(-cosine[:, source])[:5]))
print("RECIPE sum_top3_sources", " ".join(name(int(i)) for i in np.argsort(-top_k(3))[:5]))
print("RECIPE sum_top8_sources", " ".join(name(int(i)) for i in np.argsort(-top_k(8))[:5]))
print("RECIPE sum_strong_writers", " ".join(name(int(i)) for i in np.argsort(-np.maximum(cosine[:, strong], 0).sum(1))[:5]))
print("source", name(source), "seconds", round(time.time() - start, 1), "peak_GB", round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2))
"""


def main() -> None:
    model = load("medium")
    truth = induction_scores(model)
    per_seed = [seed_scores(model, seed) for seed in _TRUTH_SEEDS[:5]]
    heads = model.config.n_head
    order = np.argsort(-truth.ravel())
    print("truth, top 12: " + ", ".join(f"{int(i) // heads}.{int(i) % heads} {truth.ravel()[i]:.3f}" for i in order[:12]))
    print("the 5th and 6th by seed: " + " ".join(f"{s.ravel()[order[4]]:.3f}/{s.ravel()[order[5]]:.3f}" for s in per_seed))
    print(f"ceiling on mean true score: {truth.ravel()[order[:NAMED]].mean():.3f}, all heads' mean {truth.mean():.3f}")
    print("heads above 0.5: " + str(int((truth > 0.5).sum())))
    disguised = disguise_family(model, SEED)
    if not WEIGHTS.exists():
        save_blocks(disguised, WEIGHTS)
    original = disguised.original_head
    top = {(int(i) // heads, int(i) % heads) for i in order[:NAMED]}
    del model, disguised
    for label, dtype, scaled, centered in (("float32 scale centered", "np.float32", True, True), ("float32 scale", "np.float32", True, False), ("float32 raw", "np.float32", False, False), ("float64 scale centered", "np.float64", True, True)):
        began = time.time()
        output = run_code(TEMPLATE.format(dtype=dtype, scaled=scaled, centered=centered), WEIGHTS)
        print(f"{label}, wall {time.time() - began:.0f} s")
        for line in output.splitlines():
            if not line.startswith("RECIPE"):
                print("  " + line[:200])
                continue
            _, recipe, *picks = line.split()
            real = [(int(p.split(".")[0]), int(original[int(p.split(".")[0])][int(p.split(".")[1])])) for p in picks]
            values = [truth[pick] for pick in real]
            print(f"  {recipe}: {sum(pick in top for pick in real)} of {NAMED}, mean true score {np.mean(values):.3f}, picks {real}")
    WEIGHTS.unlink()
    print(json.dumps({"seed": SEED}))


if __name__ == "__main__":
    main()
