import numpy as np
with np.load('/Users/alexanderbodner/Documents/wsp-scaling/data/reading_python/blocks_1006.npz') as _archive:
    W = {name: _archive[name] for name in _archive.files}
del _archive
scores = []
for layer in range(4, 12):
    weight = W[f"L{layer}.qkv.W"]
    for head in range(12):
        q, k = weight[:, 64 * head : 64 * (head + 1)], weight[:, 768 + 64 * head : 768 + 64 * (head + 1)]
        scores.append((float((q * k).sum()), layer, head))
print("TOP", " ".join(f"{layer}.{head}" for _, layer, head in sorted(scores, reverse=True)[:5]))
