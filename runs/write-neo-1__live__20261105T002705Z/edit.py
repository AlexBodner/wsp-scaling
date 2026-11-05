import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(4)

sources = [(4, 6), (5, 5), (5, 7), (5, 10), (7, 9)]
targets = [
    (6, 0), (6, 1), (6, 2), (6, 8),
    (8, 0), (8, 1), (8, 2), (8, 6), (8, 7), (8, 11)
]
heads = {}

# Construct coordinate-invariant effective weight maps.
for layer, head in sorted(set(sources + targets)):
    scale = W[f"L{layer}.norm1.scale"].astype(np.float64)
    shift = W[f"L{layer}.norm1.shift"].astype(np.float64)
    a = W[f"L{layer}.qkv.W"].astype(np.float64)
    b = W[f"L{layer}.qkv.b"].astype(np.float64)

    parts = []
    for offset in (0, 768, 1536):
        sl = slice(offset + 64*head, offset + 64*(head+1))
        matrix = scale[:, None] * a[:, sl]
        matrix -= matrix.mean(axis=0, keepdims=True)
        parts.append(matrix)
    q, k, v = parts

    rows = slice(64*head, 64*(head+1))
    raw_o = W[f"L{layer}.out.W"][rows].astype(np.float64)
    o = raw_o - raw_o.mean(axis=1, keepdims=True)
    vs = slice(1536 + 64*head, 1536 + 64*(head+1))

    heads[layer, head] = dict(
        q=q, k=k, v=v, o=o, raw_o=raw_o,
        vb=shift @ a[:, vs] + b[vs],
        qn=np.sqrt(np.sum((q.T @ q) * (k.T @ k))),
        vn=np.sqrt(np.sum((v.T @ v) * (o @ o.T)))
    )

pending_outputs = []
bias_changes = {}

for source in sources:
    s = heads[source]
    u, r = np.linalg.qr(s["v"], mode="reduced")
    z = r @ s["o"]
    gradients, rhs = [], []

    for target in targets:
        if target[0] <= source[0]:
            continue
        t = heads[target]
        gk = (u.T @ t["q"]) @ t["k"].T
        gq = (u.T @ t["k"]) @ t["q"].T
        denominator = s["vn"] * t["qn"]
        kc = np.sum(z * gk) / denominator
        qc = np.sum(z * gq) / denominator

        if kc < 0.06 or kc - qc < 0.05:
            continue

        # Cancel key-composition traces and preserve query traces.
        for raw, change in ((gk, np.sum(z * gk)), (gq, 0.0)):
            # Restrict changes to preserve the complete map within
            # the source value-input subspace.
            gradient = raw - (raw @ u) @ u.T
            norm = np.linalg.norm(gradient)
            gradients.append((gradient / norm).ravel())
            rhs.append(change / norm)

    g = np.stack(gradients)
    coefficients = (
        np.linalg.pinv(g @ g.T, rcond=1e-10)
        @ np.asarray(rhs)
    )
    dz = (coefficients @ g).reshape(z.shape)

    layer, head = source
    delta_o = -np.linalg.solve(r, dz)
    new_o = (s["raw_o"] + delta_o).astype(
        W[f"L{layer}.out.W"].dtype
    )
    assert np.isfinite(new_o).all()
    actual_delta = new_o.astype(np.float64) - s["raw_o"]
    pending_outputs.append((layer, head, new_o))

    # Compensate the constant contribution using the rounded edit.
    bias_changes.setdefault(layer, np.zeros(768, dtype=np.float64))
    bias_changes[layer] -= s["vb"] @ actual_delta

pending_biases = {}
for layer, correction in bias_changes.items():
    old_b = W[f"L{layer}.out.b"]
    new_b = (old_b.astype(np.float64) + correction).astype(old_b.dtype)
    assert np.isfinite(new_b).all()
    pending_biases[layer] = new_b

for layer, head, new_o in pending_outputs:
    W[f"L{layer}.out.W"][64*head:64*(head+1), :] = new_o
for layer, new_b in pending_biases.items():
    W[f"L{layer}.out.b"][...] = new_b

_original = np.load("blocks.npz")
_changed = {name: W[name].astype(np.float32) for name in W if not np.array_equal(W[name], _original[name])}
np.savez('edited.npz', **_changed)
print("changed", len(_changed), "tensors,", sum(int((_changed[name] != _original[name]).sum()) for name in _changed), "values")
