import numpy as np
from threadpoolctl import threadpool_limits

threadpool_limits(limits=4)

D, S = 768, 64
sources = [(4, 6), (5, 5), (5, 7), (5, 10), (7, 9)]

def center(a):
    a = np.asarray(a, dtype=np.float64)
    return a - a.mean(axis=0, keepdims=True)

# Take all diagnostic quantities from the original weights.
Q, K, V, O, an, bn = [], [], [], [], [], []
for l in range(12):
    m = W[f"L{l}.qkv.W"].astype(np.float64)
    scale = W[f"L{l}.norm1.scale"].astype(np.float64)[:, None]
    for h in range(12):
        q, k, v = [
            center(scale * m[:, offset+h*S:offset+(h+1)*S])
            for offset in (0, D, 2*D)
        ]
        o = center(W[f"L{l}.out.W"][h*S:(h+1)*S].T)
        Q.append(q)
        K.append(k)
        V.append(v)
        O.append(o)
        an.append(np.sqrt(np.sum((q.T @ q) * (k.T @ k))))
        bn.append(np.sqrt(np.sum((v.T @ v) * (o.T @ o))))

early = sum(Q[h] @ K[h].T for h in (0, 2, 4, 6, 11))
ev, evec = np.linalg.eigh((early + early.T) * 0.5)
early = (evec * np.maximum(ev, 0)) @ evec.T
down = center(W["L0.down.W"].T)
metrics = [np.eye(D), early, down @ down.T]
for C in metrics:
    C *= D / np.trace(C)

updates = []
bias_updates = {}

for l, h in sources:
    p = 12*l + h
    u, rv = np.linalg.qr(V[p], mode="reduced")
    coeff = rv @ O[p].T

    selected = []
    protected_queries = []
    for r in range((l+1)*12, 144):
        gk = (u.T @ Q[r]) @ K[r].T
        gq = (u.T @ K[r]) @ Q[r].T
        denominator = an[r] * bn[p]
        key = np.sum(gk * coeff) / denominator
        query = np.sum(gq * coeff) / denominator

        if key > 0.08 and key-query > 0.04:
            selected.append(r)
        if query > 0.08:
            protected_queries.append(gq)

    gradients, rhs = [], []

    # Cancel matching traces under three weight-derived metrics.
    for C in metrics:
        uc = u.T @ C
        for r in selected:
            g = (uc @ Q[r]) @ K[r].T
            gradients.append(g)
            rhs.append(-np.sum(g * coeff))

    # Preserve direct-copy traces and strong query compositions.
    for C in metrics:
        gradients.append(u.T @ C)
        rhs.append(0.0)
    for g in protected_queries:
        gradients.append(g)
        rhs.append(0.0)

    flat = np.stack(gradients).reshape(len(gradients), -1)
    norms = np.linalg.norm(flat, axis=1)
    flat /= norms[:, None]
    rhs = np.asarray(rhs, dtype=np.float64) / norms

    # Minimum Frobenius-norm change to the effective output map.
    gram = flat @ flat.T
    multipliers = np.linalg.pinv(gram, rcond=1e-10) @ rhs
    delta = (multipliers @ flat).reshape(S, D)
    delta -= delta.mean(axis=1, keepdims=True)
    delta_o = np.linalg.solve(rv, delta)

    sl = slice(h*S, (h+1)*S)
    vs = slice(2*D+h*S, 2*D+(h+1)*S)
    name = f"L{l}.out.W"
    original = W[name][sl].astype(np.float64)
    proposed = (original + delta_o).astype(W[name].dtype)
    actual_delta = proposed.astype(np.float64) - original

    # Compensate the constant value contribution using the actual
    # rounded output-matrix update.
    value_offset = (
        W[f"L{l}.norm1.shift"].astype(np.float64)
        @ W[f"L{l}.qkv.W"][:, vs].astype(np.float64)
        + W[f"L{l}.qkv.b"][vs].astype(np.float64)
    )
    bias_name = f"L{l}.out.b"
    if bias_name not in bias_updates:
        bias_updates[bias_name] = W[bias_name].astype(np.float64).copy()
    bias_updates[bias_name] -= value_offset @ actual_delta

    assert np.isfinite(proposed).all()
    updates.append((name, sl, proposed))

for name, value in bias_updates.items():
    assert np.isfinite(value).all()

for name, sl, proposed in updates:
    W[name][sl, :] = proposed
for name, value in bias_updates.items():
    W[name][...] = value.astype(W[name].dtype)

_original = np.load("blocks.npz")
_changed = {name: W[name].astype(np.float32) for name in W if not np.array_equal(W[name], _original[name])}
np.savez('edited.npz', **_changed)
print("changed", len(_changed), "tensors,", sum(int((_changed[name] != _original[name]).sum()) for name in _changed), "values")
