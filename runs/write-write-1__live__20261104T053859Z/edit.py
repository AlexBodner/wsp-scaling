import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

threadpool_limits(limits=4)

targets = [(5, 0), (5, 1), (5, 6), (6, 4), (7, 8)]

source_O = W['L4.out.W'][384:448].astype(np.float64)
source_O -= source_O.mean(axis=1, keepdims=True)
R = np.linalg.qr(source_O.T, mode='reduced')[0]

# Protect directions derived from earlier MLP output matrices.
write_gram = np.zeros((768, 768), dtype=np.float64)
protected = {}
for layer in range(7):
    D = W[f'L{layer}.down.W'].astype(np.float64)
    D -= D.mean(axis=1, keepdims=True)
    gram = D.T @ D
    write_gram += gram / np.trace(gram)
    if layer + 1 in (5, 6, 7):
        _, C = eigh(write_gram, subset_by_index=[736, 767])
        protected[layer + 1] = C

edits = []
for layer, head in targets:
    name = f'L{layer}'
    g = W[name + '.norm1.scale'].astype(np.float64)
    shift = W[name + '.norm1.shift'].astype(np.float64)
    a, b = 64 * head, 64 * (head + 1)

    qraw = W[name + '.qkv.W'][:, a:b].astype(np.float64)
    kraw = W[name + '.qkv.W'][:, 768+a:768+b].astype(np.float64)
    qbias = shift @ qraw + W[name + '.qkv.b'][a:b]
    kbias = W[name + '.qkv.b'][768+a:768+b].astype(np.float64)

    Q = g[:, None] * qraw
    Q -= Q.mean(axis=0, keepdims=True)
    K = g[:, None] * kraw
    K -= K.mean(axis=0, keepdims=True)

    U, T = np.linalg.qr(Q, mode='reduced')
    Kc = K @ T.T
    qc = np.linalg.solve(T.T, qbias)
    projector = np.eye(64) - np.outer(qc, qc) / (qc @ qc)
    C = protected[layer]

    # Remove source coupling while preserving the query-bias term
    # and the protected key directions.
    MR = R - C @ (C.T @ R)
    G = R.T @ MR
    delta = -MR @ np.linalg.solve(G, R.T @ Kc @ projector)

    # Preserve the trace of the effective QK matrix.
    J = U @ projector
    J -= C @ (C.T @ J)
    Jnull = J - MR @ np.linalg.solve(G, R.T @ J)
    delta -= Jnull * (np.sum(delta * U) / np.sum(Jnull * Jnull))

    # Convert back to stored key coordinates.
    delta_K = np.linalg.solve(T, delta.T).T
    invg = 1.0 / g
    delta_raw = delta_K * invg[:, None]
    lift = -(invg @ delta_raw) / (invg @ invg)
    delta_raw += invg[:, None] * lift[None, :]

    new_raw = (kraw + delta_raw).astype(np.float32)
    stored_delta = new_raw.astype(np.float64) - kraw
    new_bias = (kbias - shift @ stored_delta).astype(np.float32)

    assert np.isfinite(new_raw).all()
    assert np.isfinite(new_bias).all()
    edits.append((name, a, b, new_raw, new_bias))

for name, a, b, new_raw, new_bias in edits:
    W[name + '.qkv.W'][:, 768+a:768+b] = new_raw
    W[name + '.qkv.b'][768+a:768+b] = new_bias

_original = np.load("blocks.npz")
_changed = {name: W[name].astype(np.float32) for name in W if not np.array_equal(W[name], _original[name])}
np.savez('edited.npz', **_changed)
print("changed", len(_changed), "tensors,", sum(int((_changed[name] != _original[name]).sum()) for name in _changed), "values")
