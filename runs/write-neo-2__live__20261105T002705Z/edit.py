import numpy as np
import scipy.linalg as la
from threadpoolctl import threadpool_limits

groups = [
    ((6, 0), [(4,5), (4,9), (5,0), (5,8), (5,11)]),
    ((6, 5), [(4,5), (4,9), (5,0), (5,8), (5,11)]),
    ((6, 7), [(4,5), (4,9), (5,0), (5,8), (5,11)]),
    ((6, 8), [(4,5), (4,9), (5,0), (5,8), (5,11)]),
    ((11, 7), [(0,1), (0,6), (0,7), (0,8), (0,9)]),
]

with threadpool_limits(limits=4):
    needed = set()
    for target, sources in groups:
        needed.add(target)
        needed.update(sources)

    heads = {}
    for l, h in sorted(needed):
        a, b = 64*h, 64*(h+1)
        A = W[f"L{l}.qkv.W"].astype(np.float64)
        g = W[f"L{l}.norm1.scale"].astype(np.float64)
        shift = W[f"L{l}.norm1.shift"].astype(np.float64)
        bias = W[f"L{l}.qkv.b"].astype(np.float64)

        Q, K, V = [
            g[:, None] * A[:, off+a:off+b]
            for off in (0, 768, 1536)
        ]
        for M in (Q, K, V):
            M -= M.mean(axis=0, keepdims=True)

        O = W[f"L{l}.out.W"][a:b].astype(np.float64)
        O -= O.mean(axis=1, keepdims=True)
        heads[l, h] = dict(
            Q=Q, K=K, V=V, O=O,
            q0=shift @ A[:, a:b] + bias[a:b],
        )

    patches = []
    for (l, h), sources in groups:
        t = heads[l, h]
        U, R = la.qr(t["Q"], mode="economic")
        B = R @ t["K"].T

        # Preserve the constant-query contribution.
        r = la.solve_triangular(R.T, t["q0"], lower=True)
        J = np.eye(64) - np.outer(r, r)/(r @ r)

        forward = [
            (U.T @ heads[s]["V"]) @ heads[s]["O"]
            for s in sources
        ]
        reverse = [
            (U.T @ heads[s]["O"].T) @ heads[s]["V"].T
            for s in sources
        ]
        X = np.stack([
            M.ravel()/np.linalg.norm(M) for M in forward
        ], axis=1)

        # Preserve direct QK trace and reverse-composition traces.
        preserved = [(J @ U.T).ravel()]
        preserved.extend((J @ M).ravel() for M in reverse)
        basis = np.stack(preserved, axis=1)
        basis /= np.linalg.norm(basis, axis=0, keepdims=True)
        basis = la.orth(basis)

        D = np.stack([
            (J @ X[:, j].reshape(B.shape)).ravel()
            for j in range(X.shape[1])
        ], axis=1)
        D -= basis @ (basis.T @ D)

        coefficients = la.solve(
            D.T @ D, X.T @ B.ravel(), assume_a="pos"
        )
        delta = (D @ coefficients).reshape(B.shape)
        dK = la.solve_triangular(R, delta).T

        # Set targeted composition traces to -15% of their originals.
        strength = 1.15
        ks = slice(768+64*h, 768+64*(h+1))
        g = W[f"L{l}.norm1.scale"].astype(np.float64)
        shift = W[f"L{l}.norm1.shift"].astype(np.float64)
        old = W[f"L{l}.qkv.W"][:, ks].astype(np.float64)
        old_bias = W[f"L{l}.qkv.b"][ks].astype(np.float64)

        # Choose the centering-null offset minimizing raw weight change.
        inverse_square = 1.0/(g*g)
        offset = -(inverse_square @ dK)/inverse_square.sum()
        raw_delta = strength*(dK + offset[None, :])/g[:, None]
        new = (old - raw_delta).astype(np.float32)

        # Preserve the effective key constant after float32 conversion.
        new_bias = (
            old_bias + shift @ (old - new.astype(np.float64))
        ).astype(np.float32)

        assert np.isfinite(new).all() and np.isfinite(new_bias).all()
        patches.append((l, ks, new, new_bias))

    for l, ks, new, new_bias in patches:
        W[f"L{l}.qkv.W"][:, ks] = new
        W[f"L{l}.qkv.b"][ks] = new_bias

_original = np.load("blocks.npz")
_changed = {name: W[name].astype(np.float32) for name in W if not np.array_equal(W[name], _original[name])}
np.savez('edited.npz', **_changed)
print("changed", len(_changed), "tensors,", sum(int((_changed[name] != _original[name]).sum()) for name in _changed), "values")
