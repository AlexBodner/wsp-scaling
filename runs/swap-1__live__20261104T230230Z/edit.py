import numpy as np
from threadpoolctl import threadpool_limits

threadpool_limits(limits=4)

def head(l, h):
    scale = W[f"L{l}.norm1.scale"].astype(np.float64)
    A = scale[:, None] * W[f"L{l}.qkv.W"].astype(np.float64)
    A -= A.mean(axis=0, keepdims=True)
    Q, K, V = [
        A[:, offset + 64*h:offset + 64*(h+1)]
        for offset in (0, 768, 1536)
    ]
    O = W[f"L{l}.out.W"][64*h:64*(h+1)].astype(np.float64)
    return Q, K, V, O

def low_svd(A, B):
    ua, ra = np.linalg.qr(A, mode="reduced")
    ub, rb = np.linalg.qr(B.T, mode="reduced")
    u, s, vt = np.linalg.svd(ra @ rb.T, full_matrices=False)
    return ua @ u, s, vt @ ub.T

def basis(A):
    u, s, _ = np.linalg.svd(A, full_matrices=False)
    return u[:, s > s[0] * 1e-9]

targets = [(5, 7), (5, 11), (6, 7), (7, 2), (7, 9)]
rank, gain = 24, 0.75
one = np.ones((768, 1)) / np.sqrt(768)

# Recover the shared source map using only weight products.
_, _, Vs, Os = head(4, 2)
Os -= Os.mean(axis=1, keepdims=True)
U, ss, Rt = low_svd(Vs, Os)
B = (U * ss) @ Rt

replacements = []
bias_changes = {}

for l, h in targets:
    Q, K, V, O = head(l, h)
    M = V @ O
    original_norm = np.linalg.norm(M)

    # Decode source-written directions while excluding these read spaces.
    protected = basis(np.concatenate([one, U, Q, V], axis=1))
    N = Rt.T - protected @ (protected.T @ Rt.T)
    ru, rs, rvt = np.linalg.svd(B @ N, full_matrices=False)
    inverse = (
        rvt.T * (rs / (rs**2 + (0.15 * rs[0])**2))
    ) @ ru.T

    qu, _, _ = np.linalg.svd(U.T @ Q, full_matrices=False)
    H = U @ qu[:, :rank]
    left = N @ inverse @ H
    left *= gain * original_norm / np.linalg.norm(left)

    # Allocate the remaining output dimensions to the original map.
    Ores = O - (O @ H) @ H.T
    _, _, residual_vt = low_svd(V, Ores)
    J0 = residual_vt[:64-rank].T.copy()
    J0 -= H @ (H.T @ J0)
    J0, _ = np.linalg.qr(J0, mode="reduced")
    J = np.concatenate([H, J0], axis=1)

    new_v = M @ J
    new_v[:, :rank] += left

    scale = W[f"L{l}.norm1.scale"].astype(np.float64)
    shift = W[f"L{l}.norm1.shift"].astype(np.float64)
    vsl = slice(1536 + 64*h, 1536 + 64*(h+1))
    osl = slice(64*h, 64*(h+1))

    old_raw = W[f"L{l}.qkv.W"][:, vsl].astype(np.float64)
    old_bias = W[f"L{l}.qkv.b"][vsl].astype(np.float64)
    old_constant = (shift @ old_raw + old_bias) @ O

    raw_v32 = (new_v / scale[:, None]).astype(np.float32)
    out32 = J.T.astype(np.float32)
    value_b32 = (-shift @ raw_v32.astype(np.float64)).astype(np.float32)

    new_constant = (
        shift @ raw_v32.astype(np.float64)
        + value_b32.astype(np.float64)
    ) @ out32.astype(np.float64)

    if l not in bias_changes:
        bias_changes[l] = np.zeros(768, dtype=np.float64)
    bias_changes[l] += old_constant - new_constant
    replacements.append((l, vsl, osl, raw_v32, value_b32, out32))

for l, vsl, osl, raw_v, value_b, out in replacements:
    W[f"L{l}.qkv.W"][:, vsl] = raw_v
    W[f"L{l}.qkv.b"][vsl] = value_b
    W[f"L{l}.out.W"][osl, :] = out

for l, delta in bias_changes.items():
    W[f"L{l}.out.b"][...] = (
        W[f"L{l}.out.b"].astype(np.float64) + delta
    ).astype(np.float32)

_original = np.load("blocks.npz")
_changed = {name: W[name].astype(np.float32) for name in W if not np.array_equal(W[name], _original[name])}
np.savez('edited.npz', **_changed)
print("changed", len(_changed), "tensors,", sum(int((_changed[name] != _original[name]).sum()) for name in _changed), "values")
