import numpy as np
import torch

torch.set_num_threads(4)
torch.set_grad_enabled(False)

sources = [(4, 4), (5, 0), (6, 4)]
targets = [(5, 9), (6, 8), (7, 12), (9, 8), (10, 2)]

def tensor(name, index=None):
    a = np.asarray(W[name])
    if index is not None:
        a = a[index]
    return torch.from_numpy(a).double()

def centered_scaled(a, scale):
    v = scale[:, None] * a
    return v - v.mean(0, keepdim=True)

# Derive source-output directions entirely from weight operators.
source_factors = {}
for l, h in sources:
    s = slice(64*h, 64*(h+1))
    vs = slice(2048+64*h, 2048+64*(h+1))
    scale = tensor(f"L{l}.norm1.scale")
    V = centered_scaled(
        tensor(f"L{l}.qkv.W", (slice(None), vs)), scale
    )
    O = tensor(f"L{l}.out.W", (s, slice(None)))
    O -= O.mean(1, keepdim=True)
    ev, eu = torch.linalg.eigh(V.T @ V)
    factor = O.T @ (eu * ev.clamp_min(0).sqrt()[None, :])
    source_factors[(l, h)] = factor / factor.norm()

staged = []
for l, h in targets:
    qs = slice(64*h, 64*(h+1))
    ks = slice(1024+64*h, 1024+64*(h+1))
    name = f"L{l}.qkv.W"
    bname = f"L{l}.qkv.b"
    scale = tensor(f"L{l}.norm1.scale")
    shift = tensor(f"L{l}.norm1.shift")
    rawQ = tensor(name, (slice(None), qs))
    rawK = tensor(name, (slice(None), ks))
    rawb = tensor(bname, ks)

    Q = centered_scaled(rawQ, scale)
    K = centered_scaled(rawK, scale)
    qb = shift @ rawQ + tensor(bname, qs)

    # Orthonormal coordinates for the effective QK operator.
    U, R = torch.linalg.qr(Q, mode="reduced")
    J = K @ R.T
    b = torch.linalg.solve(R.T, qb)
    b /= b.norm()

    def protect(a):
        return a - (a @ b)[:, None] * b[None, :]

    available = [s for s in sources if s[0] < l]
    S, sv, _ = torch.linalg.svd(
        torch.cat([source_factors[s] for s in available], dim=1),
        full_matrices=False
    )
    Jp, Up = protect(J), protect(U)
    SJ, SJp, SUp = S.T @ J, S.T @ Jp, S.T @ Up
    coupling0 = (sv[:, None] * SJ).norm()

    # Minimum operator change with a source-coupling penalty,
    # preserving the affine key preference and direct QK trace.
    def solve(lam, return_delta=False):
        shrink = lam * sv.square() / (1 + lam * sv.square())
        removed = shrink[:, None] * SJp
        denominator = Up.square().sum() - (
            shrink[:, None] * SUp.square()
        ).sum()
        restore = (removed * SUp).sum() / denominator
        remaining = SJ - removed + restore * (
            (1-shrink)[:, None] * SUp
        )
        ratio = float(
            (sv[:, None] * remaining).norm() / coupling0
        )
        if return_delta:
            delta = -S @ removed + restore * (
                Up - S @ (shrink[:, None] * SUp)
            )
            return delta, ratio
        return ratio

    goal = max(0.15, solve(1e12) + 0.005)
    low, high = 0.0, 1.0
    while solve(high) > goal and high < 1e12:
        high *= 4
    for _ in range(48):
        mid = (low + high) / 2
        if solve(mid) > goal:
            low = mid
        else:
            high = mid

    delta, _ = solve(high, True)
    deltaK = torch.linalg.solve(R, delta.T).T
    deltaK -= deltaK.mean(0, keepdim=True)

    # Use the centering freedom to minimize raw weight change.
    invsq = scale.square().reciprocal()
    offset = -(invsq[:, None] * deltaK).sum(0) / invsq.sum()
    raw_delta = (deltaK + offset[None, :]) / scale[:, None]

    editedK = (rawK + raw_delta).float()
    actual_raw_delta = editedK.double() - rawK
    editedb = (rawb - shift @ actual_raw_delta).float()

    # Validate the rounded arrays using weight algebra only.
    actualK = centered_scaled(editedK.double(), scale)
    actual_delta = actualK - K
    actualJ = actualK @ R.T
    actual_coupling = float(
        (sv[:, None] * (S.T @ actualJ)).norm() / coupling0
    )
    affine_error = float(
        (shift @ actual_raw_delta + editedb.double() - rawb).norm()
    )
    preference_error = float(
        (actual_delta @ qb).norm() / (K @ qb).norm()
    )
    direct_error = float(
        (actual_delta * Q).sum().abs() / J.norm()
    )
    conversion_error = float(
        (actual_delta - deltaK).norm() / deltaK.norm()
    )

    assert torch.isfinite(editedK).all()
    assert torch.isfinite(editedb).all()
    assert abs(actual_coupling - goal) < 1e-5
    assert affine_error < 1e-5
    assert preference_error < 1e-5
    assert direct_error < 1e-5
    assert conversion_error < 1e-5
    staged.append((name, bname, ks, editedK, editedb))

# Apply only the five validated key-weight and key-bias edits.
for name, bname, ks, editedK, editedb in staged:
    W[name][:, ks] = editedK.numpy()
    W[bname][ks] = editedb.numpy()

_original = np.load("blocks.npz")
_changed = {name: W[name].astype(np.float32) for name in W if not np.array_equal(W[name], _original[name])}
np.savez('edited.npz', **_changed)
print("changed", len(_changed), "tensors,", sum(int((_changed[name] != _original[name]).sum()) for name in _changed), "values")
