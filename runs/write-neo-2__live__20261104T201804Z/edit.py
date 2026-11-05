import numpy as np
import torch

torch.set_num_threads(4)

def tensor(name):
    return torch.tensor(np.asarray(W[name]).copy(), dtype=torch.float64)

def effective_factors(layer, head):
    packed = tensor(f"L{layer}.qkv.W")
    scale = tensor(f"L{layer}.norm1.scale")
    start, stop = 64 * head, 64 * (head + 1)
    factors = []
    for offset in (0, 768, 1536):
        f = scale[:, None] * packed[:, offset + start:offset + stop]
        factors.append(f - f.mean(0, keepdim=True))
    return factors

targets = [(4, 9), (5, 0), (5, 8), (5, 11), (7, 8)]
readers = [
    (6, 0), (6, 5), (6, 7), (6, 8),
    (8, 1), (8, 3), (8, 5), (8, 6), (8, 7), (8, 11),
]

reader_maps = {}
for layer, head in readers:
    q, k, _ = effective_factors(layer, head)
    reader_maps[(layer, head)] = q @ k.T

updates = {}
bias_deltas = {}

for layer, head in targets:
    _, _, value = effective_factors(layer, head)
    sl = slice(64 * head, 64 * (head + 1))
    out_key = f"L{layer}.out.W"
    original_out = tensor(out_key)[sl]
    output_mean = original_out.mean(1, keepdim=True)
    centered_out = original_out - output_mean

    # Factor the effective OV matrix in residual coordinates.
    U, R = torch.linalg.qr(value, mode="reduced")
    C = R @ centered_out
    left, _, _ = torch.linalg.svd(C, full_matrices=False)
    J, T = left[:, :4], left[:, 4:]
    B0, B1 = U @ J, U @ T
    D0, D1 = J.T @ C, T.T @ C
    original_map = U @ C

    # Preserve the leading four components and cancel the remaining
    # symmetric map within the value-input subspace.
    small = D1 @ B1
    cross = D1 @ B0 + (D0 @ B1).T
    cleaned_tail = (
        D1
        - ((small + small.T) / 2) @ B1.T
        - cross @ B0.T
    )
    cleaned = J @ D0 + T @ cleaned_tail

    selected = []
    for (reader_layer, _), attention_map in reader_maps.items():
        if reader_layer <= layer:
            continue
        cosine = (
            (attention_map * original_map).sum()
            / (attention_map.norm() * original_map.norm())
        )
        if float(cosine) > 0.05:
            selected.append(attention_map)

    # Orthogonal projection onto corrections that preserve both
    # the leading components and the symmetric-map constraints.
    def allowed_tail(x):
        x = x - (x @ B0) @ B0.T
        m = x @ B1
        return x - ((m + m.T) / 2) @ B1.T

    candidate = cleaned
    if selected:
        projected = [allowed_tail(B1.T @ a) for a in selected]
        matrix = torch.stack([x.flatten() for x in projected])
        scales = matrix.norm(dim=1)
        matrix = matrix / scales[:, None]

        cleaned_map = U @ cleaned
        rhs = torch.stack([
            (a * cleaned_map).sum() for a in selected
        ]) / scales

        gram = matrix @ matrix.T
        coefficients = torch.linalg.pinv(gram, rtol=1e-12) @ rhs
        correction = (coefficients @ matrix).reshape_as(D1)
        candidate = cleaned - T @ correction

    new_out = torch.linalg.solve(R, candidate) + output_mean
    new_array = new_out.numpy().astype(np.asarray(W[out_key]).dtype)

    # Compensate the affine value contribution using the actual
    # rounded replacement weights.
    packed = tensor(f"L{layer}.qkv.W")
    value_raw = packed[:, 1536 + sl.start:1536 + sl.stop]
    value_constant = (
        tensor(f"L{layer}.norm1.shift") @ value_raw
        + tensor(f"L{layer}.qkv.b")[1536 + sl.start:1536 + sl.stop]
    )
    rounded_out = torch.tensor(new_array, dtype=torch.float64)
    compensation = value_constant @ (original_out - rounded_out)

    if out_key not in updates:
        updates[out_key] = np.asarray(W[out_key]).copy()
    updates[out_key][sl] = new_array

    if layer not in bias_deltas:
        bias_deltas[layer] = torch.zeros(768, dtype=torch.float64)
    bias_deltas[layer] += compensation

for layer, delta in bias_deltas.items():
    key = f"L{layer}.out.b"
    updates[key] = (tensor(key) + delta).numpy().astype(
        np.asarray(W[key]).dtype
    )

for key, replacement in updates.items():
    assert replacement.shape == W[key].shape
    assert np.isfinite(replacement).all()

for key, replacement in updates.items():
    W[key][...] = replacement

_original = np.load("blocks.npz")
_changed = {name: W[name].astype(np.float32) for name in W if not np.array_equal(W[name], _original[name])}
np.savez('edited.npz', **_changed)
print("changed", len(_changed), "tensors,", sum(int((_changed[name] != _original[name]).sum()) for name in _changed), "values")
