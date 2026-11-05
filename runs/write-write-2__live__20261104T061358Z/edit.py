import numpy as np
from threadpoolctl import threadpool_limits

threadpool_limits(4)

def centered_scaled(a, scale):
    result = np.asarray(a, dtype=np.float64) * scale[:, None]
    return result - result.mean(axis=0, keepdims=True)

# Shared source output subspace, identified from weight products.
source_O = np.array(W["L4.out.W"][704:768],
                    dtype=np.float64, copy=True)
source_O -= source_O.mean(axis=1, keepdims=True)
source_basis, _ = np.linalg.qr(source_O.T, mode="reduced")

targets = [(5, 3), (5, 7), (5, 9), (6, 6), (7, 3), (7, 11)]

for layer, head in targets:
    prefix = f"L{layer}"
    qkv_name = f"{prefix}.qkv.W"
    bias_name = f"{prefix}.qkv.b"
    out_name = f"{prefix}.out.W"
    out_bias_name = f"{prefix}.out.b"

    scale = np.asarray(W[f"{prefix}.norm1.scale"], dtype=np.float64)
    shift = np.asarray(W[f"{prefix}.norm1.shift"], dtype=np.float64)
    raw = np.array(W[qkv_name], dtype=np.float64, copy=True)
    bias = np.array(W[bias_name], dtype=np.float64, copy=True)

    qs = slice(64 * head, 64 * (head + 1))
    ks = slice(768 + 64 * head, 768 + 64 * (head + 1))
    vs = slice(1536 + 64 * head, 1536 + 64 * (head + 1))

    Q = centered_scaled(raw[:, qs], scale)
    K = centered_scaled(raw[:, ks], scale)
    V = centered_scaled(raw[:, vs], scale)
    q0 = shift @ raw[:, qs] + bias[qs]
    v0 = shift @ raw[:, vs] + bias[vs]

    # Remove source-subspace key coupling, except the component needed
    # to preserve the constant-query key preference.
    metric_direction = np.linalg.solve(Q.T @ Q, q0)
    denominator = q0 @ metric_direction
    projected_K = source_basis @ (source_basis.T @ K)
    delta_K = projected_K - np.outer(
        projected_K @ q0, metric_direction
    ) / denominator

    new_K = (raw[:, ks] - delta_K / scale[:, None]).astype(
        W[qkv_name].dtype
    )
    actual_K_delta = raw[:, ks] - new_K.astype(np.float64)
    W[qkv_name][:, ks] = new_K
    W[bias_name][ks] = bias[ks] + shift @ actual_K_delta

    # Remove positive symmetric OV components within the value span.
    original_O = np.array(W[out_name][qs], dtype=np.float64, copy=True)
    centered_O = original_O - original_O.mean(axis=1, keepdims=True)
    U, R = np.linalg.qr(V, mode="reduced")
    compressed = R @ centered_O @ U
    eigenvalues, eigenvectors = np.linalg.eigh(
        (compressed + compressed.T) / 2
    )
    positive = (
        eigenvectors * np.maximum(eigenvalues, 0)
    ) @ eigenvectors.T
    delta_O = np.linalg.solve(R, positive @ U.T)

    new_O = (original_O - delta_O).astype(W[out_name].dtype)
    actual_O_delta = original_O - new_O.astype(np.float64)
    W[out_name][qs] = new_O

    # Compensate the output's affine constant using the actual
    # float32 weight change.
    W[out_bias_name][:] = (
        W[out_bias_name].astype(np.float64) + v0 @ actual_O_delta
    )

_original = np.load("blocks.npz")
_changed = {name: W[name].astype(np.float32) for name in W if not np.array_equal(W[name], _original[name])}
np.savez('edited.npz', **_changed)
print("changed", len(_changed), "tensors,", sum(int((_changed[name] != _original[name]).sum()) for name in _changed), "values")
