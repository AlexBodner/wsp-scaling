import numpy as np
from threadpoolctl import threadpool_limits

with threadpool_limits(limits=4):
    targets = [(5, 10), (6, 7), (5, 9), (7, 10), (7, 6)]
    sources = [(4, 9), (3, 6), (3, 11), (2, 3), (1, 11)]
    probes = [(1, 9), (3, 0), (4, 0)]

    def read_head(layer, head):
        prefix = f"L{layer}"
        scale = W[f"{prefix}.norm1.scale"].astype(np.float64)
        shift = W[f"{prefix}.norm1.shift"].astype(np.float64)
        sl = slice(1536 + 64*head, 1536 + 64*(head+1))
        raw = W[f"{prefix}.qkv.W"][:, sl].astype(np.float64)
        bias = W[f"{prefix}.qkv.b"][sl].astype(np.float64)
        value = scale[:, None] * raw
        value -= value.mean(axis=0, keepdims=True)
        output = W[f"{prefix}.out.W"][
            64*head:64*(head+1)
        ].astype(np.float64)
        return dict(
            V=value, O=output, raw=raw, bias=bias,
            intercept=shift @ raw + bias,
            scale=scale, shift=shift
        )

    def column_basis(matrix):
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        return u[:, s > s[0] * 1e-9]

    heads = {
        p: read_head(*p)
        for p in sorted(set(targets + sources + probes + [(5, 5)]))
    }
    maps = {
        p: a["V"] @ (a["O"] - a["O"].mean(axis=1, keepdims=True))
        for p, a in heads.items()
    }

    # Construct a comparison metric solely from OV matrix spectra.
    metric = np.zeros((768, 768), dtype=np.float64)
    for p in probes:
        matrix = -maps[p]
        eigenvalues, eigenvectors = np.linalg.eigh(
            (matrix + matrix.T) / 2
        )
        keep = eigenvalues > 0.2 * eigenvalues[-1]
        part = (
            eigenvectors[:, keep] * eigenvalues[keep]
        ) @ eigenvectors[:, keep].T
        metric += part / np.trace(part)
    metric *= 768 / np.trace(metric)

    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    token_basis = eigenvectors[
        :, np.argsort(eigenvalues)[::-1][:128]
    ]

    # Protect constant directions identifiable from the parameters.
    constants = [np.ones(768)]
    for layer in range(8):
        prefix = f"L{layer}"
        shift = W[f"{prefix}.norm1.shift"].astype(np.float64)
        values = W[f"{prefix}.qkv.W"][:, 1536:].astype(np.float64)
        biases = W[f"{prefix}.qkv.b"][1536:].astype(np.float64)
        output = W[f"{prefix}.out.W"].astype(np.float64)
        constants.extend([
            (shift @ values + biases) @ output
            + W[f"{prefix}.out.b"].astype(np.float64),
            W[f"{prefix}.down.b"].astype(np.float64),
        ])
    for p in sources + [(5, 5)]:
        constants.append(heads[p]["intercept"] @ heads[p]["O"])

    protected = column_basis(np.column_stack([
        heads[4, 9]["V"], token_basis, *constants
    ]))

    fit_metric = 0.15 * np.eye(768) + 0.85 * metric
    eigenvalues, eigenvectors = np.linalg.eigh(fit_metric)
    metric_root = (
        eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))
    ) @ eigenvectors.T

    # Fit regularized linear decoders of the earlier OV maps.
    decoders = {}
    for later in (False, True):
        source_map = sum(maps[p] for p in sources)
        if later:
            source_map = source_map + maps[5, 5]

        restricted = (
            source_map
            - (source_map @ protected) @ protected.T
        )
        u, s, vt = np.linalg.svd(
            metric_root @ restricted, full_matrices=False
        )
        keep = s > s[0] * 1e-8
        u, s, vt = u[:, keep], s[keep], vt[keep]
        ridge = (0.12 * s[0]) ** 2
        decoder = (
            vt.T * (s / (s*s + ridge))
        ) @ (u.T @ metric_root)
        decoders[later] = source_map, decoder

    updates = []
    for layer, head in targets:
        a = heads[layer, head]
        source_map, decoder = decoders[layer > 5]

        decoded = decoder @ a["V"]
        delta_value = 2.0 * decoded - source_map @ decoded
        intended = a["V"] + delta_value

        assert np.min(np.abs(a["scale"])) > 1e-6
        new_raw = (
            a["raw"] + delta_value / a["scale"][:, None]
        ).astype(np.float32)

        # Compensate using the actual rounded weight change.
        rounded_delta = new_raw.astype(np.float64) - a["raw"]
        new_bias = (
            a["bias"] - a["shift"] @ rounded_delta
        ).astype(np.float32)

        actual_value = a["scale"][:, None] * new_raw.astype(np.float64)
        actual_value -= actual_value.mean(axis=0, keepdims=True)
        actual_intercept = (
            a["shift"] @ new_raw.astype(np.float64)
            + new_bias.astype(np.float64)
        )
        assert np.isfinite(new_raw).all()
        assert np.isfinite(new_bias).all()
        assert (
            np.linalg.norm(actual_value - intended)
            / max(np.linalg.norm(intended), 1e-30)
        ) < 1e-6
        assert np.linalg.norm(
            actual_intercept - a["intercept"]
        ) < 1e-5

        updates.append((layer, head, new_raw, new_bias))

    for layer, head, new_raw, new_bias in updates:
        sl = slice(1536 + 64*head, 1536 + 64*(head+1))
        W[f"L{layer}.qkv.W"][:, sl] = new_raw
        W[f"L{layer}.qkv.b"][sl] = new_bias

_original = np.load("blocks.npz")
_changed = {name: W[name].astype(np.float32) for name in W if not np.array_equal(W[name], _original[name])}
np.savez('edited.npz', **_changed)
print("changed", len(_changed), "tensors,", sum(int((_changed[name] != _original[name]).sum()) for name in _changed), "values")
