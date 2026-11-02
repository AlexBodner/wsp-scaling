"""Checks whether typed weights look copied from real GPT-2."""

import numpy as np
import torch

RANDOM_POSITION_TRIALS = 200
GUARD_SEED = 3


@torch.no_grad()
def compare_with_real(written: dict[str, torch.Tensor], real: dict[str, torch.Tensor]) -> dict:
    """Compares nonzero typed values with real GPT-2 at the same positions.

    `real` must already use the shuffled token IDs, so that positions line up. The reference is the
    same typed values against real weights at random positions in the same tensors.
    """
    rng = np.random.default_rng(GUARD_SEED)
    names = [name for name, tensor in written.items() if tensor.count_nonzero() > 0]
    if not names:
        return {"nonzero_values": 0}
    positions = {name: written[name].nonzero(as_tuple=True) for name in names}
    typed_values = np.concatenate([written[name][positions[name]].numpy() for name in names])
    real_values = np.concatenate([real[name][positions[name]].numpy() for name in names])
    trials = []
    for _ in range(RANDOM_POSITION_TRIALS):
        random_values = [
            real[name].view(-1)[rng.integers(real[name].numel(), size=positions[name][0].numel())].numpy() for name in names
        ]
        trials.append(_correlation(typed_values, np.concatenate(random_values)))
    near_exact = np.abs(typed_values - real_values) < 1e-3
    return {
        "nonzero_values": len(typed_values),
        "correlation_with_real": _correlation(typed_values, real_values),
        "correlation_random_positions_95pct": [float(x) for x in np.percentile(trials, [2.5, 97.5])],
        "share_within_0.001_of_real": float(near_exact.mean()),
    }


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    if first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])
