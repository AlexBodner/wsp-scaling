"""Reruns the statistic both Python runs found, with no layer rule, to test whether it produces their picks.

For a later head with query-key matrix QK^T and an earlier head with value-output matrix VO, the score is
their dot product: how well what the earlier head writes matches what the later head compares.
"""
import numpy as np

from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH, disguise
from wsp.network import TokenShuffle, real_network
from wsp.reading import disguised_scores, head_scores

SEEDS = (1004, 1005, 1006, 1007, 1008)
NAMED = 5


def head_matrices(tensors):
    """Returns QK^T and VO for every head, with the layer norm's scale and mean removal included."""
    query_key, value_output = [], []
    for layer in range(LAYERS):
        scale = tensors[f"L{layer}.norm1.scale"][:, None]
        weight, output = tensors[f"L{layer}.qkv.W"], tensors[f"L{layer}.out.W"]
        for head in range(HEADS):
            start = head * HEAD_SIZE
            q, k, v = (scale * weight[:, offset + start : offset + start + HEAD_SIZE] for offset in (0, WIDTH, 2 * WIDTH))
            q, k, v = q - q.mean(0), k - k.mean(0), v - v.mean(0)
            o = output[start : start + HEAD_SIZE]
            o = o - o.mean(1, keepdims=True)
            query_key.append((q @ k.T).ravel())
            value_output.append((v @ o).ravel())
    return np.stack(query_key), np.stack(value_output)


shuffle = TokenShuffle()
real = real_network(shuffle).double()
real.set_attn_implementation("eager")
real_scores = head_scores(real)
for seed in SEEDS:
    disguised = disguise(seed, shuffle)
    truth = disguised_scores(real_scores, disguised)
    top_five = set(np.argsort(-truth.ravel())[:NAMED].tolist())
    query_key, value_output = head_matrices(disguised.tensors)
    raw = query_key @ value_output.T
    cosine = raw / (np.linalg.norm(query_key, axis=1, keepdims=True) * np.linalg.norm(value_output, axis=1)[None, :])
    earlier = np.arange(144)[:, None] // HEADS > np.arange(144)[None, :] // HEADS
    cosine = np.where(earlier, cosine, -np.inf)
    # The source is the earlier head that most later heads pair with best, found with no layer rule.
    votes = np.bincount(np.argmax(cosine[HEADS:], axis=1), minlength=144)
    source = int(np.argmax(votes))
    by_source = cosine[:, source]
    picks = set(np.argsort(-by_source)[:NAMED].tolist())
    original = int(disguised.original_head[source // HEADS][source % HEADS])
    best_any = set(np.argsort(-cosine.max(axis=1))[:NAMED].tolist())
    print(f"seed {seed}: source {source // HEADS}.{source % HEADS} (GPT-2 head {source // HEADS}.{original}), "
          f"best source's top 5 hit {len(picks & top_five)} of 5, each head's best pairing hits {len(best_any & top_five)} of 5")
