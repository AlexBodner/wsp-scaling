"""Saves the comparison Astra's code made, as a picture: each head's search map seen in an earlier head's output directions, and which weights the edit changes. Makes no Astra calls."""
import json

import numpy as np

from checks.edit_demo import ROOT, RUNS, astra_edit
from wsp.disguise import HEAD_SIZE, WIDTH, disguise
from wsp.network import TokenShuffle

SOURCE = (4, 11)
SHOWN = 32
HEADS = {"matches": (6, 9), "doesn't match": (6, 3), "hard case": (5, 8)}
OUTPUT = ROOT / "checks" / "outputs" / "edit_method_data.json"


def parts(tensors: dict, layer: int, head: int) -> tuple[np.ndarray, ...]:
    """Returns Q, K, V and O, with the layer norm's scale and mean removal, as Astra's code computes them."""
    scale = tensors[f"L{layer}.norm1.scale"][:, None]
    start = head * HEAD_SIZE
    weight = tensors[f"L{layer}.qkv.W"]
    q, k, v = (scale * weight[:, offset + start : offset + start + HEAD_SIZE] for offset in (0, WIDTH, 2 * WIDTH))
    o = tensors[f"L{layer}.out.W"][start : start + HEAD_SIZE]
    return q - q.mean(0), k - k.mean(0), v - v.mean(0), o - o.mean(1, keepdims=True)


def main() -> None:
    folder, seed = RUNS["run 2"]
    disguised = disguise(seed, TokenShuffle())
    scrambled = lambda real: (real[0], int(np.where(disguised.original_head[real[0]] == real[1])[0][0]))
    _, _, v, o = parts(disguised.tensors, *scrambled(SOURCE))
    source_map = v @ o
    left, strengths, right = np.linalg.svd(source_map, full_matrices=False)
    data = {"source": f"{SOURCE[0]}.{SOURCE[1]}", "writes": (left[:, :SHOWN].T @ source_map @ right[:SHOWN].T).round(3).tolist(), "heads": {}}
    for label, real in HEADS.items():
        q, k, _, _ = parts(disguised.tensors, *scrambled(real))
        search = q @ k.T
        cosine = float((search * source_map).sum() / (np.linalg.norm(search) * np.linalg.norm(source_map)))
        seen = left[:, :SHOWN].T @ search @ right[:SHOWN].T
        data["heads"][label] = {"head": f"{real[0]}.{real[1]}", "cosine": round(cosine, 3), "seen": seen.round(3).tolist(), "diagonal_share": round(float((np.diag(seen) ** 2).sum() / (seen**2).sum()), 3)}
        print(label, real, "cosine", round(cosine, 3), "diagonal share of energy", data["heads"][label]["diagonal_share"], "diagonal mean", round(float(np.diag(seen).mean()), 2), "off-diagonal abs mean", round(float(np.abs(seen - np.diag(np.diag(seen))).mean()), 2))
    edit = astra_edit(disguised, ROOT / "runs" / folder)
    layer, head = scrambled((6, 9))
    original = disguised.tensors[f"L{layer}.qkv.W"].astype(np.float32)
    changed = edit[f"L{layer}.qkv.W"] != original
    start = head * HEAD_SIZE
    print("head 6.9 slot: query changed", int(changed[:, start : start + HEAD_SIZE].sum()), "key changed", int(changed[:, WIDTH + start : WIDTH + start + HEAD_SIZE].sum()), "value changed", int(changed[:, 2 * WIDTH + start : 2 * WIDTH + start + HEAD_SIZE].sum()), "of", WIDTH * HEAD_SIZE, "each; layer columns changed", int(changed.any(0).sum()), "of", changed.shape[1])
    key = original[:, WIDTH + start : WIDTH + start + HEAD_SIZE]
    new_key = edit[f"L{layer}.qkv.W"][:, WIDTH + start : WIDTH + start + HEAD_SIZE]
    data["key_relative_change"] = round(float(np.linalg.norm(new_key - key) / np.linalg.norm(key)), 3)
    print("key relative change", data["key_relative_change"])
    OUTPUT.write_text(json.dumps(data))


if __name__ == "__main__":
    main()
