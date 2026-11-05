"""Saves head 6.9's search map before and after run 2's edit, in earlier head 4.11's directions, for the five top heads' cosines too. Makes no Astra calls and doesn't run the model."""
import json

import numpy as np

from checks.edit_demo import ROOT, RUNS, astra_edit
from checks.edit_method_data import SHOWN, SOURCE, parts
from wsp.disguise import disguise
from wsp.network import TokenShuffle

SHOWN_HEAD = (6, 9)
TOP_FIVE = ((5, 5), (6, 9), (7, 10), (5, 1), (7, 2))
OUTPUT = ROOT / "checks" / "outputs" / "edit_after_map.json"


def main() -> None:
    folder, seed = RUNS["run 2"]
    disguised = disguise(seed, TokenShuffle())
    scrambled = lambda real: (real[0], int(np.where(disguised.original_head[real[0]] == real[1])[0][0]))
    before = {name: values.astype(np.float32) for name, values in disguised.tensors.items()}
    after = {**before, **astra_edit(disguised, ROOT / "runs" / folder)}
    _, _, value, output = parts(before, *scrambled(SOURCE))
    source_map = value @ output
    left, _, right = np.linalg.svd(source_map, full_matrices=False)
    cosine = lambda search: float((search * source_map).sum() / (np.linalg.norm(search) * np.linalg.norm(source_map)))

    def search(tensors: dict, real: tuple[int, int]) -> np.ndarray:
        query, key, _, _ = parts(tensors, *scrambled(real))
        return query @ key.T

    data = {"source": f"{SOURCE[0]}.{SOURCE[1]}", "head": f"{SHOWN_HEAD[0]}.{SHOWN_HEAD[1]}", "cosines": {}, "kept": {}}
    for real in TOP_FIVE:
        pair = [round(cosine(search(tensors, real)), 3) for tensors in (before, after)]
        data["cosines"][f"{real[0]}.{real[1]}"] = pair
        kept = float(np.linalg.norm(search(after, real)) / np.linalg.norm(search(before, real)))
        data["kept"][f"{real[0]}.{real[1]}"] = round(kept, 3)
        print(f"{real[0]}.{real[1]}: cosine with {data['source']} before {pair[0]}, after {pair[1]}; size of its whole search map kept {kept:.3f}")
    for name, tensors in (("before", before), ("after", after)):
        data[name] = (left[:, :SHOWN].T @ search(tensors, SHOWN_HEAD) @ right[:SHOWN].T).round(3).tolist()
        print(f"{data['head']} {name}: mean diagonal {np.diag(np.array(data[name])).mean():.3f}, largest value {np.abs(np.array(data[name])).max():.3f}")
    OUTPUT.write_text(json.dumps(data))


if __name__ == "__main__":
    main()
