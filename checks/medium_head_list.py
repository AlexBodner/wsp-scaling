"""Rescores Astra's edit on GPT-2 medium and runs the simple edit on its five heads, the top 5 by attention and the best-pairing 5. Makes no Astra calls."""
import json
from pathlib import Path

import numpy as np
import torch

from checks.edit_demo import astra_edit, head_list_edit
from scripts.run_write_medium import ATTENTION_TOP, BEST_PAIRING, RUN_SEED, fresh, scores
from wsp.family import disguise_family, load

ROOT = Path(__file__).parent.parent
RUN = ROOT / "runs/write-medium__live__20261104T072216Z"
WIDTH = 1024
# Astra's edit sets this itself, and this float32 model's scores move by up to 0.6 points with the thread count.
THREADS = 4


def edited(disguised, edit: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {**disguised.tensors, **edit}


def main() -> None:
    torch.set_num_threads(THREADS)
    model = load("medium")
    disguised = disguise_family(model, RUN_SEED)
    del model
    if json.loads((RUN / "disguise_map.json").read_text()) != disguised.original_head.tolist():
        raise ValueError("this scramble isn't the run's")
    base = scores(fresh(disguised.tensors))
    print(f"unedited: repeat accuracy {base[0]:.3f}, text loss {base[1]:.3f}", flush=True)
    repeat, loss = scores(fresh(edited(disguised, astra_edit(disguised, RUN))))
    print(f"Astra's edit: repeat accuracy {repeat:.3f}, rise {loss - base[1]:.3f}", flush=True)
    astra = [tuple(head) for head in json.loads((RUN / "result.json").read_text())["answer"]]
    scrambled = lambda real: [(layer, int(np.where(disguised.original_head[layer] == head)[0][0])) for layer, head in real]
    for label, heads in (("Astra's five", astra), ("top 5 by attention", scrambled(ATTENTION_TOP)), ("best-pairing 5", scrambled(BEST_PAIRING))):
        repeat, loss = scores(fresh(edited(disguised, head_list_edit(disguised, heads, WIDTH))))
        print(f"simple edit, {label}: repeat accuracy {repeat:.3f}, rise {loss - base[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
