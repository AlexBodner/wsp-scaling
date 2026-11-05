"""Lists which heads each of Astra's two GPT-2 edits changes, in which parts, and by how much. Makes no Astra calls and doesn't run the model."""
import numpy as np

from checks.edit_demo import ROOT, RUNS, astra_edit
from checks.edit_method_data import parts
from wsp.disguise import HEAD_SIZE, WIDTH, disguise
from wsp.network import TokenShuffle

LAYERS, HEADS = 12, 12
PART_OFFSETS = {"query": 0, "key": WIDTH, "value": 2 * WIDTH}


def relative_change(before: np.ndarray, after: np.ndarray) -> float:
    return float(np.linalg.norm(after - before) / np.linalg.norm(before))


def main() -> None:
    shuffle = TokenShuffle()
    for run, (folder, seed) in RUNS.items():
        disguised = disguise(seed, shuffle)
        original = {name: values.astype(np.float32) for name, values in disguised.tensors.items()}
        changed = {**original, **astra_edit(disguised, ROOT / "runs" / folder)}
        print(f"\n{run}, {folder}")
        for layer in range(LAYERS):
            for head in range(HEADS):
                columns = lambda tensors, offset: tensors[f"L{layer}.qkv.W"][:, offset + head * HEAD_SIZE : offset + (head + 1) * HEAD_SIZE]
                rows = lambda tensors: tensors[f"L{layer}.out.W"][head * HEAD_SIZE : (head + 1) * HEAD_SIZE]
                touched = [name for name, offset in PART_OFFSETS.items() if not np.array_equal(columns(original, offset), columns(changed, offset))]
                touched += ["output"] if not np.array_equal(rows(original), rows(changed)) else []
                if touched:
                    _, _, value, output = parts(original, layer, head)
                    _, _, new_value, new_output = parts(changed, layer, head)
                    print(f"real {layer}.{disguised.original_head[layer][head]}: changed {touched}; key weights move {100 * relative_change(columns(original, WIDTH), columns(changed, WIDTH)):.0f}%, output weights move {100 * relative_change(rows(original), rows(changed)):.0f}%, VO moves {100 * relative_change(value @ output, new_value @ new_output):.0f}%", flush=True)


if __name__ == "__main__":
    main()
