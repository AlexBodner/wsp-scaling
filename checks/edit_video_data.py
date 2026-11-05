"""Saves the real measurements behind the edit video: outputs, attention and changed weights, for Astra's run 2. Makes no Astra calls."""
import json

import numpy as np
import torch

from checks.edit_demo import ROOT, RUNS, TOP_FIVE, astra_edit, continuation
from scripts.edit_figure import printed_tables
from wsp.disguise import disguise, model_from
from wsp.network import TokenShuffle
from wsp.write_task import apply_edit

PROMPT = "Context: door chair car door chair"
SOURCE = (4, 11)
EDITED_LAYERS = (5, 6, 7)
OUTPUT = ROOT / "checks" / "outputs" / "edit_video_data.json"


def main() -> None:
    shuffle = TokenShuffle()
    folder, seed = RUNS["run 2"]
    disguised = disguise(seed, shuffle)
    edit = astra_edit(disguised, ROOT / "runs" / folder)
    unedited, edited = model_from(disguised), model_from(apply_edit(disguised, edit))
    scrambled = lambda real: (real[0], int(np.where(disguised.original_head[real[0]] == real[1])[0][0]))
    tokens = shuffle.encode(PROMPT)
    words = [shuffle.decode_token(token) for token in tokens]
    with torch.no_grad():
        before = unedited(torch.tensor([tokens]), output_attentions=True).attentions
        after = edited(torch.tensor([tokens]), output_attentions=True).attentions
    row = lambda attentions, real, position: [round(float(x), 3) for x in attentions[real[0]][0, scrambled(real)[1], position]]
    changed = sum(int((values != disguised.tensors[name].astype(np.float32)).sum()) for name, values in edit.items())
    columns = {str(layer): (edit[f"L{layer}.qkv.W"] != disguised.tensors[f"L{layer}.qkv.W"].astype(np.float32)).any(0).astype(int).tolist() for layer in EDITED_LAYERS}
    data = {
        "prompt": PROMPT, "words": words, "run": "run 2",
        "before_text": continuation(unedited, shuffle, PROMPT), "after_text": continuation(edited, shuffle, PROMPT),
        "before": {f"{a}.{b}": row(before, (a, b), -1) for a, b in TOP_FIVE}, "after": {f"{a}.{b}": row(after, (a, b), -1) for a, b in TOP_FIVE},
        "source_at_each": [row(before, SOURCE, position) for position in range(len(words))],
        "values_changed": changed, "weights": 124439808,
        "keys_changed": sum(int((values != disguised.tensors[name].astype(np.float32)).sum()) for name, values in edit.items() if ".qkv." in name),
        "outputs_changed": sum(int((values != disguised.tensors[name].astype(np.float32)).sum()) for name, values in edit.items() if ".out." in name),
        "table": [[f"{a}.{b}", value] for (a, b), value in printed_tables()["Run 2"]],
        "edited_heads": sorted(f"{a}.{b}" for a, b in map(lambda pair: (pair[0], int(disguised.original_head[pair[0]][pair[1]])), json.loads((ROOT / "runs" / folder / "write_score.json").read_text())["touched_heads"])),
        "layer_columns_changed": columns,
    }
    OUTPUT.write_text(json.dumps(data))
    print(words, repr(data["before_text"]), repr(data["after_text"]), changed, changed / data["weights"], data["edited_heads"], {layer: sum(values) for layer, values in columns.items()})
    for key in ("before", "after"):
        print(key, {name: words[int(np.argmax(values))].strip() + f" {max(values):.2f}" for name, values in data[key].items()})
    print("4.11 at each word:", [(words[i].strip(), words[int(np.argmax(values))].strip(), max(values)) for i, values in enumerate(data["source_at_each"])])


if __name__ == "__main__":
    main()
