"""Splits run 2's edit into its key changes and its output changes, and scores each alone. Run 1 changed keys only. Makes no Astra calls."""
import torch

from checks.edit_demo import ROOT, RUNS, astra_edit
from wsp.disguise import disguise, model_from
from wsp.network import TokenShuffle
from wsp.write_task import apply_edit, repeat_tokens, scores, text_tokens


def main() -> None:
    shuffle = TokenShuffle()
    folder, seed = RUNS["run 2"]
    disguised = disguise(seed, shuffle)
    edit = astra_edit(disguised, ROOT / "runs" / folder)
    repeats, text = repeat_tokens(shuffle), text_tokens(shuffle)
    arms = {
        "unedited": {},
        "whole edit": edit,
        "keys only": {name: value for name, value in edit.items() if ".qkv." in name},
        "outputs only": {name: value for name, value in edit.items() if ".out." in name},
    }
    for name, tensors in arms.items():
        with torch.no_grad():
            repeat, loss = scores(model_from(apply_edit(disguised, tensors)), repeats, text)
        print(f"{name}: repeat accuracy {repeat:.3f}, text loss {loss:.3f}")


if __name__ == "__main__":
    main()
