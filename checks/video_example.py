"""Chooses the video's example by a rule: measures prompt templates over 300 random lists on GPT-2 small, then takes the first list that is right before and wrong after in both runs. Makes no Astra calls."""
import numpy as np
import torch

from checks.edit_demo import ROOT, RUNS, SEED, WORDS, astra_edit, continuation, head_list_edit
from wsp.disguise import disguise, model_from
from wsp.network import TokenShuffle
from wsp.write_task import apply_edit

LISTS = 300
TEMPLATES = {
    "Context: a b c a b": lambda a, b, c: f"Context: {a} {b} {c} {a} {b}",
    "Context: a, b, c, a, b,": lambda a, b, c: f"Context: {a}, {b}, {c}, {a}, {b},",
    "a b c a b": lambda a, b, c: f"{a} {b} {c} {a} {b}",
    "Words: a b c a b": lambda a, b, c: f"Words: {a} {b} {c} {a} {b}",
}
SHOWN = "Context: a b c a b"


def word_lists() -> list[tuple[str, str, str]]:
    rng = np.random.default_rng(SEED)
    return [tuple(WORDS[i] for i in rng.choice(len(WORDS), 3, replace=False)) for _ in range(LISTS)]


@torch.no_grad()
def tops(model, shuffle: TokenShuffle, prompts: list[str]) -> list[int]:
    return [int(model(input_ids=torch.tensor([shuffle.encode(prompt)])).logits[0, -1].argmax()) for prompt in prompts]


def main() -> None:
    shuffle = TokenShuffle()
    lists = word_lists()
    models = {}
    for run, (folder, seed) in RUNS.items():
        disguised = disguise(seed, shuffle)
        models.setdefault("unedited", model_from(disguised))
        models[f"Astra, {run}"] = model_from(apply_edit(disguised, astra_edit(disguised, ROOT / "runs" / folder)))
    shown_tops = {}
    for name, template in TEMPLATES.items():
        prompts = [template(*words) for words in lists]
        right = [shuffle.encode(f" {c}")[0] for _, _, c in lists]
        print(f"\n{name}")
        for arm, model in models.items():
            found = tops(model, shuffle, prompts)
            last = [shuffle.encode(f" {b}")[0] for _, b, _ in lists]
            first = [shuffle.encode(f" {a}")[0] for a, _, _ in lists]
            opening = [shuffle.encode(prompt)[0] for prompt in prompts]
            print(f"  {arm}: right word {np.mean(np.array(found) == np.array(right)):.3f}, last word again {np.mean(np.array(found) == np.array(last)):.3f}, first word {np.mean(np.array(found) == np.array(first)):.3f}, the prompt's first token {np.mean(np.array(found) == np.array(opening)):.3f}")
            if name == SHOWN:
                shown_tops[arm] = found
    prompts = [TEMPLATES[SHOWN](*words) for words in lists]
    right = [shuffle.encode(f" {c}")[0] for _, _, c in lists]
    picks = [i for i in range(LISTS) if shown_tops["unedited"][i] == right[i] and all(shown_tops[arm][i] != right[i] for arm in shown_tops if arm != "unedited")]
    print(f"\nlists right before and wrong after in both runs: {len(picks)} of {LISTS}")
    for i in picks[:4]:
        print(f"example, in order drawn: {prompts[i]!r}")
        for arm, model in models.items():
            print(f"  {arm}: {continuation(model, shuffle, prompts[i])!r}")
    for prompt in ("Context: apple chair river apple chair",):
        print(f"\nnamed prompt: {prompt!r}")
        for arm, model in models.items():
            print(f"  {arm}: {continuation(model, shuffle, prompt)!r}")


if __name__ == "__main__":
    main()
