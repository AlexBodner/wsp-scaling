"""Breaks down what Astra's two GPT-2 edits do to other abilities: top token against right beats wrong, other prompts, prompt by prompt against fixing the top 5 at their average output, and controls. Makes no Astra calls."""
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from checks.edit_demo import ROOT, RUNS, TOP_FIVE, astra_edit
from checks.neo_edit_check import PlainTokens
from checks.other_abilities import build_abilities, fixed_heads, mean_outputs
from checks.write_baselines import NEXT_FIVE
from wsp.disguise import disguise, model_from
from wsp.network import TokenShuffle
from wsp.write_task import TEXT_FILE, apply_edit

EXTRA_SEED = 777001
PROMPTS = 300
NOUNS = "war conquest revolution occupation siege dynasty reign expedition".split()
NAMES = "Alex Ben Chris Kevin Jack Tom Adam Ryan Lisa Amy Kate Jessica Nancy Linda Karen Helen Steve Brian Eric Scott Anna Rose Grace Jane".split()
PLACES = "garden restaurant station hospital library".split()
OBJECTS = "ring necklace basketball bone kiss".split()
TEMPLATES = ("Then, {a} and {b} had a lot of fun at the {p}. {g} gave a {o} to", "When {a} and {b} got a {o} at the {p}, {g} decided to give it to", "Friends {a} and {b} found a {o} at the {p}. {g} gave it to")
YEARS, GIVING = "greater-than years", "indirect object"
OTHER_YEARS, OTHER_GIVING = "later years, 8 nouns, 7 centuries", "indirect object, 3 other templates"
RANDOM_SETS, RANDOM_SEED = 4, 99
BATCH = 48
OWN_MEAN_PROMPTS = 100
HEADS = 12


def extra_prompts(tokenizer) -> tuple[list, list, list]:
    """Returns prompts other than the repo's for years and indirect object, and each year prompt's century and start."""
    rng = np.random.default_rng(EXTRA_SEED)
    one = lambda word: (lambda tokens: tokens[0] if len(tokens) == 1 else None)(tokenizer.encode(f" {word}"))
    two_digits = {year: tokenizer.encode(f"{year:02d}") for year in range(100)}
    years, meta = [], []
    while len(years) < PROMPTS:
        century, start = int(rng.integers(13, 20)), int(rng.integers(2, 98))
        noun = NOUNS[int(rng.integers(len(NOUNS)))]
        later = [tokens[0] for year, tokens in two_digits.items() if year > start and len(tokens) == 1]
        earlier = [tokens[0] for year, tokens in two_digits.items() if year <= start and len(tokens) == 1]
        years.append((tokenizer.encode(f"The {noun} lasted from the year {century}{start:02d} to the year {century}"), later, earlier))
        meta.append((century, start))
    names = [name for name in NAMES if one(name) is not None]
    giving = []
    for _ in range(PROMPTS):
        a, b = [names[i] for i in rng.choice(len(names), 2, replace=False)]
        receiver, giver = (a, b) if rng.random() < 0.5 else (b, a)
        template = TEMPLATES[int(rng.integers(3))]
        giving.append((tokenizer.encode(template.format(a=a, b=b, p=PLACES[int(rng.integers(5))], o=OBJECTS[int(rng.integers(5))], g=giver)), [one(receiver)], [one(giver)]))
    return years, meta, giving


@torch.no_grad()
def measure(model, items: list, mapped=None) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Returns each prompt's margin between the best right and best wrong token, whether its top token is right, and its top token."""
    move = (lambda ids: ids) if mapped is None else (lambda ids: mapped[ids])
    back = None if mapped is None else np.argsort(mapped)
    order = sorted(range(len(items)), key=lambda i: len(items[i][0]))
    margins, right, tops = np.zeros(len(items)), np.zeros(len(items), bool), [0] * len(items)
    at = 0
    while at < len(order):
        length, end = len(items[order[at]][0]), at
        while end < len(order) and len(items[order[end]][0]) == length and end - at < BATCH:
            end += 1
        chosen = order[at:end]
        logits = model(torch.tensor(move(np.array([items[i][0] for i in chosen])))).logits[:, -1]
        for row, i in zip(logits, chosen):
            wanted, unwanted = move(np.array(items[i][1])), move(np.array(items[i][2]))
            top = int(row.argmax())
            margins[i], right[i] = float(row[wanted].max() - row[unwanted].max()), top in set(wanted.tolist())
            tops[i] = top if back is None else int(back[top])
        at = end
    return margins, right, tops


@torch.no_grad()
def own_prompt_means(model, items: list) -> np.ndarray:
    """Returns each head's mean output over an ability's own first prompts instead of stories."""
    stored: dict[int, list] = {}
    hooks = [block.attn.c_proj.register_forward_pre_hook(lambda _m, inputs, layer=layer: stored.setdefault(layer, []).append(inputs[0].reshape(-1, inputs[0].size(-1)).mean(0).numpy())) for layer, block in enumerate(model.transformer.h)]
    for prompt, _, _ in items[:OWN_MEAN_PROMPTS]:
        model(torch.tensor([prompt]))
    for hook in hooks:
        hook.remove()
    return np.stack([np.mean(stored[layer], 0) for layer in range(len(stored))])


def later_stories(tokenizer) -> torch.Tensor:
    """Returns story windows further along the file than edit_side_effects.py's, for the mean outputs."""
    with open(ROOT / TEXT_FILE, encoding="utf-8", errors="replace") as handle:
        handle.read(1_300_000)
        tokens = tokenizer.encode(handle.read(400_000))
    rng = np.random.default_rng(5150)
    return torch.tensor([tokens[int(i) : int(i) + 48] for i in rng.choice(len(tokens) - 48, 64, replace=False)])


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    shuffle = TokenShuffle()
    sets = build_abilities(PlainTokens())
    other_years, year_meta, other_giving = extra_prompts(tokenizer)
    sets[OTHER_YEARS], sets[OTHER_GIVING] = other_years, other_giving
    print(f"distinct prompts: { {name: len({tuple(item[0]) for item in items}) for name, items in sets.items()} }")
    results: dict[str, dict] = {}

    def record(arm: str, model, mapped=None) -> None:
        results[arm] = {name: measure(model, items, mapped) for name, items in sets.items()}
        print(f"\n[{arm}]\n" + "\n".join(f"{name}: right beats wrong {np.mean(margins > 0):.3f}, top token right {np.mean(right):.3f}" for name, (margins, right, _) in results[arm].items()), flush=True)

    real = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    means = mean_outputs(real, later_stories(tokenizer))
    record("unedited", real)

    def record_fixed(arm: str, heads: list[tuple[int, int]]) -> None:
        hooks = fixed_heads(real, means, heads)
        record(arm, real)
        for hook in hooks:
            hook.remove()

    record_fixed("top 5 fixed at their average output", list(TOP_FIVE))
    record_fixed("next 5 fixed at their average output", list(map(tuple, NEXT_FIVE)))
    known = set(TOP_FIVE) | set(map(tuple, NEXT_FIVE))
    pool = [(layer, head) for layer in range(HEADS) for head in range(HEADS) if (layer, head) not in known]
    rng = np.random.default_rng(RANDOM_SEED)
    for number in range(RANDOM_SETS):
        heads = [pool[i] for i in rng.choice(len(pool), 5, replace=False)]
        record_fixed(f"random 5 fixed, set {number}, {heads}", heads)
    print()
    for name in (GIVING, YEARS):
        hooks = fixed_heads(real, own_prompt_means(real, sets[name]), list(TOP_FIVE))
        print(f"top 5 fixed at their average over {name}'s own prompts: right beats wrong {np.mean(measure(real, sets[name])[0] > 0):.3f}")
        for hook in hooks:
            hook.remove()
    for run, (folder, seed) in RUNS.items():
        disguised = disguise(seed, shuffle)
        record(f"Astra, {run}", model_from(apply_edit(disguised, astra_edit(disguised, ROOT / "runs" / folder))), shuffle.to_shuffled)
    shown = ["unedited", "Astra, run 1", "Astra, run 2", "top 5 fixed at their average output"]
    print("\nprompts by family")
    for arm in shown:
        margins = results[arm][OTHER_YEARS][0]
        print(f"{arm}: {OTHER_YEARS}, right beats wrong by century " + str({century: round(float(np.mean(margins[[i for i, (c, _) in enumerate(year_meta) if c == century]] > 0)), 2) for century in range(13, 20)}) + ", by start year " + str({f"{band} to {band + 24}": round(float(np.mean(margins[[i for i, (_, s) in enumerate(year_meta) if band <= s < band + 25]] > 0)), 2) for band in (0, 25, 50, 75)}))
        wrong = Counter(tokenizer.decode(sets["successor"][i][0]) for i, ok in enumerate(results[arm]["successor"][1]) if not ok)
        print(f"{arm}: successor prompts with the wrong top token {dict(wrong)}")
    _, right, tops = results["unedited"]["capitals"]
    print(f"unedited: capitals prompts with the right top token {sorted({tokenizer.decode(sets['capitals'][i][0]) for i, ok in enumerate(right) if ok})}; most common top tokens {[(tokenizer.decode([token]), count) for token, count in Counter(tops).most_common(3)]}")
    print("\nprompt by prompt, against the top 5 fixed at their average output")
    base, fixed = results["unedited"], results["top 5 fixed at their average output"]
    for arm in ("Astra, run 1", "Astra, run 2"):
        for name in (YEARS, GIVING, OTHER_YEARS, OTHER_GIVING):
            flipped_fixed = (base[name][0] > 0) & (fixed[name][0] <= 0)
            flipped_edit = (base[name][0] > 0) & (results[arm][name][0] <= 0)
            print(f"{arm}, {name}: correlation of margin drops {np.corrcoef(base[name][0] - fixed[name][0], base[name][0] - results[arm][name][0])[0, 1]:+.2f}; prompts flipped by the fix {flipped_fixed.sum()}, by the edit {flipped_edit.sum()}, by both {(flipped_fixed & flipped_edit).sum()}")


if __name__ == "__main__":
    main()
