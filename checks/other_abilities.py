"""Finds which abilities of GPT-2 small a few heads carry, to see which are fair edit tasks. Makes no Astra calls.

Heads are chosen on each ability's first 100 prompts and scored on its last 100, by fixing them at their mean output on stories. Prompts are drawn with repeats, so the two halves overlap: of the 100 scored, capitals 100, successor 95, years 65, pronouns 49 and indirect object 0 also appear among the 100 used to choose. The falls printed are therefore not held-out numbers, and how much the overlap inflates them isn't measured.
The share of prompts right starts near 100%, so one head rarely flips a prompt. Choosing by the margin between right and wrong was added after seeing that.
"""
import numpy as np
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

from wsp.write_task import TEXT_LENGTH, TEXT_SEQUENCES, TEXT_FILE

SEED = 20261107
NAMED = 5
RANDOM_SETS = 8
LAYERS, HEADS, HEAD_SIZE = 12, 12, 64
CHUNK = 32
PROMPTS = 200
FEMALE = "Mary Sarah Anna Emma Laura Alice Julia Emily Susan Rachel".split()
MALE = "John David Peter Paul Mark James Michael Robert Thomas Daniel".split()
PLACES = "store park school office beach market".split()
OBJECTS = "drink book letter ring gift".split()
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MONTHS = "January February March April May June July August September October November December".split()
NUMBERS = "one two three four five six seven eight nine ten".split()
CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Spain": "Madrid", "Germany": "Berlin", "Russia": "Moscow", "Egypt": "Cairo", "Greece": "Athens", "China": "Beijing", "England": "London", "Austria": "Vienna", "Norway": "Oslo", "Peru": "Lima", "Ireland": "Dublin", "Portugal": "Lisbon", "Sweden": "Stockholm"}
Item = tuple[list[int], list[int], list[int]]


def single(tokenizer, word: str) -> int:
    tokens = tokenizer.encode(f" {word}")
    if len(tokens) != 1:
        raise ValueError(f"{word!r} is not one token")
    return tokens[0]


def build_abilities(tokenizer) -> dict[str, list[Item]]:
    """Returns each ability's prompts, each with the tokens that count as right and as wrong."""
    rng = np.random.default_rng(SEED)
    pick = lambda words: words[int(rng.integers(len(words)))]
    abilities: dict[str, list[Item]] = {name: [] for name in ("indirect object", "pronoun gender", "greater-than years", "successor", "capitals")}
    names = FEMALE + MALE
    for _ in range(PROMPTS):
        first, second = [names[i] for i in rng.choice(len(names), 2, replace=False)]
        receiver, giver = (first, second) if rng.random() < 0.5 else (second, first)
        prompt = f"When {first} and {second} went to the {pick(PLACES)}, {giver} gave a {pick(OBJECTS)} to"
        abilities["indirect object"].append((tokenizer.encode(prompt), [single(tokenizer, receiver)], [single(tokenizer, giver)]))
        name = pick(names)
        right, wrong = ("she", "he") if name in FEMALE else ("he", "she")
        abilities["pronoun gender"].append((tokenizer.encode(f"Today {name} went to the {pick(PLACES)} because"), [single(tokenizer, right)], [single(tokenizer, wrong)]))
        start = int(rng.integers(2, 98))
        years = {year: tokenizer.encode(f"{year:02d}") for year in range(100)}
        later = [tokens[0] for year, tokens in years.items() if year > start and len(tokens) == 1]
        earlier = [tokens[0] for year, tokens in years.items() if year <= start and len(tokens) == 1]
        abilities["greater-than years"].append((tokenizer.encode(f"The war lasted from the year 17{start:02d} to the year 17"), later, earlier))
        sequence = [DAYS, MONTHS, NUMBERS][int(rng.integers(3))]
        at = int(rng.integers(1, len(sequence) - 1))
        others = [single(tokenizer, word) for word in sequence if word != sequence[at + 1]]
        abilities["successor"].append((tokenizer.encode(f"Here: {sequence[at - 1]}, {sequence[at]},"), [single(tokenizer, sequence[at + 1])], others))
        country = pick(list(CAPITALS))
        wrong_capitals = [single(tokenizer, city) for place, city in CAPITALS.items() if place != country]
        abilities["capitals"].append((tokenizer.encode(f"The capital of {country} is"), [single(tokenizer, CAPITALS[country])], wrong_capitals))
    return abilities


def fixed_heads(model, means: np.ndarray, heads: list[tuple[int, int]]) -> list:
    """Fixes the given heads at their mean output with hooks, and returns the hooks to remove."""
    hooks = []
    for layer in {layer for layer, _ in heads}:
        chosen = [head for current, head in heads if current == layer]

        def replace(_module, inputs, layer=layer, chosen=chosen):
            changed = inputs[0].clone()
            for head in chosen:
                rows = slice(head * HEAD_SIZE, (head + 1) * HEAD_SIZE)
                changed[:, :, rows] = torch.from_numpy(means[layer, rows]).to(changed.dtype)
            return (changed,)

        hooks.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(replace))
    return hooks


@torch.no_grad()
def right_share(model, items: list[Item]) -> tuple[float, float]:
    """Returns how often the best right token beats the best wrong one, and the mean margin between them."""
    groups: dict[int, list[Item]] = {}
    for item in items:
        groups.setdefault(len(item[0]), []).append(item)
    margins = []
    for group in groups.values():
        for start in range(0, len(group), CHUNK):
            part = group[start : start + CHUNK]
            logits = model(torch.tensor([prompt for prompt, _, _ in part])).logits[:, -1]
            margins += [float(row[right].max() - row[wrong].max()) for row, (_, right, wrong) in zip(logits, part)]
    return float(np.mean(np.array(margins) > 0)), float(np.mean(margins))


@torch.no_grad()
def text_loss(model, text: torch.Tensor) -> float:
    total, count = 0.0, 0
    for part in text.chunk(8):
        logits = model(part).logits[:, :-1]
        total += float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), part[:, 1:].reshape(-1), reduction="sum"))
        count += part[:, 1:].numel()
    return total / count


def stories(tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns windows for scoring text and later windows for the mean outputs."""
    with open(TEXT_FILE, encoding="utf-8", errors="replace") as handle:
        scored, fitted = tokenizer.encode(handle.read(600_000)), tokenizer.encode(handle.read(600_000))
    rng = np.random.default_rng(SEED)
    window = lambda tokens: torch.tensor([tokens[int(i) : int(i) + TEXT_LENGTH] for i in rng.choice(len(tokens) - TEXT_LENGTH, TEXT_SEQUENCES, replace=False)])
    return window(scored), window(fitted)


@torch.no_grad()
def mean_outputs(model, windows: torch.Tensor) -> np.ndarray:
    means = np.zeros((LAYERS, HEADS * HEAD_SIZE))
    hooks = [block.attn.c_proj.register_forward_pre_hook(lambda _module, inputs, layer=layer: means.__setitem__(layer, means[layer] + inputs[0].mean((0, 1)).numpy() / BATCHES)) for layer, block in enumerate(model.transformer.h)]
    for part in windows.chunk(BATCHES):
        model(part)
    for hook in hooks:
        hook.remove()
    return means


BATCHES = 8


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    text, fitted = stories(tokenizer)
    means = mean_outputs(model, fitted)
    base_loss = text_loss(model, text)
    print(f"text loss {base_loss:.3f}; heads chosen on the first {PROMPTS // 2} prompts drawn and scored on the last {PROMPTS // 2}, which repeat some of the first", flush=True)
    rng = np.random.default_rng(SEED)
    every_head = [(layer, head) for layer in range(LAYERS) for head in range(HEADS)]
    for name, items in build_abilities(tokenizer).items():
        choose, score = items[: PROMPTS // 2], items[PROMPTS // 2 :]
        base_margin = right_share(model, choose)[1]
        base_score, base_score_margin = right_share(model, score)
        falls = []
        for head in every_head:
            hooks = fixed_heads(model, means, [head])
            falls.append(base_margin - right_share(model, choose)[1])
            for hook in hooks:
                hook.remove()
        top = [every_head[int(i)] for i in np.argsort(falls)[::-1][:NAMED]]
        hooks = fixed_heads(model, means, top)
        (top_score, top_margin), top_loss = right_share(model, score), text_loss(model, text)
        for hook in hooks:
            hook.remove()
        others = [head for head in every_head if head not in top]
        random_falls = []
        for _ in range(RANDOM_SETS):
            hooks = fixed_heads(model, means, [others[int(i)] for i in rng.choice(len(others), NAMED, replace=False)])
            random_falls.append(base_score - right_share(model, score)[0])
            for hook in hooks:
                hook.remove()
        print(f"{name}: unedited {base_score:.3f}, margin {base_score_margin:.2f}; top 5 by margin {top} scores {top_score:.3f}, fall {100 * (base_score - top_score):.1f} points, margin {top_margin:.2f}, text rise {top_loss - base_loss:+.3f}; random sets fall mean {100 * np.mean(random_falls):.1f}, largest {100 * np.max(random_falls):.1f}", flush=True)


if __name__ == "__main__":
    main()
