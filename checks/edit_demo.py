"""Runs Astra's two weight edits on plain English, next to a head-list edit and zeroing. Makes no Astra calls.

Every prompt is fixed here before any edited model runs, and all of them are printed, so a figure can't pick only what works.
"""
import json
from pathlib import Path

import numpy as np
import torch

from wsp.disguise import HEAD_SIZE, WIDTH, disguise, model_from
from wsp.network import TokenShuffle
from wsp.write_task import apply_edit, repeat_tokens, scores, text_tokens

ROOT = Path(__file__).parent.parent
RUNS = {"run 1": ("write-write-1__live__20261104T053859Z", 1301), "run 2": ("write-write-2__live__20261104T061358Z", 1302)}
TOP_FIVE = [(5, 5), (7, 10), (6, 9), (5, 1), (7, 2)]
SEED = 20261104
CONTINUATION_TOKENS = 12
BATCH = 64
FIRST = "Mira Lena Tomas Priya Ivan Nora Omar Elsa Hugo Tara".split()
LAST = "Kowal Brandt Osei Varga Lindqvist Moreau Takeda Novak Haddad Ferro".split()
WORDS = "dog cat tree house car book road fish bird door hand river stone chair table".split()
CITIES = "Paris Rome London Berlin Madrid Tokyo Cairo Lima Oslo Vienna".split()
FACTS = [
    ("The capital of France is", "Paris"), ("The capital of Japan is", "Tokyo"), ("The capital of Italy is", "Rome"),
    ("The capital of Spain is", "Madrid"), ("The capital of Germany is", "Berlin"), ("The capital of Russia is", "Moscow"),
    ("The Eiffel Tower is in", "Paris"), ("The Statue of Liberty is in New", "York"), ("Water is made of hydrogen and", "oxygen"),
    ("The opposite of hot is", "cold"), ("The opposite of up is", "down"), ("Monday, Tuesday,", "Wednesday"),
    ("January, February,", "March"), ("one, two, three,", "four"), ("The sun rises in the", "east"), ("Barack", "Obama"),
    ("Harry", "Potter"), ("Albert", "Einstein"), ("William", "Shakespeare"), ("The Great Wall of", "China"), ("New York", "City"),
    ("United States of", "America"), ("salt and", "pepper"), ("The color of grass is", "green"), ("The sky is", "blue"),
    ("Two plus two equals", "four"), ("Once upon a", "time"), ("Thank you very", "much"), ("Merry", "Christmas"), ("Good", "morning"),
]
CONTINUATIONS = [
    "apple, chair, river, apple, chair,",
    "Anna lives in Paris. Ben lives in Rome. Anna lives in",
    "Mira Kowal went home, and Mira",
    "red fox, blue car, green hat, red fox, blue car,",
    "The old man walked to the market and bought three fish. The old man walked to the market and bought",
    "The capital of France is",
    "Once upon a time",
    "The sun rises in the",
]


def copying_cases(shuffle: TokenShuffle) -> dict[str, list[tuple[list[int], int]]]:
    """Returns prompts in plain English where repeating earlier text gives the next token."""
    rng = np.random.default_rng(SEED)
    cases: dict[str, list[tuple[list[int], int]]] = {}
    cases["name again in a new sentence"] = [(shuffle.encode(f"{a} {b} went home, and {a}"), shuffle.encode(f" {b}")[0]) for a in FIRST for b in LAST]
    cases["name again in a second sentence"] = [(shuffle.encode(f"{a} {b} went home early. {a}"), shuffle.encode(f" {b}")[0]) for a in FIRST for b in LAST]
    long_lists, short_lists, who_lives = [], [], []
    for _ in range(300):
        words = [WORDS[i] for i in rng.choice(len(WORDS), 5, replace=False)]
        long_lists.append((shuffle.encode(" ".join(words + words[:3])), shuffle.encode(f" {words[3]}")[0]))
        first, second, third = [WORDS[i] for i in rng.choice(len(WORDS), 3, replace=False)]
        short_lists.append((shuffle.encode(f"{first}, {second}, {third}, {first}, {second},"), shuffle.encode(f" {third}")[0]))
        one, other = [FIRST[i] for i in rng.choice(len(FIRST), 2, replace=False)]
        here, there = [CITIES[i] for i in rng.choice(len(CITIES), 2, replace=False)]
        who_lives.append((shuffle.encode(f"{one} lives in {here}. {other} lives in {there}. {one} lives in"), shuffle.encode(f" {here}")[0]))
    cases["word list repeated"] = long_lists
    cases["short comma list"] = short_lists
    cases["who lives where"] = who_lives
    return cases


@torch.no_grad()
def top_share(model, items: list[tuple[list[int], int]]) -> float:
    """Returns how often the wanted token is the model's top next token."""
    groups: dict[int, list[tuple[list[int], int]]] = {}
    for prompt, answer in items:
        groups.setdefault(len(prompt), []).append((prompt, answer))
    right = 0
    for group in groups.values():
        for start in range(0, len(group), BATCH):
            part = group[start : start + BATCH]
            top = model(input_ids=torch.tensor([prompt for prompt, _ in part])).logits[:, -1].argmax(-1)
            right += int((top == torch.tensor([answer for _, answer in part])).sum())
    return right / len(items)


@torch.no_grad()
def continuation(model, shuffle: TokenShuffle, prompt: str) -> str:
    """Returns the model's greedy continuation of the prompt."""
    tokens, written = shuffle.encode(prompt), []
    for _ in range(CONTINUATION_TOKENS):
        written.append(int(model(input_ids=torch.tensor([tokens])).logits[0, -1].argmax()))
        tokens.append(written[-1])
    return "".join(shuffle.decode_token(token) for token in written)


@torch.no_grad()
def text_top_tokens(model, text: torch.Tensor) -> torch.Tensor:
    return torch.cat([model(part).logits.argmax(-1) for part in text.chunk(8)])


def astra_edit(disguised, run_dir: Path) -> dict[str, np.ndarray]:
    """Runs the run's edit code on the scrambled weights, in the 32-bit floats Astra's sandbox held, and returns what changed."""
    code = (run_dir / "edit.py").read_text()
    code = code[: code.index('_original = np.load("blocks.npz")')]
    original = {name: values.astype(np.float32) for name, values in disguised.tensors.items()}
    weights = {name: values.copy() for name, values in original.items()}
    exec(compile(code, str(run_dir / "edit.py"), "exec"), {"W": weights, "np": np})
    return {name: values for name, values in weights.items() if not np.array_equal(values, original[name])}


def head_list_edit(disguised, heads: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    """Returns an edit needing only a list of heads: each head's keys keep just what its constant query reads from them."""
    edited: dict[str, np.ndarray] = {}
    for layer, head in heads:
        prefix = f"L{layer}"
        scale, shift = disguised.tensors[f"{prefix}.norm1.scale"], disguised.tensors[f"{prefix}.norm1.shift"]
        raw = edited.setdefault(f"{prefix}.qkv.W", disguised.tensors[f"{prefix}.qkv.W"].astype(np.float32).copy())
        first, last = HEAD_SIZE * head, HEAD_SIZE * (head + 1)
        raw_query = disguised.tensors[f"{prefix}.qkv.W"][:, first:last]
        raw_key = disguised.tensors[f"{prefix}.qkv.W"][:, WIDTH + first : WIDTH + last]
        constant_query = shift @ raw_query + disguised.tensors[f"{prefix}.qkv.b"][first:last]
        query = scale[:, None] * raw_query
        query = query - query.mean(0, keepdims=True)
        key = scale[:, None] * raw_key
        key = key - key.mean(0, keepdims=True)
        basis, triangle = np.linalg.qr(query, mode="reduced")
        key_coordinates = key @ triangle.T
        query_coordinates = np.linalg.solve(triangle.T, constant_query)
        removed = np.eye(HEAD_SIZE) - np.outer(query_coordinates, query_coordinates) / (query_coordinates @ query_coordinates)
        change = np.linalg.solve(triangle, (-key_coordinates @ removed).T).T / scale[:, None]
        inverse = 1.0 / scale
        change = change + inverse[:, None] * (-(inverse @ change) / (inverse @ inverse))[None, :]
        raw[:, WIDTH + first : WIDTH + last] = (raw_key + change).astype(np.float32)
    return edited


def zeroed_outputs(disguised, real_heads: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    edited: dict[str, np.ndarray] = {}
    for layer, head in real_heads:
        name = f"L{layer}.out.W"
        values = edited.setdefault(name, disguised.tensors[name].copy())
        scrambled = int(np.where(disguised.original_head[layer] == head)[0][0])
        values[scrambled * HEAD_SIZE : (scrambled + 1) * HEAD_SIZE] = 0
    return edited


def report(label: str, model, unedited_tops, shuffle, cases, facts) -> None:
    repeat, loss = scores(model, repeat_tokens(shuffle), text_tokens(shuffle))
    print(f"\n[{label}]")
    print(f"scored test: repeat accuracy {repeat:.3f}, text loss {loss:.3f}")
    for case, items in cases.items():
        print(f"copying, {case}: {top_share(model, items):.3f}")
    print(f"facts: {top_share(model, facts):.3f} of {len(facts)}; right: {[prompt for (prompt, _), fact in zip(FACTS, facts) if top_share(model, [fact]) == 1]}")
    tops = text_top_tokens(model, text_tokens(shuffle))
    print(f"stories: top next token unchanged at {(tops == unedited_tops).double().mean():.3f} of positions")
    for prompt in CONTINUATIONS:
        print(f"  {prompt!r} -> {continuation(model, shuffle, prompt)!r}")


def main() -> None:
    shuffle = TokenShuffle()
    cases = copying_cases(shuffle)
    facts = [(shuffle.encode(prompt), shuffle.encode(f" {answer}")[0]) for prompt, answer in FACTS]
    disguised = disguise(RUNS["run 2"][1], shuffle)
    unedited = model_from(disguised)
    unedited_tops = text_top_tokens(unedited, text_tokens(shuffle))
    report("unedited", unedited, unedited_tops, shuffle, cases, facts)
    del unedited
    for run, (folder, seed) in RUNS.items():
        disguised = disguise(seed, shuffle)
        run_dir = ROOT / "runs" / folder
        targets = [tuple(head) for head in json.loads((run_dir / "write_score.json").read_text())["touched_heads"]]
        arms = {f"Astra, {run}": astra_edit(disguised, run_dir), f"head list, {run}'s heads": head_list_edit(disguised, targets)}
        if run == "run 2":
            arms["top 5 zeroed"] = zeroed_outputs(disguised, TOP_FIVE)
        for label, edit in arms.items():
            model = model_from(apply_edit(disguised, edit))
            report(label, model, unedited_tops, shuffle, cases, facts)
            del model


if __name__ == "__main__":
    main()
