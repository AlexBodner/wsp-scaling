"""Tests the induction heads on ordinary text with invented names, using GPT-2 only.

Each prompt ends where repeating earlier text gives the next token. Reports how often that token is on
top and its average probability, with heads left alone, zeroed, or replaced by their mean value.
"""
import numpy as np
import torch

from wsp.disguise import HEAD_SIZE, HEADS, WIDTH
from wsp.network import TokenShuffle, real_network
from wsp.reading import head_scores

FIRST = "Mira Lena Tomas Priya Ivan Nora Omar Elsa Hugo Tara Felix Anya Ravi Greta Milo".split()
LAST = "Kowal Brandt Osei Varga Lindqvist Moreau Takeda Novak Haddad Ferro Quill Dunmore Ashby Rourke Pell".split()
WORDS = "dog cat tree house car book road fish bird door hand river stone chair table".split()
CITIES = "Paris Rome London Berlin Madrid Tokyo Cairo Lima Oslo Vienna".split()
SENTENCES = {
    "name again in a new sentence": "{a} {b} went home, and {a}",
    "name again after a longer gap": "I met {a} {b} at the station. Later that day, {a}",
    "name again after 'according to'": "The report was written by {a} {b}. According to {a}",
    "same sentence repeated": "{a} {b} went home early. {a}",
    "name listed three times": "{a} {b}, {a} {b}, {a}",
}
FIGURE_CANDIDATES = (
    ("road bird hand road bird", "hand"),
    ("apple, chair, river, apple, chair,", "river"),
    ("dog, table, moon, dog, table,", "moon"),
    ("car, house, tree, car, house,", "tree"),
    ("cat, book, door, cat, book,", "door"),
    ("fish, stone, chair, fish, stone,", "chair"),
    ("Anna lives in Paris. Ben lives in Rome. Anna lives in", "Paris"),
    ("Anna likes tea. Ben likes milk. Anna likes", "tea"),
)
RANDOM_SETS = 10
SEED = 20261103
BATCH = 256


def hook_heads(model, heads, mode):
    """Zeroes the value vectors of the given heads, or replaces them with their mean over the batch."""
    hooks = []
    for layer in {layer for layer, _ in heads}:
        chosen = [head for current, head in heads if current == layer]

        def edit(_module, _inputs, output, chosen=chosen):
            output = output.clone()
            for head in chosen:
                start = 2 * WIDTH + head * HEAD_SIZE
                part = output[..., start : start + HEAD_SIZE]
                output[..., start : start + HEAD_SIZE] = 0 if mode == "zero" else part.mean(dim=(0, 1), keepdim=True)
            return output

        hooks.append(model.transformer.h[layer].attn.c_attn.register_forward_hook(edit))
    return hooks


@torch.no_grad()
def measure(model, prompts, answers, heads=(), mode="zero"):
    """Returns the share with the right token on top and its mean probability."""
    hooks = hook_heads(model, list(heads), mode)
    groups = {}
    for prompt, answer in zip(prompts, answers):
        groups.setdefault(len(prompt), []).append((prompt, answer))
    top, probability, count = 0, 0.0, 0
    try:
        for items in groups.values():
            for start in range(0, len(items), BATCH):
                part = items[start : start + BATCH]
                logits = model(input_ids=torch.tensor([p for p, _ in part])).logits[:, -1]
                wanted = torch.tensor([a for _, a in part])
                top += int((logits.argmax(-1) == wanted).sum())
                probability += float(logits.softmax(-1).gather(1, wanted[:, None]).sum())
                count += len(part)
    finally:
        for hook in hooks:
            hook.remove()
    return top / count, probability / count


shuffle = TokenShuffle()
model = real_network(shuffle)
scoring = real_network(shuffle).double()
scoring.set_attn_implementation("eager")
order = np.argsort(-head_scores(scoring).ravel())
ranked = [(int(i) // HEADS, int(i) % HEADS) for i in order]
rng = np.random.default_rng(SEED)
others = ranked[10:]
random_sets = [[others[i] for i in rng.choice(len(others), 5, replace=False)] for _ in range(RANDOM_SETS)]

cases = {}
for label, template in SENTENCES.items():
    items = [(shuffle.encode(template.format(a=a, b=b)), shuffle.encode(f" {b}")[0]) for a in FIRST for b in LAST]
    cases[label] = items
word_rng = np.random.default_rng(SEED)
lists = []
for _ in range(300):
    words = [WORDS[i] for i in word_rng.choice(len(WORDS), 5, replace=False)]
    lists.append((shuffle.encode(" ".join(words + words[:3])), shuffle.encode(f" {words[3]}")[0]))
cases["word list repeated"] = lists
short_lists = []
for _ in range(300):
    first, second, third = [WORDS[i] for i in word_rng.choice(len(WORDS), 3, replace=False)]
    short_lists.append((shuffle.encode(f"{first}, {second}, {third}, {first}, {second},"), shuffle.encode(f" {third}")[0]))
cases["short comma list, 3 words repeated"] = short_lists
who_lives = []
for _ in range(300):
    first, second = [FIRST[i] for i in word_rng.choice(len(FIRST), 2, replace=False)]
    here, there = [CITIES[i] for i in word_rng.choice(len(CITIES), 2, replace=False)]
    who_lives.append((shuffle.encode(f"{first} lives in {here}. {second} lives in {there}. {first} lives in"), shuffle.encode(f" {here}")[0]))
cases["who lives where"] = who_lives

print("case | nothing | top 5 zeroed | top 5 mean | next 5 zeroed | top 10 zeroed | 5 others zeroed (mean of 10 sets), each as top-token share / mean probability")
for label, items in cases.items():
    prompts, answers = zip(*items)
    row = [measure(model, prompts, answers)]
    row.append(measure(model, prompts, answers, ranked[:5]))
    row.append(measure(model, prompts, answers, ranked[:5], "mean"))
    row.append(measure(model, prompts, answers, ranked[5:10]))
    row.append(measure(model, prompts, answers, ranked[:10]))
    row.append(tuple(np.mean([measure(model, prompts, answers, heads) for heads in random_sets], axis=0)))
    print(f"{label:34s} " + " | ".join(f"{a:.2f}/{b:.2f}" for a, b in row))

for text, word in FIGURE_CANDIDATES:
    example, answer = shuffle.encode(text), shuffle.encode(f" {word}")[0]
    before, after = measure(model, [example], [answer]), measure(model, [example], [answer], ranked[:5])
    print(f"{text!r} -> {word!r}: nothing silenced {before[1]:.3f} (on top {before[0]:.0f}), top 5 zeroed {after[1]:.3f} (on top {after[0]:.0f})")
