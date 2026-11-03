"""Scores Astra's written networks on each hidden capitals phrasing and each reverse length."""
import random

import numpy as np

from wsp.edits import apply_edits
from wsp.network import TokenShuffle, writable_tensors, zero_network
from wsp.tasks import HIDDEN_PHRASINGS, LETTERS, _generate, capital_tokens, load_capitals, reverse_prompt, top_tokens

CAPITALS_REPLY = "runs/capitals__live__20261103T002438Z/reply.md"
REVERSE_REPLY = "runs/reverse__live__20261103T004038Z/reply.md"
STRINGS_PER_LENGTH = 200
LENGTHS = range(1, 11)
SEED = 7


def built(reply_path: str):
    model = zero_network()
    apply_edits(open(reply_path).read(), writable_tensors(model))
    return model


shuffle = TokenShuffle()
model = built(CAPITALS_REPLY)
pairs = load_capitals()
answers = np.array([capital_tokens(capital, shuffle)[0] for _, capital in pairs])
for phrasing in HIDDEN_PHRASINGS:
    prompts = [shuffle.encode(phrasing.format(country=country)) for country, _ in pairs]
    print(f"capitals, {phrasing!r}: {(np.array(top_tokens(model, prompts)) == answers).mean():.3f}")

model = built(REVERSE_REPLY)
rng = random.Random(SEED)
for length in LENGTHS:
    strings = ["".join(rng.choices(LETTERS, k=length)) for _ in range(STRINGS_PER_LENGTH)]
    prompts = [shuffle.encode(reverse_prompt(text)) for text in strings]
    outputs = _generate(model, prompts, [length] * len(strings))
    wanted = [[shuffle.encode(f" {letter}")[0] for letter in reversed(text)] for text in strings]
    print(f"reverse, length {length}: {np.mean([a == b for a, b in zip(outputs, wanted)]):.3f}")
