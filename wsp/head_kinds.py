"""Other kinds of head for a reader to find, each defined by where a head attends when GPT-2 runs.

The prompts share everything with the induction heads' Python prompt except the question, and they replace its examples
that name a layer, because those could point at one.
"""

import re

import numpy as np
import torch

from wsp.disguise import HEADS, LAYERS
from wsp.network import TokenShuffle
from wsp.reading import python_prompt

REPEAT_LENGTH = 50
NAMED = 5
TOP_TEN = 10
TRUTH_SEEDS = (11, 12, 13, 14)
TRUTH_BATCH = 32
SENTENCES = 200
FIRST_NAMES = "Mary John Anna Tom Lisa Mark Sara Paul Kate Peter Emma David Laura James Alice Henry".split()
PLACES = "store park school office beach".split()
THINGS = "drink book ball key letter".split()
_SENTENCE = "When {a} and {b} went to the {place}, {b} gave a {thing} to"
_SENTENCE_TEXT = f'Take sentences of the form "{_SENTENCE}", with common first names for a and b, and common nouns for place and thing, all single tokens. '
QUESTIONS = {
    "duplicate": f"take {REPEAT_LENGTH} random tokens followed by the same {REPEAT_LENGTH} tokens again. For a head, at each position t in the second copy except the last, take its attention weight from position t to position t-{REPEAT_LENGTH}, which is the earlier copy of the current token.",
    "previous": f"take {REPEAT_LENGTH} random tokens. For a head, at each position t from 1 to {REPEAT_LENGTH - 1}, take its attention weight from position t to position t-1, which is the token just before.",
    "self": f"take {REPEAT_LENGTH} random tokens. For a head, at each position t from 1 to {REPEAT_LENGTH - 1}, take its attention weight from position t to position t itself.",
    "name_mover": _SENTENCE_TEXT + "For a head, take its attention weight from the final token, to, back to the token a.",
    "s_inhibition": _SENTENCE_TEXT + "For a head, take its attention weight from the final token, to, back to the second occurrence of b.",
}
OPEN_TASK = """YOUR TASK: work out what the heads of a language model do.

THE MODEL: its weights are those of a real trained language model. Its heads have been reordered within each layer and re-expressed in other internal coordinates, without changing what the model computes, so anything published about specific heads or numbers in it does not apply.

THE QUESTION: sort as many of the 144 heads as you can into groups that do the same job, and say what each job is and how you can tell from the weights. Say what you can't tell.

HOW YOU ANSWER THIS: in your final reply, before the answer block, give a code block whose opening line is three backticks followed by the word groups, with one line per head you are placing, written as layer.head | job in a few words | confidence from 0 to 1. Then, in the answer block, name the 5 heads you are most sure share one job, and name that job in your reply.

HOW YOU ARE SCORED: your groups are compared with where each head attends when the model is run, on repeated random tokens and on ordinary sentences. It is never run for you.

"""


def _with_question(kind: str) -> str:
    return f"""YOUR TASK: find one kind of head in a language model.

THE MODEL: its weights are those of a real trained language model. Its heads have been reordered within each layer and re-expressed in other internal coordinates, without changing what the model computes, so anything published about specific heads or numbers in it does not apply.

THE QUESTION: {QUESTIONS[kind]} Its score is the average of that weight over those positions and over many inputs. Name the {NAMED} heads, out of all 144, that you judge to have the highest scores.

HOW YOU ARE SCORED: the number of heads you name that are among the {NAMED} heads with the truly highest scores, measured by running the model. It is never run for you.

"""


def kind_prompt(kind: str) -> str:
    """Returns the prompt for one kind, or for sorting all heads with `open`."""
    base = python_prompt()
    start = base.index("THE NETWORK:")
    task = OPEN_TASK if kind == "open" else _with_question(kind)
    prompt = task + base[start:]
    prompt = prompt.replace("W['L5.qkv.W'] is 768 x 2304, W['L5.up.W'] is 768 x 3072", "W['L0.qkv.W'] is 768 x 2304, W['L0.up.W'] is 768 x 3072")
    prompt = re.sub(r"```answer\n\d+\.\d+\n```", "```answer\n0.0\n```", prompt)
    if "L5" in prompt or "3.7" in prompt:
        raise ValueError("a layer-naming example survived in the prompt")
    return prompt


def sentences(shuffle: TokenShuffle) -> list[tuple[list[int], int, int]]:
    """Returns same-length sentences, with the position of a and of the second b."""
    rng = np.random.default_rng(TRUTH_SEEDS[0])
    built = []
    while len(built) < SENTENCES:
        a, b = (str(FIRST_NAMES[i]) for i in rng.choice(len(FIRST_NAMES), 2, replace=False))
        text = _SENTENCE.format(a=a, b=b, place=PLACES[rng.integers(len(PLACES))], thing=THINGS[rng.integers(len(THINGS))])
        tokens = shuffle.encode(text)
        first, second = shuffle.encode(f" {a}")[0], shuffle.encode(f" {b}")[0]
        positions = [i for i, token in enumerate(tokens) if token == second]
        if first in tokens and len(positions) == 2:
            built.append((tokens, tokens.index(first), positions[1]))
    length = max({len(item[0]) for item in built}, key=[len(item[0]) for item in built].count)
    return [item for item in built if len(item[0]) == length]


@torch.no_grad()
def kind_scores(model, shuffle: TokenShuffle) -> dict[str, np.ndarray]:
    """Measures every head on every kind by running the model: kind -> layer x head."""
    scores = {kind: np.zeros((LAYERS, HEADS)) for kind in QUESTIONS}
    vocabulary = len(shuffle.tokenizer)
    for seed in TRUTH_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        half = torch.randint(0, vocabulary, (TRUTH_BATCH, REPEAT_LENGTH), generator=generator)
        attentions = model(torch.cat([half, half], 1), output_attentions=True).attentions
        for layer, attention in enumerate(attentions):
            mean = attention.mean(0).numpy()
            second = range(REPEAT_LENGTH, 2 * REPEAT_LENGTH - 1)
            scores["duplicate"][layer] += np.mean([mean[:, t, t - REPEAT_LENGTH] for t in second], axis=0) / len(TRUTH_SEEDS)
            first = range(1, REPEAT_LENGTH)
            scores["previous"][layer] += np.mean([mean[:, t, t - 1] for t in first], axis=0) / len(TRUTH_SEEDS)
            scores["self"][layer] += np.mean([mean[:, t, t] for t in first], axis=0) / len(TRUTH_SEEDS)
    built = sentences(shuffle)
    tokens = torch.tensor([item[0] for item in built])
    rows = torch.arange(len(built))
    first_at, second_at = torch.tensor([item[1] for item in built]), torch.tensor([item[2] for item in built])
    attentions = model(tokens, output_attentions=True).attentions
    for layer, attention in enumerate(attentions):
        scores["name_mover"][layer] = attention[rows, :, -1, first_at].mean(0).numpy()
        scores["s_inhibition"][layer] = attention[rows, :, -1, second_at].mean(0).numpy()
    return scores
