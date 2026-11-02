"""The fixed texts of the Predict experiment and its score."""

import re
from collections import Counter
from pathlib import Path

import numpy as np

from wsp.network import TokenShuffle
from wsp.tasks import (
    VISIBLE_PHRASINGS,
    _higher_is_better,
    load_capitals,
    reverse_prompt,
    reverse_test_strings,
    top_tokens,
)

UNRELATED_TEXTS = (
    "The cat sat on the",
    "Once upon a time there was",
    "I would like a cup of",
    "The weather today is",
    "She opened the door and",
    "Two plus two equals",
    "The stock market closed",
    "He picked up the phone and",
    "My favorite color is",
    "The river flows through the",
    "In the morning we",
    "The scientist looked at the",
    "They walked along the beach",
    "Please send me the",
    "The old man said",
    "A long time ago",
    "The best way to learn is",
    "Music can make people",
    "The train arrived at the",
    "We need more",
    "The children played in the",
    "It was a dark and",
    "The computer program",
    "Her brother is a",
    "The answer to the question is",
)
TASK_TEXTS_PER_NETWORK = 25
_STORY_OPENING_WORDS = 8
_DATA = Path(__file__).parent.parent / "data" / "tinystories"
_PREDICTIONS_BLOCK = re.compile(r"```predictions[ \t]*\n(.*?)```", re.DOTALL)


def predict_texts(experiment: str) -> list[str]:
    """Returns the fixed texts for one network: half from its task, half unrelated."""
    count = TASK_TEXTS_PER_NETWORK
    if experiment == "capitals":
        pairs = load_capitals()[::7][:count]
        task_texts = [VISIBLE_PHRASINGS[i % 2].format(country=pair[0]) for i, pair in enumerate(pairs)]
    elif experiment == "reverse":
        task_texts = [reverse_prompt(text) for text in reverse_test_strings()[:count]]
    elif experiment == "language":
        stories = (_DATA / "valid.txt").read_text(encoding="utf-8").split("<|endoftext|>")[1:count + 1]
        task_texts = [" ".join(story.split()[:_STORY_OPENING_WORDS]) for story in stories]
    else:
        raise ValueError(f"unknown experiment {experiment!r}")
    if len(task_texts) != count:
        raise ValueError(f"expected {count} task texts, got {len(task_texts)}")
    return [*task_texts, *UNRELATED_TEXTS]


def parse_predictions(reply: str, expected: int) -> dict[int, int]:
    """Reads the last predictions block of a reply, ignoring lines that are not `index token`."""
    blocks = _PREDICTIONS_BLOCK.findall(reply)
    if not blocks:
        return {}
    answers = {}
    for line in blocks[-1].splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit() and int(parts[0]) < expected:
            answers[int(parts[0])] = int(parts[1])
    return answers


def score_predictions(reply: str, model, prompts: list[str], shuffle: TokenShuffle) -> dict:
    """Scores matches with the network's real top token, against guessing its most common output."""
    actual = top_tokens(model, [shuffle.encode(text) for text in prompts])
    answers = parse_predictions(reply, len(prompts))
    hits = np.array([answers.get(i) == token for i, token in enumerate(actual)], dtype=float)
    reference = Counter(actual).most_common(1)[0][1] / len(actual)
    return {"match": _higher_is_better(hits, reference), "answered": len(answers), "distinct_outputs": len(set(actual))}
