"""The test sets, trivial references and scores for the three writing tasks."""

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from wsp.network import TokenShuffle

_CAPITALS_FILE = Path(__file__).parent / "capitals.json"
_DATA = Path(__file__).parent.parent / "data" / "tinystories"
VISIBLE_PHRASINGS = ("The capital of {country} is", "{country}'s capital is")
HIDDEN_PHRASINGS = ("In {country}, the capital city is", "What is the capital of {country}? The answer is")
REVERSE_TEST_SIZE = 1000
REVERSE_LENGTHS = (3, 6)
REVERSE_SEED = 7
LANGUAGE_STORIES = 1000
LANGUAGE_MAX_TOKENS = 512
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 11
LETTERS = "abcdefghijklmnopqrstuvwxyz"
_STORY_SEPARATOR = "<|endoftext|>"


@dataclass
class Score:
    """One network's result on one measure, with its 95% interval and trivial reference."""

    value: float
    low: float
    high: float
    reference: float
    passes: bool


def _bootstrap_mean(per_unit: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    picks = rng.integers(0, len(per_unit), (BOOTSTRAP_RESAMPLES, len(per_unit)))
    means = per_unit[picks].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _higher_is_better(per_unit: np.ndarray, reference: float) -> Score:
    low, high = _bootstrap_mean(per_unit)
    return Score(float(per_unit.mean()), low, high, reference, low > reference)


@torch.no_grad()
def top_tokens(model, prompts: list[list[int]]) -> list[int]:
    """Returns each prompt's top next token, with ties going to the lowest ID."""
    results = [0] * len(prompts)
    by_length: dict[int, list[int]] = {}
    for position, prompt in enumerate(prompts):
        by_length.setdefault(len(prompt), []).append(position)
    for length, positions in by_length.items():
        if length == 0:
            raise ValueError("a prompt has no tokens")
        batch = torch.tensor([prompts[position] for position in positions])
        best = model(input_ids=batch).logits[:, -1].argmax(dim=-1)
        for position, token in zip(positions, best.tolist()):
            results[position] = token
    return results


def load_capitals() -> list[tuple[str, str]]:
    """Returns the fixed (country, capital) test pairs."""
    return [tuple(pair) for pair in json.loads(_CAPITALS_FILE.read_text())]


def capital_tokens(capital: str, shuffle: TokenShuffle) -> list[int]:
    """Returns the shuffled tokens of a capital's name as it follows a prompt."""
    return shuffle.encode(f" {capital}")


def score_capitals(model, shuffle: TokenShuffle) -> dict:
    """Scores the visible and hidden phrasings, and well-known and obscure capitals separately."""
    pairs = load_capitals()
    answers = [capital_tokens(capital, shuffle)[0] for _, capital in pairs]
    reference = Counter(answers).most_common(1)[0][1] / len(answers)
    single_token = np.array([len(capital_tokens(capital, shuffle)) == 1 for _, capital in pairs])

    def hits(phrasings: tuple[str, ...]) -> np.ndarray:
        rows = []
        for phrasing in phrasings:
            prompts = [shuffle.encode(phrasing.format(country=country)) for country, _ in pairs]
            rows.append(np.array(top_tokens(model, prompts)) == np.array(answers))
        return np.stack(rows).mean(axis=0)

    visible, hidden = hits(VISIBLE_PHRASINGS), hits(HIDDEN_PHRASINGS)
    return {
        "countries": len(pairs),
        "visible": _higher_is_better(visible, reference),
        "hidden": _higher_is_better(hidden, reference),
        "visible_one_token_capitals": float(visible[single_token].mean()),
        "visible_multi_token_capitals": float(visible[~single_token].mean()),
        "one_token_capitals": int(single_token.sum()),
    }


def reverse_test_strings() -> list[str]:
    """Returns the fixed random strings of the reverse test."""
    rng = random.Random(REVERSE_SEED)
    low, high = REVERSE_LENGTHS
    return ["".join(rng.choices(LETTERS, k=rng.randint(low, high))) for _ in range(REVERSE_TEST_SIZE)]


def reverse_prompt(letters: str) -> str:
    """Returns the prompt text for one string."""
    return f"rev: {' '.join(letters)} ->"


@torch.no_grad()
def _generate(model, prompts: list[list[int]], steps: list[int]) -> list[list[int]]:
    outputs: list[list[int]] = [[] for _ in prompts]
    current = [list(prompt) for prompt in prompts]
    for step in range(max(steps)):
        active = [i for i, wanted in enumerate(steps) if step < wanted]
        for i, token in zip(active, top_tokens(model, [current[i] for i in active])):
            outputs[i].append(token)
            current[i].append(token)
    return outputs


def score_reverse(model, shuffle: TokenShuffle) -> dict:
    """Scores exact whole-string reversal, against echoing the input or a constant answer."""
    strings = reverse_test_strings()
    answers = [[shuffle.encode(f" {letter}")[0] for letter in reversed(text)] for text in strings]
    if any(len(shuffle.encode(f" {letter}")) != 1 for letter in LETTERS):
        raise ValueError("a spaced letter is not one token, so the reverse test is not well defined")
    prompts = [shuffle.encode(reverse_prompt(text)) for text in strings]
    outputs = _generate(model, prompts, [len(text) for text in strings])
    exact = np.array([output == answer for output, answer in zip(outputs, answers)], dtype=float)
    echo = sum(text == text[::-1] for text in strings) / len(strings)
    constant = Counter(text[::-1] for text in strings).most_common(1)[0][1] / len(strings)
    positions = [output[i] == answer[i] for output, answer in zip(outputs, answers) for i in range(len(answer))]
    return {
        "exact": _higher_is_better(exact, max(echo, constant)),
        "per_position": float(np.mean(positions)),
    }


def language_stories(shuffle: TokenShuffle) -> list[list[int]]:
    """Returns the fixed held-out stories as shuffled tokens, cut to the length limit."""
    text = (_DATA / "valid.txt").read_text(encoding="utf-8")
    stories = [story.strip() for story in text.split(_STORY_SEPARATOR) if story.strip()]
    if len(stories) < LANGUAGE_STORIES:
        raise ValueError(f"expected at least {LANGUAGE_STORIES} stories, found {len(stories)}")
    return [shuffle.encode(story)[:LANGUAGE_MAX_TOKENS] for story in stories[:LANGUAGE_STORIES]]


def token_counts(shuffle: TokenShuffle) -> np.ndarray:
    """Returns how often each shuffled token occurs in the first 100 MB of the training split."""
    text = (_DATA / "train_first100mb.txt").read_text(encoding="utf-8", errors="ignore").replace(_STORY_SEPARATOR, "\n")
    counts = np.zeros(len(shuffle.tokenizer), dtype=np.int64)
    chunks = [text[start : start + 1_000_000] for start in range(0, len(text), 1_000_000)]
    for encoded in shuffle.tokenizer(chunks, add_special_tokens=False).input_ids:
        np.add.at(counts, shuffle.to_shuffled[encoded], 1)
    return counts


@torch.no_grad()
def story_bits(model, stories: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Returns each story's total bits of next-token loss and its number of predicted tokens."""
    bits, tokens = [], []
    for story in stories:
        ids = torch.tensor([story])
        logits = model(input_ids=ids).logits[0, :-1]
        loss = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum")
        bits.append(loss.item() / np.log(2))
        tokens.append(len(story) - 1)
    return np.array(bits), np.array(tokens)


def frequency_bits(stories: list[list[int]], counts: np.ndarray) -> np.ndarray:
    """Returns each story's total bits under add-one-smoothed token frequencies."""
    probability = (counts + 1) / (counts + 1).sum()
    return np.array([-np.log2(probability[story[1:]]).sum() for story in stories])


def score_language(model, stories: list[list[int]], reference_bits: np.ndarray) -> dict:
    """Scores bits per token, and the paired difference from the token frequency reference."""
    bits, tokens = story_bits(model, stories)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    picks = rng.integers(0, len(stories), (BOOTSTRAP_RESAMPLES, len(stories)))
    per_token = bits[picks].sum(axis=1) / tokens[picks].sum(axis=1)
    difference = (bits - reference_bits)[picks].sum(axis=1) / tokens[picks].sum(axis=1)
    low, high = np.percentile(difference, [2.5, 97.5])
    reference = float(reference_bits.sum() / tokens.sum())
    return {
        "bits_per_token": Score(
            float(bits.sum() / tokens.sum()),
            float(np.percentile(per_token, 2.5)),
            float(np.percentile(per_token, 97.5)),
            reference,
            bool(high < 0),
        ),
        "difference_interval": (float(low), float(high)),
    }
