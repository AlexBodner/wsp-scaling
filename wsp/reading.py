"""The reading test: name a disguised GPT-2's copying heads, reading its weights through a viewer."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from wsp.disguise import HEADS, LAYERS, DisguisedModel, disguise
from wsp.network import TokenShuffle
from wsp.prompts import NETWORK_OPENING, NETWORK_STRUCTURE
from wsp.viewer import WeightViewer, parse_answer

READ_BUDGET = 20000
MAX_TURNS = 6
HEADS_TO_NAME = 5
POSITIVE_HEADS = 10
PASS_HITS = 6
PASS_OUT_OF = 10
READ_SEEDS = (1001, 1002)
BLIND_SEED = 1001
HINTED_SEED = 1003
_TRUTH_SEEDS = (0, 1, 2)
_TRUTH_BATCH = 32
_REPEAT_LENGTH = 50
_TRUTH_TOKEN_RANGE = (1000, 50000)

_GELU_NOTE = "gelu here is the standard smooth approximation of u * Phi(u), which differs from it by at most 0.0005."

_TASK = f"""YOUR TASK: find the copying heads of a language model.

THE MODEL: its weights are those of a real trained language model. Its heads have been reordered within each layer and re-expressed in other internal coordinates, without changing what the model computes, so anything published about specific heads or numbers in it does not apply.

THE QUESTION: take {_REPEAT_LENGTH} random tokens followed by the same {_REPEAT_LENGTH} tokens again. For a head, at each position t in the second copy except the last, take its attention weight from position t to position t-{_REPEAT_LENGTH}+1, which is the token just after the earlier copy of the current token. Its score is the average of that weight over those positions and over many random sequences. Name the {HEADS_TO_NAME} heads, out of all {LAYERS * HEADS}, that you judge to have the highest scores.

HOW YOU ARE SCORED: the number of heads you name that are among the {POSITIVE_HEADS} heads with the truly highest scores, measured by running the model. It is never run for you.
"""

_CALCULATION_LIMITS = "You have no code execution, no calculator and no way to run the model. All arithmetic is your own."

_VIEWER = f"""HOW TO READ THE WEIGHTS: you may read any part of any tensor, with a budget of {READ_BUDGET:,} values for the whole task and at most {MAX_TURNS} turns. To read, end your reply with a code block whose opening line is three backticks followed by the word read, with one request per line:

```read
L5.qkv.W[0:4, 0:64]
L5.qkv.b[64:128]
E[17, :]
```
a:b is half-open, a:b:s takes every s-th index, and ':' alone is the whole axis. Each value returned is charged to the budget, and it is shown rounded to 3 significant digits. A request over the remaining budget is refused and not charged. Token IDs are arbitrary. Each turn you get back your earlier written replies and the values returned, but not your private reasoning, so write down anything you want to keep.
"""

_ANSWER = f"""HOW TO ANSWER: when you are done, end your reply with a code block whose opening line is three backticks followed by the word answer, with one head per line as layer.head, counting from 0, for example:

```answer
3.7
```
Name {HEADS_TO_NAME} heads. Your first reply with an answer block ends the task.
"""

_HINT = """A HINT: for one head, write Q for its 64-column part of qkv.W at the query offset and K for the part at the key offset. Summing the products of Q and K over the same input rows, Q[r] . K[r], for a sample of rows r, tends to be larger for heads that attend to the token after an earlier copy of the current token.
"""


def _structure() -> str:
    return NETWORK_STRUCTURE.replace(
        "gelu(u) = u * Phi(u), element by element, where Phi is the standard normal cumulative distribution.", _GELU_NOTE
    )


def read_prompt() -> str:
    """Returns the opening prompt of a reading run."""
    return "\n".join((_TASK, NETWORK_OPENING, "", _structure(), _CALCULATION_LIMITS, "", _VIEWER, _ANSWER))


def hinted_prompt() -> str:
    """Returns the opening prompt of the hinted run, which names a statistic to compute."""
    return "\n".join((_TASK, NETWORK_OPENING, "", _structure(), _CALCULATION_LIMITS, "", _HINT, _VIEWER, _ANSWER))


def blind_prompt() -> str:
    """Returns the prompt of the blind run: the same question with no weights to read."""
    limits = f"{_CALCULATION_LIMITS} You cannot read any weights in this task. This is a single response."
    return "\n".join((_TASK, NETWORK_OPENING, "", _structure(), limits, "", _ANSWER))


@torch.no_grad()
def head_scores(model: GPT2LMHeadModel) -> np.ndarray:
    """Measures each head's score on the question by running the model: layer x head."""
    total = np.zeros((LAYERS, HEADS))
    for seed in _TRUTH_SEEDS:
        generator = torch.Generator().manual_seed(seed)
        tokens = torch.randint(*_TRUTH_TOKEN_RANGE, (_TRUTH_BATCH, _REPEAT_LENGTH), generator=generator)
        attentions = model(torch.cat([tokens, tokens], 1), output_attentions=True).attentions
        for layer, attention in enumerate(attentions):
            for t in range(_REPEAT_LENGTH, 2 * _REPEAT_LENGTH - 1):
                total[layer] += attention[:, :, t, t - _REPEAT_LENGTH + 1].double().mean(0).numpy()
    return total / (len(_TRUTH_SEEDS) * (_REPEAT_LENGTH - 1))


def disguised_scores(real_scores: np.ndarray, disguised: DisguisedModel) -> np.ndarray:
    """Returns each disguised head's score, given the scores measured on the real model."""
    return np.stack([real_scores[layer, disguised.original_head[layer]] for layer in range(LAYERS)])


def positive_heads(scores: np.ndarray) -> set[tuple[int, int]]:
    """Returns the heads with the truly highest scores."""
    order = np.argsort(-scores.ravel())[:POSITIVE_HEADS]
    return {(int(i) // HEADS, int(i) % HEADS) for i in order}


@dataclass
class ReadingResult:
    """What one run did and how it scored."""

    seed: int
    budget: int
    turns: int
    values_read: int
    answer: list[tuple[int, int]] | None
    hits: int
    requests: list[dict]


def run_reading(
    prompt: str,
    disguised: DisguisedModel,
    positives: set[tuple[int, int]],
    seed: int,
    budget: int,
    call: Callable[[str, str], str],
    run_dir: Path,
) -> ReadingResult:
    """Runs the reader for up to the turn limit and saves every turn.

    Args:
        prompt: The opening prompt.
        disguised: The disguised model the viewer reads.
        positives: The true top heads, which only the scorer sees.
        seed: The disguise seed, for the record.
        budget: Values the reader may read. 0 means a blind single reply.
        call: Sends one prompt and returns the reply: a sealed Astra call, or a scripted reader.
        run_dir: Where the run is saved.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    viewer = WeightViewer(disguised.tensors, budget)
    transcript = ""
    answer = None
    turns = 0
    turn_limit = MAX_TURNS if budget else 1
    with (run_dir / "transcript.jsonl").open("a", encoding="utf-8") as log:
        while turns < turn_limit and answer is None:
            turns += 1
            full_prompt = prompt + transcript
            reply = call(full_prompt, f"turn-{turns}")
            answer = parse_answer(reply, HEADS_TO_NAME)
            served = None if answer is not None or not budget else viewer.serve(reply)
            log.write(json.dumps({"turn": turns, "reply": reply, "viewer": served}) + "\n")
            log.flush()
            if served is not None:
                left = turn_limit - turns
                transcript += f"\n\nYOUR REPLY IN TURN {turns}:\n{reply}\n\nVIEWER RESPONSE:\n{served}\n\nTurns remaining: {left}."
    result = ReadingResult(
        seed=seed,
        budget=budget,
        turns=turns,
        values_read=budget - viewer.remaining,
        answer=answer,
        hits=sum(head in positives for head in (answer or [])),
        requests=viewer.requests,
    )
    (run_dir / "result.json").write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result
