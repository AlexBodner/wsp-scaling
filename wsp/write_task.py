"""A read-then-write task: from scrambled weights, edit GPT-2 so it stops repeating and keeps predicting ordinary text.

The reader never runs the model. Its edit is a Python block that changes W in place, run later in the same sandbox, and
scored by running the edited model.
"""

import re

import numpy as np
import torch

from wsp.disguise import HEAD_SIZE, WIDTH, DisguisedModel, model_from
from wsp.head_kinds import REPEAT_LENGTH
from wsp.network import TokenShuffle
from wsp.reading import python_prompt

TEXT_FILE = "data/tinystories/valid.txt"
TEXT_SEQUENCES = 64
TEXT_LENGTH = 48
REPEAT_SEQUENCES = 64
SEED = 20261104
EDITED_FILE = "edited.npz"
_EDIT_BLOCK = re.compile(r"```edit[ \t]*\n(.*?)```", re.DOTALL)
_SAVE_EDITS = """
_original = np.load("blocks.npz")
_changed = {name: W[name].astype(np.float32) for name in W if not np.array_equal(W[name], _original[name])}
np.savez(%r, **_changed)
print("changed", len(_changed), "tensors,", sum(int((_changed[name] != _original[name]).sum()) for name in _changed), "values")
""" % EDITED_FILE
TASK = """YOUR TASK: change a language model's weights so that it loses one ability and keeps the rest.

THE MODEL: its weights are those of a real trained language model. Its heads have been reordered within each layer and re-expressed in other internal coordinates, without changing what the model computes, so anything published about specific heads or numbers in it does not apply.

THE GOAL: when given random tokens followed by the same tokens again, the model predicts the next token of the repeat. Change the weights so that this falls as far as possible, while its predictions on ordinary English text change as little as possible.

HOW YOU ARE SCORED: we run the edited model. Repeat score: how often its top next token is right in the second copy of 50 random tokens repeated, lower is better. Text score: how much its cross-entropy rises on ordinary English stories, lower is better. Both count. It is never run for you, and you never see these numbers.

HOW YOU EDIT: in your final reply, before the answer block, give a code block whose opening line is three backticks followed by the word edit. It holds Python that changes the arrays in W in place, by any computation you like. Only the tensors in the blocks can change. We run it once, after you finish, in the same sandbox. Then, in the answer block, name the 5 heads your edit most depends on. If you are told a turn is your last, it still needs the edit block as well as the answer block.

"""


def write_prompt() -> str:
    """Returns the prompt, sharing everything after the task with the induction prompt, and with no example naming a layer."""
    base = python_prompt()
    prompt = TASK + base[base.index("THE NETWORK:") :]
    prompt = prompt.replace("W['L5.qkv.W'] is 768 x 2304, W['L5.up.W'] is 768 x 3072", "W['L0.qkv.W'] is 768 x 2304, W['L0.up.W'] is 768 x 3072")
    prompt = re.sub(r"```answer\n\d+\.\d+\n```", "```answer\n0.0\n```", prompt)
    if "L5" in prompt or "3.7" in prompt:
        raise ValueError("a layer-naming example survived in the prompt")
    return prompt


def edit_code(reply: str) -> str | None:
    """Returns the edit block's code wrapped to save what changed, or None if the reply has no edit block."""
    blocks = _EDIT_BLOCK.findall(reply)
    return blocks[-1] + _SAVE_EDITS if blocks else None


def apply_edit(disguised: DisguisedModel, edited: dict[str, np.ndarray]) -> DisguisedModel:
    """Returns the disguised model with edited tensors replaced. Only blocks can change."""
    tensors = dict(disguised.tensors)
    for name, values in edited.items():
        if name not in tensors or name in ("E", "P", "F.scale", "F.shift"):
            raise ValueError(f"the edit may not change {name}")
        if tensors[name].shape != values.shape:
            raise ValueError(f"the edit changed {name}'s shape")
        tensors[name] = values.astype(np.float64)
    return DisguisedModel(tensors, disguised.original_head)


def repeat_tokens(shuffle: TokenShuffle) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED)
    half = torch.randint(0, len(shuffle.tokenizer), (REPEAT_SEQUENCES, REPEAT_LENGTH), generator=generator)
    return torch.cat([half, half], 1)


def text_tokens(shuffle: TokenShuffle, path: str = TEXT_FILE) -> torch.Tensor:
    """Returns fixed-length runs of real stories, taken from the start of the validation file."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        text = handle.read(600_000)
    tokens = shuffle.encode(text)
    rng = np.random.default_rng(SEED)
    starts = rng.choice(len(tokens) - TEXT_LENGTH, TEXT_SEQUENCES, replace=False)
    return torch.tensor([tokens[int(start) : int(start) + TEXT_LENGTH] for start in starts])


@torch.no_grad()
def scores(model, repeats: torch.Tensor, text: torch.Tensor) -> tuple[float, float]:
    """Returns repeat accuracy and cross-entropy on real text."""
    batches, right = 8, []
    for part in repeats.chunk(batches):
        logits = model(part).logits[:, REPEAT_LENGTH:-1]
        right.append((logits.argmax(-1) == part[:, REPEAT_LENGTH + 1 :]).double().mean())
    total, count = 0.0, 0
    for part in text.chunk(batches):
        logits = model(part).logits[:, :-1]
        total += float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), part[:, 1:].reshape(-1), reduction="sum"))
        count += part[:, 1:].numel()
    return float(torch.stack(right).mean()), total / count


def score_disguised(disguised: DisguisedModel, shuffle: TokenShuffle) -> tuple[float, float]:
    return scores(model_from(disguised), repeat_tokens(shuffle), text_tokens(shuffle))


@torch.no_grad()
def detailed_scores(model, unedited, shuffle: TokenShuffle) -> dict[str, float]:
    """Returns more ways to see a change, because top-1 can fall while copying survives."""
    repeats, text = repeat_tokens(shuffle), text_tokens(shuffle)
    repeat_loss = text_divergence = 0.0
    for part in repeats.chunk(8):
        logits = model(part).logits[:, REPEAT_LENGTH:-1]
        repeat_loss += float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), part[:, REPEAT_LENGTH + 1 :].reshape(-1))) / 8
    for part in text.chunk(8):
        log_new = model(part).logits.log_softmax(-1)
        log_old = unedited(part).logits.log_softmax(-1)
        text_divergence += float((log_old.exp() * (log_old - log_new)).sum(-1).mean()) / 8
    pool = text.ravel()
    generator = torch.Generator().manual_seed(SEED)
    half = pool[torch.randint(0, len(pool), (REPEAT_SEQUENCES, REPEAT_LENGTH), generator=generator)]
    common = torch.cat([half, half], 1)
    right = []
    for part in common.chunk(8):
        logits = model(part).logits[:, REPEAT_LENGTH:-1]
        right.append((logits.argmax(-1) == part[:, REPEAT_LENGTH + 1 :]).double().mean())
    return {"repeat_loss": repeat_loss, "text_divergence_from_unedited": text_divergence, "repeat_accuracy_on_common_tokens": float(torch.stack(right).mean())}


def touched_heads(disguised: DisguisedModel, edited: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    """Returns the scrambled heads whose query, key, value or output weights the edit changed."""
    touched: set[tuple[int, int]] = set()
    for name, values in edited.items():
        before = disguised.tensors[name].astype(np.float32)
        layer = int(name[1 : name.index(".")])
        if name.endswith("qkv.W"):
            columns = np.nonzero((values != before).any(0))[0]
            touched |= {(layer, (int(column) % WIDTH) // HEAD_SIZE) for column in columns}
        elif name.endswith("out.W"):
            rows = np.nonzero((values != before).any(1))[0]
            touched |= {(layer, int(row) // HEAD_SIZE) for row in rows}
    return sorted(touched)


def zero_heads(disguised: DisguisedModel, heads: list[tuple[int, int]]) -> DisguisedModel:
    """Returns the model with these scrambled heads' output rows zeroed: deletion, with no further understanding."""
    tensors = dict(disguised.tensors)
    for layer, head in heads:
        name = f"L{layer}.out.W"
        values = tensors[name].copy()
        values[head * HEAD_SIZE : (head + 1) * HEAD_SIZE] = 0
        tensors[name] = values
    return DisguisedModel(tensors, disguised.original_head)
