"""Tests whether GPT-2's top-scoring heads cause repeating: silences heads and measures repeat accuracy on repeated random tokens."""
import numpy as np
import torch

from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH
from wsp.network import TokenShuffle, real_network
from wsp.reading import head_scores

SEQUENCE_LENGTH = 50
BATCH = 64
SEED = 20261103
RANDOM_SETS = 20
HEADS_SILENCED = 5


def silence(model, heads: list[tuple[int, int]]) -> list:
    """Zeroes the value vectors of the given heads, and returns the hooks to remove afterwards."""
    hooks = []
    for layer in {layer for layer, _ in heads}:
        chosen = [head for current, head in heads if current == layer]

        def zero_values(_module, _inputs, output, chosen=chosen):
            output = output.clone()
            for head in chosen:
                start = 2 * WIDTH + head * HEAD_SIZE
                output[..., start : start + HEAD_SIZE] = 0
            return output

        hooks.append(model.transformer.h[layer].attn.c_attn.register_forward_hook(zero_values))
    return hooks


@torch.no_grad()
def repeat_accuracy(model, tokens: torch.Tensor) -> float:
    """Returns how often the top next token in the second copy is the right one."""
    logits = model(input_ids=tokens).logits
    predicted = logits[:, SEQUENCE_LENGTH : -1].argmax(dim=-1)
    return float((predicted == tokens[:, SEQUENCE_LENGTH + 1 :]).float().mean())


def accuracy_without(model, tokens: torch.Tensor, heads: list[tuple[int, int]]) -> float:
    hooks = silence(model, heads)
    try:
        return repeat_accuracy(model, tokens)
    finally:
        for hook in hooks:
            hook.remove()


shuffle = TokenShuffle()
model = real_network(shuffle)
scoring = real_network(shuffle).double()
scoring.set_attn_implementation("eager")
order = np.argsort(-head_scores(scoring).ravel())
ranked = [(int(i) // HEADS, int(i) % HEADS) for i in order]
generator = torch.Generator().manual_seed(SEED)
half = torch.randint(0, len(shuffle.tokenizer), (BATCH, SEQUENCE_LENGTH), generator=generator)
tokens = torch.cat([half, half], dim=1)
rng = np.random.default_rng(SEED)
print("top five:", ranked[:5], " next five:", ranked[5:10])
print(f"nothing silenced: {repeat_accuracy(model, tokens):.3f}")
print(f"top five silenced: {accuracy_without(model, tokens, ranked[:5]):.3f}")
print(f"next five silenced: {accuracy_without(model, tokens, ranked[5:10]):.3f}")
others = ranked[10:]
scores = [accuracy_without(model, tokens, [others[i] for i in rng.choice(len(others), HEADS_SILENCED, replace=False)]) for _ in range(RANDOM_SETS)]
print(f"5 other random heads silenced, {RANDOM_SETS} sets: mean {np.mean(scores):.3f}, lowest {min(scores):.3f}, highest {max(scores):.3f}")
