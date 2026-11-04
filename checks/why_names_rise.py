"""Tests why repeated names get likelier when the top induction heads are zeroed. Makes no Astra calls.

Those heads push the right name up directly, so zeroing them should lower it. The idea tested here is that they also
tell late heads what to look for, and that some of those late heads then push the name down.
"""
import numpy as np
import torch

from checks.signed_ladder import FIRST, LAST, TEMPLATES, TOP_INDUCTION, name, silenced
from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH
from wsp.network import TokenShuffle, real_network

SUPPRESSORS = [(10, 7), (11, 10)]
BOOSTERS = [(9, 6), (9, 9), (10, 0), (10, 1)]
LAST_LAYERS = [(layer, head) for layer in range(8, LAYERS) for head in range(HEADS)]
SEED = 20261104
RANDOM_SETS = 8


@torch.no_grad()
def scored(model, tokens: torch.Tensor, answers: torch.Tensor, hooks: list) -> tuple[float, float]:
    rows = torch.arange(len(answers))
    logits = model(tokens).logits[:, -1]
    for hook in hooks:
        hook.remove()
    return float((logits.argmax(-1) == answers).double().mean()), float(logits.softmax(-1)[rows, answers].mean())


@torch.no_grad()
def query_patch_hooks(model, tokens: torch.Tensor, sources: list, targets: list) -> list:
    """Returns hooks that give only the targets' queries the residual they'd see with the sources silenced."""
    cached, hooks = {}, []
    for layer in {layer for layer, _ in targets}:
        def store(_module, inputs, layer=layer):
            cached[layer] = inputs[0].detach()

        hooks.append(model.transformer.h[layer].attn.c_attn.register_forward_pre_hook(store))
    hooks += silenced(model, sources)
    model(tokens)
    for hook in hooks:
        hook.remove()
    hooks = []
    for layer in {layer for layer, _ in targets}:
        chosen = [head for current, head in targets if current == layer]

        def replace_queries(module, _inputs, output, layer=layer, chosen=chosen):
            output = output.clone()
            changed = cached[layer] @ module.weight + module.bias
            for head in chosen:
                start = head * HEAD_SIZE
                output[..., start : start + HEAD_SIZE] = changed[..., start : start + HEAD_SIZE]
            return output

        hooks.append(model.transformer.h[layer].attn.c_attn.register_forward_hook(replace_queries))
    return hooks


def main() -> None:
    shuffle = TokenShuffle()
    model = real_network(shuffle).double()
    rng = np.random.default_rng(SEED)
    single = lambda word: len(shuffle.encode(f" {word}")) == 1
    pairs = [(a, b) for a in FIRST if single(a) for b in LAST if single(b)]
    print("on top/probability of the right surname")
    for label, template in TEMPLATES.items():
        encoded = [(shuffle.encode(template.format(a=a, b=b)), shuffle.encode(f" {b}")[0]) for a, b in pairs]
        length = max(set(len(p) for p, _ in encoded), key=[len(p) for p, _ in encoded].count)
        kept = [(p, a) for p, a in encoded if len(p) == length]
        tokens, answers = torch.tensor([p for p, _ in kept]), torch.tensor([a for _, a in kept])
        builders = {
            "nothing": lambda: [],
            "five zeroed": lambda: silenced(model, TOP_INDUCTION),
            "10.7 and 11.10 zeroed": lambda: silenced(model, SUPPRESSORS),
            "9.6 9.9 10.0 10.1 zeroed": lambda: silenced(model, BOOSTERS),
            "five zeroed for 10.7 and 11.10's queries only": lambda: query_patch_hooks(model, tokens, TOP_INDUCTION, SUPPRESSORS),
            "five zeroed for the boosters' queries only": lambda: query_patch_hooks(model, tokens, TOP_INDUCTION, BOOSTERS),
            "five zeroed for queries in layers 8 to 11": lambda: query_patch_hooks(model, tokens, TOP_INDUCTION, LAST_LAYERS),
        }
        results = {condition: scored(model, tokens, answers, build()) for condition, build in builders.items()}
        others = [(layer, head) for layer in range(4, 8) for head in range(HEADS) if (layer, head) not in TOP_INDUCTION]
        controls = np.array([scored(model, tokens, answers, query_patch_hooks(model, tokens, [others[i] for i in rng.choice(len(others), 5, replace=False)], LAST_LAYERS)) for _ in range(RANDOM_SETS)])
        print(f"{label} (n={len(kept)})")
        for condition, (top, probability) in results.items():
            print(f"  {condition}: {top:.2f}/{probability:.2f}")
        print(f"  {RANDOM_SETS} random sets of 5 in layers 4 to 7, same query path: {controls[:, 0].mean():.2f}/{controls[:, 1].mean():.2f}, highest probability {controls[:, 1].max():.2f}")


if __name__ == "__main__":
    main()
