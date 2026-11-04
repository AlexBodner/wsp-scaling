"""Tests two open questions on real GPT-2. Makes no Astra calls.

1. Do late heads form a further stage after the induction heads, with heads that push against what they attend to?
2. Why do repeated names in sentences get likelier when the top induction heads are zeroed? This repeats that test with
   names that are one token each, and measures what those heads do directly to the answer.
"""
import numpy as np
import torch

from checks.head_types import ioi_prompts
from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH
from wsp.network import TokenShuffle, real_network

SEED = 20261104
REPEAT_LENGTH = 25
BATCH = 32
TOP_INDUCTION = [(5, 5), (7, 10), (6, 9), (5, 1), (7, 2)]
LATE_HEADS = [(9, 6), (9, 9), (10, 0), (10, 1), (10, 7), (11, 10)]
SHOWN = 6
RANDOM_SETS = 8
FIRST = "Mary John Anna Tom Lisa Mark Sara Paul Kate Peter Emma David Laura James Alice Henry".split()
LAST = "Smith Jones Brown Taylor Wilson Clark Hall Allen Young King Scott Green Baker Adams Hill Moore".split()
TEMPLATES = {
    "name again in a new sentence": "{a} {b} went home, and {a}",
    "name again after a longer gap": "I met {a} {b} at the station. Later that day, {a}",
    "name again after 'according to'": "The report was written by {a} {b}. According to {a}",
    "same sentence repeated": "{a} {b} went home early. {a}",
    "name listed three times": "{a} {b}, {a} {b}, {a}",
}


def name(head: tuple[int, int]) -> str:
    return f"{head[0]}.{head[1]}"


def silenced(model, heads: list[tuple[int, int]]) -> list:
    """Zeroes the given heads' value vectors, and returns the hooks to remove afterwards."""
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
def run_with_heads(model, tokens: torch.Tensor):
    """Runs the model, and returns its output, each head's output vectors by layer, and the final norm's input."""
    values, final_input, hooks = {}, {}, []
    for layer in range(LAYERS):
        def store(_module, _inputs, output, layer=layer):
            values[layer] = output[..., 2 * WIDTH :].detach()

        hooks.append(model.transformer.h[layer].attn.c_attn.register_forward_hook(store))

    def store_final(_module, inputs):
        final_input["x"] = inputs[0].detach()

    hooks.append(model.transformer.ln_f.register_forward_pre_hook(store_final))
    output = model(tokens, output_attentions=True)
    for hook in hooks:
        hook.remove()
    per_head = {}
    for layer in range(LAYERS):
        batch, length, _ = values[layer].shape
        v = values[layer].view(batch, length, HEADS, HEAD_SIZE).transpose(1, 2)
        attended = output.attentions[layer] @ v
        out = model.transformer.h[layer].attn.c_proj.weight
        per_head[layer] = torch.stack([attended[:, head] @ out[head * HEAD_SIZE : (head + 1) * HEAD_SIZE] for head in range(HEADS)], 1)
    return output, per_head, final_input["x"]


def direct_effect(model, per_head: dict, final_input: torch.Tensor, positions: torch.Tensor, up: torch.Tensor, down: torch.Tensor | None = None) -> np.ndarray:
    """Returns each head's direct effect on a token's logit at given positions, through the final norm, averaged over the batch.

    With no `down` token, the logit is measured against the mean over the whole vocabulary.
    """
    rows = torch.arange(len(positions))
    scale = model.transformer.ln_f.weight
    deviation = final_input[rows, positions].std(-1, keepdim=True)
    unembed = model.lm_head.weight
    direction = unembed[up] - (unembed[down] if down is not None else unembed.mean(0, keepdim=True))
    effect = np.zeros((LAYERS, HEADS))
    for layer, vectors in per_head.items():
        chosen = vectors[rows, :, positions] / deviation[:, None] * scale
        effect[layer] = (chosen * direction[:, None]).sum(-1).mean(0).detach().numpy()
    return effect


@torch.no_grad()
def path_attention(model, tokens: torch.Tensor, sources: list, targets: list, offset: int):
    """Gives only one part of the target heads (0 query, 1 key) the residual they'd see with the sources silenced."""
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

        def replace_part(module, _inputs, output, layer=layer, chosen=chosen):
            output = output.clone()
            changed = cached[layer] @ module.weight + module.bias
            for head in chosen:
                start = offset * WIDTH + head * HEAD_SIZE
                output[..., start : start + HEAD_SIZE] = changed[..., start : start + HEAD_SIZE]
            return output

        hooks.append(model.transformer.h[layer].attn.c_attn.register_forward_hook(replace_part))
    output = model(tokens, output_attentions=True)
    for hook in hooks:
        hook.remove()
    return output


def ladder(model, shuffle: TokenShuffle, rng: np.random.Generator) -> None:
    generator = torch.Generator().manual_seed(SEED)
    half = torch.randint(0, len(shuffle.tokenizer), (BATCH, REPEAT_LENGTH), generator=generator)
    tokens = torch.cat([half, half], 1)
    output, per_head, final_input = run_with_heads(model, tokens)
    totals = np.zeros((LAYERS, HEADS))
    for t in range(REPEAT_LENGTH, 2 * REPEAT_LENGTH - 1):
        positions = torch.full((BATCH,), t)
        totals += direct_effect(model, per_head, final_input, positions, tokens[:, t + 1])
    repeat_effect = totals / (REPEAT_LENGTH - 1)

    prompts = ioi_prompts(shuffle)
    length = len(prompts[0][0])
    prompts = [prompt for prompt in prompts if len(prompt[0]) == length]
    ioi_tokens = torch.tensor([prompt[0] for prompt in prompts])
    receiver_at = torch.tensor([prompt[1] for prompt in prompts])
    receiver, subject = torch.tensor([prompt[4] for prompt in prompts]), torch.tensor([prompt[5] for prompt in prompts])
    rows = torch.arange(len(prompts))
    ioi_output, ioi_heads, ioi_final = run_with_heads(model, ioi_tokens)
    ioi_effect = direct_effect(model, ioi_heads, ioi_final, torch.full((len(prompts),), length - 1), receiver, subject)

    print("1a. Each head's direct effect on the logit")
    for label, effect in (("the right next token on repeats", repeat_effect), ("the right name over the subject on sentences", ioi_effect)):
        order = np.argsort(effect.ravel())
        print(f"  {label}, most negative: " + ", ".join(f"{name((int(i) // HEADS, int(i) % HEADS))} {effect.ravel()[i]:.2f}" for i in order[:SHOWN]))
        print(f"  {label}, most positive: " + ", ".join(f"{name((int(i) // HEADS, int(i) % HEADS))} {effect.ravel()[i]:.2f}" for i in order[::-1][:SHOWN]))
    print("  named heads, on repeats / on sentences: " + ", ".join(f"{name(h)} {repeat_effect[h]:.2f}/{ioi_effect[h]:.2f}" for h in TOP_INDUCTION + LATE_HEADS))

    def induction_attention(result, head) -> float:
        attention = result.attentions[head[0]][:, head[1]]
        return float(np.mean([attention[:, t, t - REPEAT_LENGTH + 1].mean() for t in range(REPEAT_LENGTH, 2 * REPEAT_LENGTH - 1)]))

    def receiver_attention(result, head) -> float:
        return float(result.attentions[head[0]][rows, head[1], -1, receiver_at].mean())

    others = [(int(i) // HEADS, int(i) % HEADS) for i in range(4 * HEADS, 9 * HEADS) if (int(i) // HEADS, int(i) % HEADS) not in TOP_INDUCTION]
    print("1b. Late heads' attention on repeats, with the top five silenced only in that part of the late heads")
    for label, offset in (("queries", 0), ("keys", 1)):
        result = path_attention(model, tokens, TOP_INDUCTION, LATE_HEADS, offset)
        print(f"  {label}: " + ", ".join(f"{name(h)} {induction_attention(output, h):.2f} to {induction_attention(result, h):.2f}" for h in LATE_HEADS))
    controls = np.zeros((RANDOM_SETS, len(LATE_HEADS)))
    for index in range(RANDOM_SETS):
        picked = [others[i] for i in rng.choice(len(others), len(TOP_INDUCTION), replace=False)]
        result = path_attention(model, tokens, picked, LATE_HEADS, 0)
        controls[index] = [induction_attention(result, h) for h in LATE_HEADS]
    print(f"  {RANDOM_SETS} random sets of 5 heads in layers 4 to 8, queries: " + ", ".join(f"{name(h)} mean {controls[:, i].mean():.2f} lowest {controls[:, i].min():.2f}" for i, h in enumerate(LATE_HEADS)))

    print("1c. On sentences, attention at the end to the right name, with name movers silenced only in late heads' queries")
    movers = [(9, 9), (9, 6), (10, 0)]
    negatives = [(10, 7), (11, 10)]
    for label, offset in (("queries", 0), ("keys", 1)):
        result = path_attention(model, ioi_tokens, movers, negatives, offset)
        print(f"  {label}: " + ", ".join(f"{name(h)} {receiver_attention(ioi_output, h):.2f} to {receiver_attention(result, h):.2f}" for h in negatives))
    controls = np.zeros((RANDOM_SETS, len(negatives)))
    others = [(9, h) for h in range(12) if (9, h) not in movers] + [(8, h) for h in range(12)]
    for index in range(RANDOM_SETS):
        picked = [others[i] for i in rng.choice(len(others), len(movers), replace=False)]
        result = path_attention(model, ioi_tokens, picked, negatives, 0)
        controls[index] = [receiver_attention(result, h) for h in negatives]
    print(f"  {RANDOM_SETS} random sets of 3 heads in layers 8 and 9, queries: " + ", ".join(f"{name(h)} mean {controls[:, i].mean():.2f} lowest {controls[:, i].min():.2f}" for i, h in enumerate(negatives)))


def names_test(model, shuffle: TokenShuffle, rng: np.random.Generator) -> None:
    single = lambda word: len(shuffle.encode(f" {word}")) == 1
    first, last = [w for w in FIRST if single(w)], [w for w in LAST if single(w)]
    print(f"2. Repeated names, one token each ({len(first)} first names, {len(last)} surnames)")
    others = [(layer, head) for layer in range(LAYERS) for head in range(HEADS) if (layer, head) not in TOP_INDUCTION + LATE_HEADS]
    random_sets = [[others[i] for i in rng.choice(len(others), len(TOP_INDUCTION), replace=False)] for _ in range(RANDOM_SETS)]
    for label, template in TEMPLATES.items():
        prompts = [shuffle.encode(template.format(a=a, b=b)) for a in first for b in last]
        length = max(set(map(len, prompts)), key=list(map(len, prompts)).count)
        kept = [(p, shuffle.encode(f" {b}")[0]) for p, (a, b) in zip(prompts, [(a, b) for a in first for b in last]) if len(p) == length]
        tokens = torch.tensor([p for p, _ in kept])
        answers = torch.tensor([a for _, a in kept])
        rows = torch.arange(len(kept))

        def scored(heads) -> tuple[float, float]:
            hooks = silenced(model, heads)
            with torch.no_grad():
                logits = model(tokens).logits[:, -1]
            for hook in hooks:
                hook.remove()
            return float((logits.argmax(-1) == answers).double().mean()), float(logits.softmax(-1)[rows, answers].mean())

        base, zeroed, late = scored([]), scored(TOP_INDUCTION), scored(LATE_HEADS)
        random_scores = np.mean([scored(heads) for heads in random_sets], axis=0)
        output, per_head, final_input = run_with_heads(model, tokens)
        effect = direct_effect(model, per_head, final_input, torch.full((len(kept),), length - 1), answers)
        surname_position = torch.tensor([p.index(a) for p, a in kept])
        attention = np.mean([float(output.attentions[h[0]][rows, h[1], -1, surname_position].mean()) for h in TOP_INDUCTION])
        print(f"  {label} (n={len(kept)}): on top/probability, nothing {base[0]:.2f}/{base[1]:.2f}, five zeroed {zeroed[0]:.2f}/{zeroed[1]:.2f}, late heads zeroed {late[0]:.2f}/{late[1]:.2f}, 5 others {random_scores[0]:.2f}/{random_scores[1]:.2f}")
        print(f"    the five attend to the surname {attention:.2f}; direct effect on its logit, five {sum(effect[h] for h in TOP_INDUCTION):.2f}, late heads " + ", ".join(f"{name(h)} {effect[h]:.2f}" for h in LATE_HEADS))


def main() -> None:
    shuffle = TokenShuffle()
    model = real_network(shuffle).double()
    model.set_attn_implementation("eager")
    rng = np.random.default_rng(SEED)
    ladder(model, shuffle, rng)
    names_test(model, shuffle, rng)


if __name__ == "__main__":
    main()
