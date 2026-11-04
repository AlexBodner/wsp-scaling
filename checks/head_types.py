"""Measures which kinds of head GPT-2 has, by running it, and how far weight-only statistics find them. Makes no Astra calls.

Each kind is defined by where a head attends, on repeated random tokens or on indirect-object sentences, or by what
its output does to the answer. For each kind this prints the top heads and the gap below them, to show which kinds
have a clean top set a reader could be scored on, then how well simple statistics on the blocks find that set.
"""
import numpy as np
import torch

from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH
from wsp.network import TokenShuffle, real_network, writable_tensors

REPEAT_LENGTH = 25
BATCH = 32
SEED = 20261104
TOP = 5
SHOWN = 8
NAMES = "Mary John Anna Tom Lisa Mark Sara Paul Kate Peter Emma David Laura James Alice Henry".split()
PLACES = "store park school office beach".split()
OBJECTS = "drink book ball key letter".split()
IOI_PROMPTS = 200


def name(index: int) -> str:
    return f"{index // HEADS}.{index % HEADS}"


@torch.no_grad()
def random_token_patterns(model, vocabulary: int) -> dict[str, np.ndarray]:
    """Returns each head's attention pattern on random tokens repeated once."""
    generator = torch.Generator().manual_seed(SEED)
    half = torch.randint(0, vocabulary, (BATCH, REPEAT_LENGTH), generator=generator)
    attentions = model(torch.cat([half, half], 1), output_attentions=True).attentions
    kinds = {kind: np.zeros((LAYERS, HEADS)) for kind in ("induction", "duplicate", "previous", "self", "first position")}
    second = range(REPEAT_LENGTH, 2 * REPEAT_LENGTH - 1)
    for layer, attention in enumerate(attentions):
        mean = attention.mean(0).numpy()
        for t in second:
            kinds["induction"][layer] += mean[:, t, t - REPEAT_LENGTH + 1] / len(second)
            kinds["duplicate"][layer] += mean[:, t, t - REPEAT_LENGTH] / len(second)
        first = range(1, REPEAT_LENGTH)
        kinds["previous"][layer] = np.mean([mean[:, t, t - 1] for t in first], axis=0)
        kinds["self"][layer] = np.mean([mean[:, t, t] for t in first], axis=0)
        kinds["first position"][layer] = np.mean([mean[:, t, 0] for t in first], axis=0)
    return {kind: values.ravel() for kind, values in kinds.items()}


def ioi_prompts(shuffle: TokenShuffle) -> list[tuple[list[int], int, int, int, int, int]]:
    """Builds same-length sentences, with where each name sits, its second subject, and the answer token."""
    rng = np.random.default_rng(SEED)
    prompts = []
    while len(prompts) < IOI_PROMPTS:
        receiver, subject = [str(NAMES[i]) for i in rng.choice(len(NAMES), 2, replace=False)]
        place, thing = PLACES[rng.integers(len(PLACES))], OBJECTS[rng.integers(len(OBJECTS))]
        text = f"When {receiver} and {subject} went to the {place}, {subject} gave a {thing} to"
        tokens = shuffle.encode(text)
        receiver_token, subject_token = shuffle.encode(f" {receiver}")[0], shuffle.encode(f" {subject}")[0]
        positions = [i for i, token in enumerate(tokens) if token == subject_token]
        if receiver_token in tokens and len(positions) == 2:
            prompts.append((tokens, tokens.index(receiver_token), positions[0], positions[1], receiver_token, subject_token))
    return prompts


@torch.no_grad()
def ioi_patterns(model, shuffle: TokenShuffle) -> tuple[dict[str, np.ndarray], float]:
    """Returns attention at the final token to each name, and at the second subject to the first, on indirect-object sentences."""
    prompts = ioi_prompts(shuffle)
    length = len(prompts[0][0])
    prompts = [prompt for prompt in prompts if len(prompt[0]) == length]
    tokens = torch.tensor([prompt[0] for prompt in prompts])
    output = model(tokens, output_attentions=True)
    rows = torch.arange(len(prompts))
    receiver_at, first_subject, second_subject = (torch.tensor([prompt[i] for prompt in prompts]) for i in (1, 2, 3))
    kinds = {kind: np.zeros((LAYERS, HEADS)) for kind in ("end to receiver (name mover)", "end to second subject (S-inhibition)", "second subject to first (duplicate in text)")}
    for layer, attention in enumerate(output.attentions):
        kinds["end to receiver (name mover)"][layer] = attention[rows, :, -1, receiver_at].mean(0).numpy()
        kinds["end to second subject (S-inhibition)"][layer] = attention[rows, :, -1, second_subject].mean(0).numpy()
        kinds["second subject to first (duplicate in text)"][layer] = attention[rows, :, second_subject, first_subject].mean(0).numpy()
    logits = output.logits[:, -1]
    receiver_tokens, subject_tokens = torch.tensor([prompt[4] for prompt in prompts]), torch.tensor([prompt[5] for prompt in prompts])
    accuracy = float((logits[rows, receiver_tokens] > logits[rows, subject_tokens]).double().mean())
    return {kind: values.ravel() for kind, values in kinds.items()}, accuracy


def weight_statistics(tensors: dict) -> dict[str, np.ndarray]:
    """Returns simple per-head statistics from the blocks alone, the tensors a reader gets."""
    query_key, value_output = [], []
    for layer in range(LAYERS):
        scale = tensors[f"L{layer}.norm1.scale"][:, None]
        weight, output = scale * tensors[f"L{layer}.qkv.W"], tensors[f"L{layer}.out.W"]
        for head in range(HEADS):
            start = head * HEAD_SIZE
            q, k, v = (weight[:, offset + start : offset + start + HEAD_SIZE] for offset in (0, WIDTH, 2 * WIDTH))
            query_key.append(q @ k.T)
            value_output.append(v @ output[start : start + HEAD_SIZE])
    a, b = np.stack([m.ravel() for m in query_key]), np.stack([m.ravel() for m in value_output])
    cosine = a @ b.T / np.outer(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
    earlier = np.arange(144)[:, None] // HEADS > np.arange(144)[None, :] // HEADS
    later = np.arange(144)[:, None] // HEADS < np.arange(144)[None, :] // HEADS
    return {
        "trace(QK^T), own match": np.array([np.trace(m) / np.linalg.norm(m) for m in query_key]),
        "trace(VO), own copying": np.array([np.trace(m) / np.linalg.norm(m) for m in value_output]),
        "best cosine as reader of an earlier head": np.where(earlier, cosine, -np.inf).max(1),
        "best cosine as writer for a later head": np.where(later, cosine, -np.inf).max(0),
        "minus trace(VO), own negative copying": np.array([-np.trace(m) / np.linalg.norm(m) for m in value_output]),
    }


def main() -> None:
    shuffle = TokenShuffle()
    model = real_network(shuffle).double()
    model.set_attn_implementation("eager")
    kinds = random_token_patterns(model, len(shuffle.tokenizer))
    sentences, accuracy = ioi_patterns(model, shuffle)
    kinds.update(sentences)
    print(f"indirect-object sentences: right name beats the subject {accuracy:.3f}")

    tensors = {key: value.detach().numpy().astype(np.float64) for key, value in writable_tensors(model).items()}
    statistics = weight_statistics(tensors)
    print("\nKinds of head, top heads by the measured score, and the gap below the top 5")
    tops = {}
    for kind, values in kinds.items():
        order = np.argsort(-values)
        tops[kind] = set(order[:TOP].tolist())
        print(f"  {kind}: " + ", ".join(f"{name(int(i))} {values[i]:.2f}" for i in order[:SHOWN]) + f" | 5th/6th {values[order[TOP - 1]]:.2f}/{values[order[TOP]]:.2f}")

    print("\nHow many of each kind's top 5 a statistic's top 5 finds, and how far it beats the kind's own overlap by chance (0.17)")
    header = list(statistics)
    for kind in kinds:
        cells = [f"{len(set(np.argsort(-statistics[label])[:TOP].tolist()) & tops[kind])}" for label in header]
        print(f"  {kind}: " + ", ".join(f"{label} {cell}" for label, cell in zip(header, cells)))

    print("\nOverlap between kinds' top 5, to see whether they are really different heads")
    labels = list(tops)
    for i, first in enumerate(labels):
        shared = [f"{second} {len(tops[first] & tops[second])}" for second in labels[i + 1 :] if tops[first] & tops[second]]
        if shared:
            print(f"  {first}: " + ", ".join(shared))


if __name__ == "__main__":
    main()
