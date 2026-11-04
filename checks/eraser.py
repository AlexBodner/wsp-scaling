"""Tests whether negative value-output weights mark GPT-2 heads that push down what they attend to. Makes no Astra calls.

Each head's statistic is its negative trace(VO) share: the sum of the magnitudes of the negative real parts of VO's
eigenvalues, over the sum of the magnitudes of all real parts. VO has the same nonzero eigenvalues as the smaller OV, with
the layer norm's scale and mean removal included. Minus trace(VO) over VO's norm, as in head_types.py, is printed as a
second reading. The behaviour is each head's direct effect on the logit of the token it attends to most at each position,
measured as in signed_ladder.py, on the second copy of repeated random tokens and on stories from data/tinystories/valid.txt.
Controls are a within-layer shuffle of the statistic and, for each top head, the most positive unused head from its layer.
"""
import gc
import resource
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from checks.head_types import name
from checks.other_models import check_slices, head_parts
from checks.signed_ladder import direct_effect, run_with_heads
from wsp.disguise import HEADS, LAYERS
from wsp.network import TokenShuffle, real_network

SEED = 20261104
REPEAT_LENGTH = 25
BATCH = 32
CHUNK = 4
STORIES = 128
STORY_TOKENS = 64
TEXT = Path("data/tinystories/valid.txt")
SEPARATOR = "<|endoftext|>"
TOP = 10
TOTAL = LAYERS * HEADS
DRAWS = 2000
NAMED = (10 * HEADS + 7, 11 * HEADS + 10)
PREDICTED_SPEARMAN = 0.3
SLICE_TOLERANCE = 1e-4
# Direct effects on the right next token on repeats, from checks/outputs/signed_ladder.txt.
LADDER = {10 * HEADS + 7: -1.24, 11 * HEADS + 10: -0.72}
LADDER_TOLERANCE = 0.005


def negative_statistics(model) -> dict[str, np.ndarray]:
    """Returns each head's negative trace(VO) share, its trace(VO) over the norm, and how far the eigenvalues miss the trace."""
    share, trace_share, worst = np.zeros(TOTAL), np.zeros(TOTAL), 0.0
    for index, (_, _, v, o) in enumerate(head_parts(model)):
        real = np.linalg.eigvals(o @ v).real
        trace = np.einsum("ij,ji->", v, o)
        norm = np.sqrt(((v.T @ v) * (o @ o.T)).sum())
        worst = max(worst, abs(real.sum() - trace))
        share[index] = -real[real < 0].sum() / np.abs(real).sum()
        trace_share[index] = trace / norm
    return {"share": share, "trace over norm": trace_share, "worst": np.array(worst)}


@torch.no_grad()
def attended_effect(model, tokens: torch.Tensor, attentions: tuple, per_head: dict, final_input: torch.Tensor, positions: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns each head's direct effect on the logit of the token it attends to most, summed over batch and positions.

    Also returns the summed largest attention and how many of those positions are the head attending to itself.
    """
    batch = len(tokens)
    scale = model.transformer.ln_f.weight
    unembed = model.lm_head.weight
    centre = unembed.mean(0)
    deviation = final_input[:, positions].std(-1, keepdim=True)
    rows = torch.arange(batch)[:, None, None]
    effect, largest, itself = np.zeros((LAYERS, HEADS)), np.zeros((LAYERS, HEADS)), np.zeros((LAYERS, HEADS))
    for layer, vectors in per_head.items():
        pattern = attentions[layer][:, :, positions]
        peak, picked = pattern.max(-1)
        direction = unembed[tokens[rows, picked]] - centre
        chosen = vectors[:, :, positions] / deviation[:, None] * scale
        effect[layer] = (chosen * direction).sum(-1).sum((0, 2)).numpy()
        largest[layer] = peak.sum((0, 2)).numpy()
        itself[layer] = (picked == positions[None, None]).sum((0, 2)).numpy()
    return effect, largest, itself


def repeat_tokens(vocabulary: int) -> torch.Tensor:
    """Returns the repeated random tokens signed_ladder.py uses."""
    generator = torch.Generator().manual_seed(SEED)
    half = torch.randint(0, vocabulary, (BATCH, REPEAT_LENGTH), generator=generator)
    return torch.cat([half, half], 1)


def story_tokens(shuffle: TokenShuffle) -> torch.Tensor:
    """Returns the opening tokens of random whole stories, leaving out the fragment the file starts with."""
    stories = TEXT.read_text().split(SEPARATOR)[1:]
    chosen = []
    for index in np.random.default_rng(SEED).permutation(len(stories)):
        tokens = shuffle.encode(stories[index].strip())
        if len(tokens) >= STORY_TOKENS:
            chosen.append(tokens[:STORY_TOKENS])
        if len(chosen) == STORIES:
            return torch.tensor(chosen)
    raise ValueError("too few long stories")


def measured(model, tokens: torch.Tensor, positions: torch.Tensor, ladder_check: bool = False) -> dict[str, np.ndarray]:
    """Returns the per-head direct effect on the attended token at the given positions, in chunks, and the ladder's effect."""
    effect, largest, itself, next_token = (np.zeros((LAYERS, HEADS)) for _ in range(4))
    for chunk in tokens.split(CHUNK):
        output, per_head, final_input = run_with_heads(model, chunk)
        parts = attended_effect(model, chunk, output.attentions, per_head, final_input, positions)
        effect, largest, itself = effect + parts[0], largest + parts[1], itself + parts[2]
        if ladder_check:
            for t in positions.tolist():
                next_token += len(chunk) * direct_effect(model, per_head, final_input, torch.full((len(chunk),), t), chunk[:, t + 1])
        del output, per_head, final_input
    count = len(tokens) * len(positions)
    return {"effect": effect.ravel() / count, "largest": largest.ravel() / count, "itself": itself.ravel() / count, "next": next_token.ravel() / count}


def matched_positive(share: np.ndarray, trace: np.ndarray, top: np.ndarray) -> list[int | None]:
    """For each top head, returns the most positive unused head of its layer with a positive trace, outside the top set."""
    used, matches = set(top.tolist()), []
    for head in top:
        layer = int(head) // HEADS
        candidates = [i for i in range(layer * HEADS, (layer + 1) * HEADS) if i not in used and trace[i] > 0]
        pick = min(candidates, key=lambda i: share[i]) if candidates else None
        matches.append(pick)
        if pick is not None:
            used.add(pick)
    return matches


def within_layer_chance(statistic: np.ndarray, effects: dict[str, np.ndarray], rng: np.random.Generator) -> dict:
    """Shuffles the statistic within layers, and returns each draw's Spearman and top mean effect, and whether the named heads stay on top."""
    ranks = rankdata(statistic)
    effect_ranks = {label: rankdata(values) for label, values in effects.items()}
    spearman = {label: np.zeros(DRAWS) for label in effects}
    top_mean = {label: np.zeros(DRAWS) for label in effects}
    named = np.zeros(DRAWS, dtype=bool)
    for draw in range(DRAWS):
        permutation = np.concatenate([layer * HEADS + rng.permutation(HEADS) for layer in range(LAYERS)])
        shuffled = statistic[permutation]
        top = np.argsort(-shuffled)[:TOP]
        named[draw] = all(head in top for head in NAMED)
        for label, values in effects.items():
            spearman[label][draw] = np.corrcoef(ranks[permutation], effect_ranks[label])[0, 1]
            top_mean[label][draw] = values[top].mean()
    return {"spearman": spearman, "top_mean": top_mean, "named": named}


def main() -> None:
    shuffle = TokenShuffle()
    model = real_network(shuffle).double()
    model.set_attn_implementation("eager")
    gc.collect()
    worst = check_slices(model)
    print(f"slices rebuild the model's attention and output in float64, worst error {worst:.2e}")
    if not worst < SLICE_TOLERANCE:
        raise ValueError(f"slices do not match the model, error {worst}")

    statistics = negative_statistics(model)
    share, trace = statistics["share"], statistics["trace over norm"]
    print(f"eigenvalues of OV sum to trace(VO), worst miss {float(statistics['worst']):.2e}")

    repeats = measured(model, repeat_tokens(len(shuffle.tokenizer)), torch.arange(REPEAT_LENGTH, 2 * REPEAT_LENGTH - 1), ladder_check=True)
    print("next-token effect on repeats reproduces signed_ladder.txt: " + ", ".join(f"{name(head)} {repeats['next'][head]:.2f} (was {LADDER[head]:.2f})" for head in LADDER))
    for head, expected in LADDER.items():
        if abs(repeats["next"][head] - expected) > LADDER_TOLERANCE:
            raise ValueError(f"{name(head)}: {repeats['next'][head]} does not reproduce {expected}")
    text = measured(model, story_tokens(shuffle), torch.arange(1, STORY_TOKENS))
    effects = {"repeats": repeats["effect"], "stories": text["effect"]}
    print(f"sample: repeats {BATCH} sequences, {REPEAT_LENGTH - 1} positions in the second copy; stories {STORIES} sequences, {STORY_TOKENS - 1} positions")

    order = np.argsort(-share)
    top = order[:TOP]
    rank = np.argsort(order) + 1
    print(f"\nTop {TOP} of {TOTAL} by negative trace(VO) share, with share, trace(VO) over norm, and on repeats / stories: direct effect on the attended token, mean largest attention, share of positions on itself")
    for head in top:
        print(f"  {name(int(head))}: share {share[head]:.3f}, trace {trace[head]:.3f}, effect {repeats['effect'][head]:.3f}/{text['effect'][head]:.3f}, "
              f"attention {repeats['largest'][head]:.2f}/{text['largest'][head]:.2f}, on itself {repeats['itself'][head]:.2f}/{text['itself'][head]:.2f}")
    named_top = all(head in top for head in NAMED)
    print("named heads: " + ", ".join(f"{name(head)} rank {rank[head]}, share {share[head]:.3f}, effect {repeats['effect'][head]:.3f}/{text['effect'][head]:.3f}" for head in NAMED))
    by_trace = np.argsort(trace)
    print("second reading, lowest trace(VO) over norm: " + ", ".join(f"{name(int(i))} {trace[i]:.3f}" for i in by_trace[:TOP]) +
          "; named heads rank " + ", ".join(f"{name(head)} {int(np.flatnonzero(by_trace == head)[0]) + 1}" for head in NAMED))

    print("\nMost negative direct effects on the attended token")
    for label, values in effects.items():
        lowest = np.argsort(values)[:TOP]
        print(f"  {label}: " + ", ".join(f"{name(int(i))} {values[i]:.3f} (share rank {rank[i]})" for i in lowest))

    rng = np.random.default_rng(SEED)
    chance = within_layer_chance(share, effects, rng)
    print(f"\nSpearman between negative share and direct effect on all {TOTAL} heads, with {DRAWS} within-layer shuffles")
    spearmans = {}
    for label, values in effects.items():
        observed = float(np.corrcoef(rankdata(share), rankdata(values))[0, 1])
        spearmans[label] = observed
        drawn = chance["spearman"][label]
        trace_rho = float(np.corrcoef(rankdata(-trace), rankdata(values))[0, 1])
        print(f"  {label}: {observed:.3f}, shuffles mean {drawn.mean():.3f}, 2.5th to 97.5th percentile {np.percentile(drawn, 2.5):.3f} to {np.percentile(drawn, 97.5):.3f}, "
              f"chance of this magnitude or more {np.mean(np.abs(drawn) >= abs(observed)):.4f}; second reading {trace_rho:.3f}")

    matches = matched_positive(share, trace, top)
    pairs = [(int(head), match) for head, match in zip(top, matches)]
    print("\nMatched same-layer positive heads, top head to match, share and effect on repeats / stories")
    for head, match in pairs:
        shown = "no match" if match is None else f"{name(match)} share {share[match]:.3f}, effect {repeats['effect'][match]:.3f}/{text['effect'][match]:.3f}"
        print(f"  {name(head)} to {shown}")
    kept = [match for _, match in pairs if match is not None]
    means = {}
    print(f"\nMean direct effect on the attended token, top {TOP}, matched ({len(kept)} heads), and within-layer shuffles of the top {TOP}")
    for label, values in effects.items():
        means[label] = (float(values[top].mean()), float(values[kept].mean()))
        drawn = chance["top_mean"][label]
        print(f"  {label}: top {means[label][0]:.3f}, matched {means[label][1]:.3f}, all heads {values.mean():.3f}, shuffles mean {drawn.mean():.3f}, "
              f"chance of this low or lower {np.mean(drawn <= means[label][0]):.4f}")
    print(f"  chance both named heads are in the top {TOP} under the shuffle: {chance['named'].mean():.4f}")

    print("\nprediction: " + "; ".join((
        f"10.7 and 11.10 in the top {TOP}: {named_top}",
        f"Spearman magnitude at least {PREDICTED_SPEARMAN} on repeats and stories: {all(abs(value) >= PREDICTED_SPEARMAN for value in spearmans.values())}",
        f"Spearman negative on both, the mechanism's sign: {all(value < 0 for value in spearmans.values())}",
        f"top {TOP} mean below 0 on both: {all(top_value < 0 for top_value, _ in means.values())}",
        f"matched mean not below 0 on both: {all(matched >= 0 for _, matched in means.values())}",
    )))
    print(f"peak resident memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
