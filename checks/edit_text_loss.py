"""Measures how much text loss Astra's two GPT-2 edits add on several sets of story windows, and where in a long window the rise falls. Makes no Astra calls."""
import numpy as np
import torch

from checks.edit_demo import ROOT, RUNS, TOP_FIVE, astra_edit
from checks.edit_limits import story_windows
from checks.other_abilities import fixed_heads, mean_outputs
from wsp.disguise import disguise, model_from
from wsp.network import TokenShuffle
from wsp.write_task import TEXT_FILE, TEXT_LENGTH, TEXT_SEQUENCES, apply_edit, text_tokens

OTHER_SEEDS = (1, 2, 3)
MANY_WINDOWS, MANY_SEED = 400, 11
MEAN_SEED = 5
BATCH = 8
READ_CHARACTERS = 1_000_000
# Predicted positions in a long window: the span of a short window, up to token 128, and past it.
SPANS = ((0, TEXT_LENGTH - 1), (TEXT_LENGTH - 1, 127), (127, 255))


def windows(tokens: list[int], count: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    return torch.tensor([tokens[int(i) : int(i) + TEXT_LENGTH] for i in rng.choice(len(tokens) - TEXT_LENGTH, count, replace=False)])


@torch.no_grad()
def token_losses(model, text: torch.Tensor) -> torch.Tensor:
    """Returns each predicted token's cross-entropy, one row per window."""
    rows = []
    for part in text.split(BATCH):
        logits = model(part).logits[:, :-1]
        rows.append(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), part[:, 1:].reshape(-1), reduction="none").reshape(part.size(0), -1).double())
    return torch.cat(rows)


def main() -> None:
    shuffle = TokenShuffle()
    with open(ROOT / TEXT_FILE, encoding="utf-8", errors="replace") as handle:
        tokens = shuffle.encode(handle.read(READ_CHARACTERS))
    sets = {f"scored, {TEXT_SEQUENCES} windows of {TEXT_LENGTH}": text_tokens(shuffle)}
    sets.update({f"other {TEXT_SEQUENCES} windows of {TEXT_LENGTH}, seed {seed}": windows(tokens, TEXT_SEQUENCES, seed) for seed in OTHER_SEEDS})
    sets[f"other {MANY_WINDOWS} windows of {TEXT_LENGTH}"] = windows(tokens, MANY_WINDOWS, MANY_SEED)
    long_label = "100 windows of 256"
    sets[long_label] = story_windows(shuffle)[1]
    disguised = disguise(RUNS["run 2"][1], shuffle)
    model = model_from(disguised)
    base = {label: token_losses(model, text) for label, text in sets.items()}
    slot = lambda layer, head: int(np.where(disguised.original_head[layer] == head)[0][0])
    hooks = fixed_heads(model, mean_outputs(model, windows(tokens, TEXT_SEQUENCES, MEAN_SEED)), [(layer, slot(layer, head)) for layer, head in TOP_FIVE])
    arms = {"top 5 fixed at their average output, which needs a forward pass": {label: token_losses(model, text) for label, text in sets.items()}}
    for hook in hooks:
        hook.remove()
    for run, (folder, seed) in RUNS.items():
        edited_from = disguise(seed, shuffle)
        edited = model_from(apply_edit(edited_from, astra_edit(edited_from, ROOT / "runs" / folder)))
        arms[f"Astra, {run}"] = {label: token_losses(edited, text) for label, text in sets.items()}
    for arm, losses in arms.items():
        print(f"\n[{arm}]")
        for label, loss in losses.items():
            rise = loss - base[label]
            print(f"{label}: unedited {base[label].mean():.4f}, rise {rise.mean():+.4f} = {100 * rise.mean() / base[label].mean():+.2f}%, standard error over windows {100 * rise.mean(1).std() / np.sqrt(rise.size(0)) / base[label].mean():.2f}%")
        for first, last in SPANS:
            before, rise = base[long_label][:, first:last], (losses[long_label] - base[long_label])[:, first:last]
            print(f"  {long_label}, predicted positions {first + 1} to {last}: unedited {before.mean():.3f}, rise {rise.mean():+.4f} = {100 * rise.mean() / before.mean():+.2f}%", flush=True)


if __name__ == "__main__":
    main()
