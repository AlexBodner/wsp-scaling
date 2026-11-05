"""Measures what else Astra's edits changed: five other abilities on the edited models, against the same abilities when random heads are hit. Makes no Astra calls."""
import numpy as np
from transformers import AutoTokenizer, GPT2LMHeadModel

from checks.edit_demo import ROOT, RUNS, astra_edit, head_list_edit
from checks.neo_edit_check import PlainTokens
from checks.edit_demo import TOP_FIVE
from checks.other_abilities import build_abilities, fixed_heads, mean_outputs, right_share, stories
from scripts.run_write_neo import fresh
from wsp.disguise import disguise, model_from
from wsp.family import disguise_family, load
from wsp.network import TokenShuffle
from wsp.write_task import apply_edit

# The Neo runs this table describes. Named here so that later runs in runs/ don't change it.
NEO_RUNS = {"run 1": ("write-neo-1__live__20261104T195206Z", 1501), "run 2": ("write-neo-2__live__20261104T201804Z", 1502)}
RANDOM_SETS = 4
HEADS = 12


def report(label: str, model, abilities: dict) -> None:
    print(f"{label}: " + " | ".join(f"{name} {right_share(model, items)[0]:.3f} (margin {right_share(model, items)[1]:.2f})" for name, items in abilities.items()), flush=True)


def main() -> None:
    shuffle = TokenShuffle()
    abilities = build_abilities(shuffle)
    print("GPT-2 small: share of 200 prompts, drawn with repeats, where a right token beats the listed wrong ones, and the mean margin")
    report("unedited", model_from(disguise(RUNS["run 2"][1], shuffle)), abilities)
    rng = np.random.default_rng(0)
    for run, (folder, seed) in RUNS.items():
        disguised = disguise(seed, shuffle)
        report(f"Astra, {run}", model_from(apply_edit(disguised, astra_edit(disguised, ROOT / "runs" / folder))), abilities)
    disguised = disguise(RUNS["run 2"][1], shuffle)
    for number in range(RANDOM_SETS):
        heads = [(int(i) // HEADS, int(i) % HEADS) for i in rng.choice(HEADS * HEADS, 5, replace=False)]
        report(f"simple edit, random 5, set {number}", model_from(apply_edit(disguised, head_list_edit(disguised, heads))), abilities)
    tokens = PlainTokens()
    abilities = build_abilities(tokens)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    real = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").eval()
    report("unedited, real weights", real, build_abilities(tokenizer))
    hooks = fixed_heads(real, mean_outputs(real, stories(tokenizer)[1]), list(TOP_FIVE))
    report("top 5 fixed at their mean, real weights", real, build_abilities(tokenizer))
    for hook in hooks:
        hook.remove()
    print("\nGPT-Neo 125M")
    report("unedited", fresh(disguise_family(load("neo"), 1501).tensors), abilities)
    for run, (folder, seed) in NEO_RUNS.items():
        disguised = disguise_family(load("neo"), seed)
        report(f"Astra, {run}", fresh({**disguised.tensors, **astra_edit(disguised, ROOT / "runs" / folder)}), abilities)


if __name__ == "__main__":
    main()
