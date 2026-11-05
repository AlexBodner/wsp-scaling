"""Asks a reader to swap one GPT-2 ability for a specific other one, from scrambled weights: repeat the current word instead of continuing the repeat. `live` makes sealed Astra calls."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from checks.edit_demo import FACTS, RUNS, TOP_FIVE, astra_edit, top_share
from checks.other_abilities import build_abilities, right_share
from checks.swap_feasibility import DUPLICATE_HEADS, copy_search, word_lists
from scripts.run_reading_python import spent_so_far
from wsp.code_reader import _FORWARD_PASS_WORDS, run_code, run_code_reading, save_blocks
from wsp.disguise import HEADS, disguise, model_from
from wsp.head_kinds import REPEAT_LENGTH
from wsp.network import TokenShuffle, real_network
from wsp.reading import disguised_scores, head_scores
from wsp.write_task import EDITED_FILE, apply_edit, edit_code, repeat_tokens, text_tokens, touched_heads, write_prompt

ROOT = Path(__file__).parent.parent
MANIFEST = ROOT / "prompts" / "SWAP_MANIFEST.json"
SEEDS = {"1": 1601, "2": 1602}
MAX_COST = 30.0
NAMED = 5
TOP_TEN = 10
_SCRIPTED_REPLY = 'Zeroing one head.\n\n```edit\nW["L5.out.W"][0:64, :] = 0\n```\n\n```answer\n5.0\n5.1\n5.2\n5.3\n5.4\n```'
TASK = """YOUR TASK: change a language model's weights so that it swaps one ability for a specific other one and keeps the rest.

THE MODEL: its weights are those of a real trained language model. Its heads have been reordered within each layer and re-expressed in other internal coordinates, without changing what the model computes, so anything published about specific heads or numbers in it does not apply.

THE GOAL: when given random tokens followed by the same tokens again, the model now predicts the next token of the repeat. Change the weights so that in the second copy its top next token is instead the token it was just given, the current token, repeated. Everywhere else, including the first copy of the random tokens, its top next token should stay as it was, and its predictions on ordinary English text should change as little as possible.

HOW YOU ARE SCORED: we run the edited model. Swap score: how often its top next token equals the current token in the second copy of 50 random tokens repeated, higher is better. Echo score: how often its top next token equals the current token in the first copy and on stories, lower is better. Text score: how much its cross-entropy rises on ordinary English stories, lower is better. All count. It is never run for you, and you never see these numbers.

HOW YOU EDIT: in your final reply, before the answer block, give a code block whose opening line is three backticks followed by the word edit. It holds Python that changes the arrays in W in place, by any computation you like. Only the tensors in the blocks can change. We run it once, after you finish, in the same sandbox. Then, in the answer block, name the 5 heads your edit most depends on. If you are told a turn is your last, it still needs the edit block as well as the answer block.

"""


def swap_prompt() -> str:
    base = write_prompt()
    return TASK + base[base.index("THE NETWORK:") :]


@torch.no_grad()
def scores(model, shuffle: TokenShuffle) -> dict:
    """Returns the swap score, the old behaviour, text loss, list shares, other abilities and facts."""
    repeats = repeat_tokens(shuffle)
    echo = following = count = first_copy_echo = first_copy = 0
    for part in repeats.chunk(8):
        logits = model(part).logits
        top = logits[:, REPEAT_LENGTH:-1].argmax(-1)
        echo += int((top == part[:, REPEAT_LENGTH:-1]).sum())
        following += int((top == part[:, REPEAT_LENGTH + 1 :]).sum())
        count += top.numel()
        early = logits[:, :REPEAT_LENGTH].argmax(-1)
        first_copy_echo += int((early == part[:, :REPEAT_LENGTH]).sum())
        first_copy += early.numel()
    story_echo = story_tokens = 0
    for part in text_tokens(shuffle).chunk(8):
        top = model(part).logits.argmax(-1)
        story_echo += int((top == part).sum())
        story_tokens += top.numel()
    total, tokens = 0.0, 0
    for part in text_tokens(shuffle).chunk(8):
        logits = model(part).logits[:, :-1]
        total += float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), part[:, 1:].reshape(-1), reduction="sum"))
        tokens += part[:, 1:].numel()
    lists = word_lists(shuffle)
    batch = torch.tensor([item[0] for item in lists])
    tops = torch.cat([model(batch[i : i + 64]).logits[:, -1].argmax(-1) for i in range(0, len(batch), 64)]).tolist()
    facts = [(shuffle.encode(prompt), shuffle.encode(f" {answer}")[0]) for prompt, answer in FACTS]
    return {
        "swap_random_tokens": echo / count, "old_behaviour_random_tokens": following / count, "first_copy_echo": first_copy_echo / first_copy, "story_echo": story_echo / story_tokens, "text_loss": total / tokens,
        "lists_next_word": float(np.mean([t == item[1] for t, item in zip(tops, lists)])), "lists_current_word": float(np.mean([t == item[2] for t, item in zip(tops, lists)])),
        "abilities": {name: [round(value, 3) for value in right_share(model, items)] for name, items in build_abilities(shuffle).items()},
        "facts_right": sum(top_share(model, [fact]) == 1 for fact in facts),
    }


class CurrentTokenBias(torch.nn.Module):
    """Wraps a model and adds a fixed amount to the current token's logit, the cheapest way to echo, to test whether the scores catch it."""

    def __init__(self, model, bias: float) -> None:
        super().__init__()
        self.model, self.bias = model, bias

    def forward(self, input_ids=None, **kwargs):
        output = self.model(input_ids, **kwargs)
        output.logits.scatter_add_(2, input_ids[..., None], torch.full((*input_ids.shape, 1), self.bias, dtype=output.logits.dtype))
        return output


@torch.no_grad()
def blended_search(model, heads, sources, share: float) -> None:
    from checks.swap_feasibility import columns

    for (layer, head), (source_layer, source_head) in zip(heads, sources):
        to_q, to_k, _ = columns(model, layer, head)
        from_q, from_k, _ = columns(model, source_layer, source_head)
        attn, source = model.transformer.h[layer].attn.c_attn, model.transformer.h[source_layer].attn.c_attn
        for to, frm in ((to_q, from_q), (to_k, from_k)):
            attn.weight[:, to] = (1 - share) * attn.weight[:, to] + share * source.weight[:, frm]
            attn.bias[to] = (1 - share) * attn.bias[to] + share * source.bias[frm]


def baselines() -> None:
    shuffle = TokenShuffle()
    fresh = lambda: real_network(shuffle).double().eval()
    def report(label: str, model) -> None:
        result = scores(model, shuffle)
        print(label, "text rise %+.3f" % (result["text_loss"] - base["text_loss"]), json.dumps(result), flush=True)
    model = fresh()
    model.set_attn_implementation("eager")
    base = scores(model, shuffle)
    print("unedited", json.dumps(base), flush=True)
    for label, heads in (("top 2 search like duplicate heads", TOP_FIVE[:2]), ("top 3", TOP_FIVE[:3]), ("top 4", TOP_FIVE[:4]), ("top 5", TOP_FIVE)):
        model = fresh()
        model.set_attn_implementation("eager")
        copy_search(model, heads, DUPLICATE_HEADS)
        report(label, model)
    for dropped in range(len(TOP_FIVE)):
        model = fresh()
        model.set_attn_implementation("eager")
        heads = [head for index, head in enumerate(TOP_FIVE) if index != dropped]
        copy_search(model, heads, DUPLICATE_HEADS)
        report(f"4 heads, leaving out {TOP_FIVE[dropped]}", model)
    for share in (0.3, 0.4):
        model = fresh()
        model.set_attn_implementation("eager")
        blended_search(model, TOP_FIVE, DUPLICATE_HEADS, share)
        report(f"top 5, blend {share}", model)
    for bias in (2.0, 4.0, 6.0, 8.0):
        model = fresh()
        model.set_attn_implementation("eager")
        report(f"logit bias {bias} toward the current token, not a weight edit", CurrentTokenBias(model, bias))
    for run, (folder, seed) in RUNS.items():
        disguised = disguise(seed, shuffle)
        report(f"the earlier edit, {run}", model_from(apply_edit(disguised, astra_edit(disguised, ROOT / "runs" / folder))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "baselines", "live", "scripted"])
    parser.add_argument("--arm", choices=list(SEEDS), default="1")
    arguments = parser.parse_args()
    if arguments.mode == "baselines":
        baselines()
        return
    prompt = swap_prompt()
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if arguments.mode == "freeze":
        (ROOT / "prompts" / "swap.txt").write_text(prompt)
        MANIFEST.write_text(json.dumps({"swap": digest}))
        print("froze the prompt")
        return
    if digest != json.loads(MANIFEST.read_text())["swap"]:
        raise ValueError("the prompt no longer matches its frozen hash")
    seed = SEEDS[arguments.arm]
    shuffle = TokenShuffle()
    disguised = disguise(seed, shuffle)
    weights = ROOT / "data" / "write" / f"blocks_{seed}.npz"
    if not weights.exists():
        save_blocks(disguised, weights)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / f"swap-{arguments.arm}__{arguments.mode}__{stamp}"
    run_dir.mkdir(parents=True)
    if arguments.mode == "live":
        from wsp.astra import bring_calls_home, call_astra

        def call(text: str, item: str) -> str:
            return call_astra(text, run_dir, item)["reply"]

        spent = lambda: spent_so_far(run_dir)
    else:
        call, spent = (lambda text, item: _SCRIPTED_REPLY), (lambda: 0.0)
    real = real_network(shuffle).double()
    real.set_attn_implementation("eager")
    order = np.argsort(-disguised_scores(head_scores(real), disguised).ravel())
    del real
    top = {(int(i) // HEADS, int(i) % HEADS) for i in order[:NAMED]}
    top_ten = {(int(i) // HEADS, int(i) % HEADS) for i in order[:TOP_TEN]}
    result = run_code_reading(prompt, weights, top, top_ten, seed, call, spent, run_dir, MAX_COST)
    if arguments.mode == "live":
        bring_calls_home(run_dir)
    (run_dir / "disguise_map.json").write_text(json.dumps(disguised.original_head.tolist()))
    print(json.dumps(result.__dict__, default=list)[:300], flush=True)
    last_reply = [json.loads(line) for line in (run_dir / "transcript.jsonl").read_text().splitlines()][-1]["reply"]
    code = edit_code(last_reply)
    summary: dict = {"seed": seed, "cost": spent(), "named_heads": [list(pick) for pick in result.answer or []]}
    if code is None:
        summary["counted_as"] = "no edit block"
    else:
        (run_dir / "edit.py").write_text(code)
        summary["edit_forward_pass_flags"] = sorted(set(_FORWARD_PASS_WORDS.findall(code)))
        edited_path = run_dir / EDITED_FILE
        summary["edit_output"] = run_code(code, weights, keep=(EDITED_FILE, edited_path))[-600:]
        if edited_path.exists():
            with np.load(edited_path) as archive:
                edited = {name: archive[name] for name in archive.files}
            summary["before"] = scores(model_from(disguised), shuffle)
            changed = apply_edit(disguised, edited)
            summary["after"] = scores(model_from(changed), shuffle)
            summary["values_changed"] = sum(int((edited[name] != disguised.tensors[name].astype(np.float32)).sum()) for name in edited)
            summary["touched_heads"] = touched_heads(disguised, edited)
            summary["tensors_changed"] = sorted(edited)[:40]
            edited_path.unlink()
    (run_dir / "swap_score.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:1800])
    weights.unlink()


if __name__ == "__main__":
    main()
