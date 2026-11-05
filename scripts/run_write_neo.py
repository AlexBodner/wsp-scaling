"""Asks a reader to edit a scrambled GPT-Neo 125M, trained separately from GPT-2, so it stops repeating. `live` and `blind` make sealed Astra calls.

GPT-Neo's top induction heads sit in layers 6 and 8, so remembering GPT-2 points to the wrong heads. `blind` asks with no weights, to show what memory alone gives.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from checks.edit_demo import head_list_edit
from checks.other_models import cosines, statistic_picks
from scripts.run_reading_python import spent_so_far
from scripts.run_write_medium import _SCRIPTED_REPLY, LAST_TURN, scores, zeroed
from wsp.code_reader import _FORWARD_PASS_WORDS, run_code, run_code_reading, save_blocks
from wsp.family import disguise_family, family_prompt, induction_scores, load
from wsp.network import writable_tensors
from wsp.reading import blind_prompt
from wsp.write_task import EDITED_FILE, TASK, edit_code

import torch

ROOT = Path(__file__).parent.parent
MANIFEST = ROOT / "prompts" / "WRITE_NEO_MANIFEST.json"
ARMS = {"1": 1501, "2": 1502}
BASELINE_SEED = 1599
BLIND_SEED = 1598
MAX_COST = 6.0
NAMED = 5
TOP_TEN = 10
HEADS = 12
RANDOM_SETS = 8
SCALE_LINE = ("softmax over i <= j of (q_j . k_i) / 8", "softmax over i <= j of (q_j . k_i)")
POSITION_LINE = ("P: 1024 x 768", "P: 2048 x 768")


def prompt_for(model) -> str:
    base = family_prompt(model)
    for old, new in (SCALE_LINE, POSITION_LINE):
        if old not in base:
            raise ValueError(f"expected to replace {old!r}")
        base = base.replace(old, new)
    return TASK + base[base.index("THE NETWORK:") :]


def blind_for(model) -> str:
    prompt = blind_prompt()
    weights = f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    if "124,439,808" not in prompt:
        raise ValueError("the blind prompt no longer states the weight count")
    prompt = prompt.replace("124,439,808", weights)
    for old, new in (SCALE_LINE, POSITION_LINE):
        if old not in prompt:
            raise ValueError(f"expected to replace {old!r}")
        prompt = prompt.replace(old, new)
    return prompt


def fresh(tensors: dict[str, np.ndarray]):
    """Returns a runnable float32 GPT-Neo holding these weights."""
    model = load("neo").double().float()
    target = writable_tensors(model)
    with torch.no_grad():
        for name, values in tensors.items():
            target[name].copy_(torch.from_numpy(values.astype(np.float32)))
    return model


def truth_lists(model, order_map: np.ndarray) -> tuple[list[tuple[int, int]], set, set]:
    """Returns the real top heads, and the top 5 and top 10 in the scramble's numbering."""
    truth = induction_scores(model.double().float())
    real = [(int(i) // HEADS, int(i) % HEADS) for i in np.argsort(-truth.ravel())[:TOP_TEN]]
    scrambled = np.stack([truth[layer, order_map[layer]] for layer in range(model.config.n_layer)])
    order = np.argsort(-scrambled.ravel())
    return real, {(int(i) // HEADS, int(i) % HEADS) for i in order[:NAMED]}, {(int(i) // HEADS, int(i) % HEADS) for i in order[:TOP_TEN]}


def baselines() -> None:
    model = load("neo")
    truth = induction_scores(model.double().float())
    order = np.argsort(-truth.ravel())
    top = [(int(i) // HEADS, int(i) % HEADS) for i in order[:NAMED]]
    following = [(int(i) // HEADS, int(i) % HEADS) for i in order[NAMED : 2 * NAMED]]
    print("top heads: " + ", ".join(f"{i // HEADS}.{i % HEADS} {truth.ravel()[i]:.3f}" for i in order[:12]), flush=True)
    layers = np.repeat(np.arange(12), HEADS)
    picks = statistic_picks(cosines(load("neo")), layers[:, None] > layers[None, :])
    name = lambda index: f"{index // HEADS}.{index % HEADS}"
    print(f"script's statistic: source {name(picks['source'])}, best pairing {[name(i) for i in picks['best_pairing']]}, hits {len(set(picks['best_pairing']) & set(order[:NAMED].tolist()))} of {NAMED}", flush=True)
    disguised = disguise_family(load("neo"), BASELINE_SEED)
    base = scores(fresh(disguised.tensors))
    print(f"unedited: repeat accuracy {base[0]:.3f}, text loss {base[1]:.3f}", flush=True)
    scrambled = lambda real: [(layer, int(np.where(disguised.original_head[layer] == head)[0][0])) for layer, head in real]
    report = lambda label, tensors: (lambda result: print(f"{label}: repeat accuracy {result[0]:.3f}, rise {result[1] - base[1]:+.3f}", flush=True))(scores(fresh(tensors)))
    simple = lambda real: {**disguised.tensors, **head_list_edit(disguised, scrambled(real))}
    report("top 5 zeroed", zeroed(disguised, top))
    report("next 5 zeroed", zeroed(disguised, following))
    report("simple edit, top 5", simple(top))
    report("simple edit, next 5", simple(following))
    report("all of layers 6 to 8 zeroed", zeroed(disguised, [(layer, head) for layer in (6, 7, 8) for head in range(HEADS)]))
    rng = np.random.default_rng(BASELINE_SEED)
    others = [(layer, head) for layer in range(12) for head in range(HEADS) if (layer, head) not in top + following]
    for number in range(RANDOM_SETS):
        report(f"simple edit, random 5, set {number}", simple([others[int(i)] for i in rng.choice(len(others), NAMED, replace=False)]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "baselines", "blind", "live", "scripted"])
    parser.add_argument("--arm", choices=list(ARMS), default="1")
    arguments = parser.parse_args()
    if arguments.mode == "baselines":
        baselines()
        return
    model = load("neo")
    prompt, blind = prompt_for(model), blind_for(model)
    hashes = {"write_neo": hashlib.sha256(prompt.encode()).hexdigest(), "blind_neo": hashlib.sha256(blind.encode()).hexdigest()}
    if arguments.mode == "freeze":
        (ROOT / "prompts" / "write_neo.txt").write_text(prompt)
        (ROOT / "prompts" / "blind_neo.txt").write_text(blind)
        MANIFEST.write_text(json.dumps(hashes))
        print("froze the prompts")
        return
    if hashes != json.loads(MANIFEST.read_text()):
        raise ValueError("the prompts no longer match their frozen hashes")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if arguments.mode == "blind":
        from wsp.astra import bring_calls_home, call_astra

        run_dir = ROOT / "runs" / f"neo-blind__live__{stamp}"
        run_dir.mkdir(parents=True)
        reply = call_astra(blind, run_dir, "blind")["reply"]
        (run_dir / "reply.md").write_text(reply)
        print(reply[-400:], f"\nlist price ${spent_so_far(run_dir):.2f}")
        bring_calls_home(run_dir)
        return
    seed = ARMS[arguments.arm]
    disguised = disguise_family(model, seed)
    real, top, top_ten = truth_lists(load("neo"), disguised.original_head)
    print("seed's scores before any call:", scores(fresh(disguised.tensors)), "real top", real, flush=True)
    del model
    weights = ROOT / "data" / "family" / f"blocks_{seed}.npz"
    if not weights.exists():
        save_blocks(disguised, weights)
    run_dir = ROOT / "runs" / f"write-neo-{arguments.arm}__{arguments.mode}__{stamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "disguise_map.json").write_text(json.dumps(disguised.original_head.tolist()))
    if arguments.mode == "live":
        from wsp.astra import bring_calls_home, call_astra

        def call(text: str, item: str) -> str:
            return call_astra(text, run_dir, item)["reply"]

        spent = lambda: spent_so_far(run_dir)
    else:
        call, spent = (lambda text, item: _SCRIPTED_REPLY), (lambda: 0.0)
    result = run_code_reading(prompt, weights, top, top_ten, seed, call, spent, run_dir, MAX_COST, LAST_TURN)
    print(json.dumps(result.__dict__, default=list)[:300], flush=True)
    last_reply = [json.loads(line) for line in (run_dir / "transcript.jsonl").read_text().splitlines()][-1]["reply"]
    code = edit_code(last_reply)
    summary: dict = {"seed": seed, "cost": spent(), "named_heads": [list(pick) for pick in result.answer or []], "real_top": real}
    if code is None:
        summary["counted_as"] = "no edit block"
    else:
        (run_dir / "edit.py").write_text(code)
        summary["edit_forward_pass_flags"] = sorted(set(_FORWARD_PASS_WORDS.findall(code)))
        edited_path = run_dir / EDITED_FILE
        summary["edit_output"] = run_code(code, weights, keep=(EDITED_FILE, edited_path))[-600:]
        if edited_path.exists():
            tensors = dict(disguised.tensors)
            with np.load(edited_path) as archive:
                for name in archive.files:
                    if name not in tensors or name in ("E", "P", "F.scale", "F.shift") or tensors[name].shape != archive[name].shape:
                        raise ValueError(f"the edit may not change {name}")
                    tensors[name] = archive[name]
                summary["tensors_changed"] = sorted(archive.files)[:60]
                summary["values_changed"] = int(sum((archive[name] != disguised.tensors[name].astype(np.float32)).sum() for name in archive.files))
            summary["before"] = scores(fresh(disguised.tensors))
            summary["after"] = scores(fresh(tensors))
            edited_path.unlink()
    (run_dir / "write_score.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:1500])
    weights.unlink()
    if arguments.mode == "live":
        bring_calls_home(run_dir)


if __name__ == "__main__":
    main()
