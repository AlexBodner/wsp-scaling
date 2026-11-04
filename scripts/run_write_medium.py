"""Asks a reader with Python to edit a scrambled GPT-2 medium so it stops repeating. `--live` makes sealed Astra calls, and `baselines` makes none."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from scripts.run_reading_python import spent_so_far
from wsp.code_reader import _FORWARD_PASS_WORDS, run_code, run_code_reading, save_blocks
from wsp.family import FamilyDisguise, disguise_family, family_prompt, induction_scores, load
from wsp.network import writable_tensors
from wsp.write_task import EDITED_FILE, SEED, TASK, TEXT_FILE, TEXT_LENGTH, TEXT_SEQUENCES, edit_code

ROOT = Path(__file__).parent.parent
RUN_SEED = 1401
BASELINE_SEED = 1499
MANIFEST = ROOT / "prompts" / "WRITE_MEDIUM_MANIFEST.json"
MAX_COST = 4.8
LAST_TURN = "THIS IS YOUR LAST TURN: end this reply with an edit block and an answer block, or you score zero."
_SCRIPTED_REPLY = 'Zeroing one head.\n\n```edit\nW["L5.out.W"][0:64, :] = 0\n```\n\n```answer\n5.0\n5.1\n5.2\n5.3\n5.4\n```'
NAMED = 5
TOP_TEN = 10
REPEAT_LENGTH = 50
REPEAT_SEQUENCES = 48
CHUNK = 4
BEST_PAIRING = [(6, 1), (7, 2), (5, 8), (6, 15), (9, 9)]
ATTENTION_TOP = [(9, 9), (18, 5), (11, 1), (12, 1), (7, 2)]


def prompt_for(model) -> str:
    base = family_prompt(model)
    return TASK + base[base.index("THE NETWORK:") :]


def fresh(model_tensors: dict[str, np.ndarray]):
    """Returns a runnable float32 medium holding these weights. Going through float64 avoids NaN logits seen on fresh loads."""
    model = load("medium").double().float()
    target = writable_tensors(model)
    with torch.no_grad():
        for name, values in model_tensors.items():
            target[name].copy_(torch.from_numpy(values.astype(np.float32)))
    return model


@torch.no_grad()
def scores(model) -> tuple[float, float]:
    """Returns repeat accuracy on repeated random tokens, and cross-entropy on real text."""
    generator = torch.Generator().manual_seed(SEED)
    half = torch.randint(1000, 50000, (REPEAT_SEQUENCES, REPEAT_LENGTH), generator=generator)
    tokens = torch.cat([half, half], 1)
    right = []
    for part in tokens.split(CHUNK):
        logits = model(part).logits[:, REPEAT_LENGTH:-1]
        if not torch.isfinite(logits).all():
            raise ValueError("non-finite logits")
        right.append((logits.argmax(-1) == part[:, REPEAT_LENGTH + 1 :]).float().mean())
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    with open(ROOT / TEXT_FILE, encoding="utf-8", errors="replace") as handle:
        encoded = tokenizer.encode(handle.read(600_000))
    rng = np.random.default_rng(SEED)
    starts = rng.choice(len(encoded) - TEXT_LENGTH, TEXT_SEQUENCES, replace=False)
    text = torch.tensor([encoded[int(start) : int(start) + TEXT_LENGTH] for start in starts])
    total, count = 0.0, 0
    for part in text.split(CHUNK):
        logits = model(part).logits[:, :-1]
        total += float(torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), part[:, 1:].reshape(-1), reduction="sum"))
        count += part[:, 1:].numel()
    return float(torch.stack(right).mean()), total / count


def zeroed(disguised: FamilyDisguise, real_heads: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    tensors = dict(disguised.tensors)
    for layer, head in real_heads:
        name = f"L{layer}.out.W"
        values = tensors[name].copy()
        scrambled = int(np.where(disguised.original_head[layer] == head)[0][0])
        values[scrambled * 64 : (scrambled + 1) * 64] = 0
        tensors[name] = values
    return tensors


def baselines() -> None:
    model = load("medium")
    disguised = disguise_family(model, BASELINE_SEED)
    del model
    base = scores(fresh(disguised.tensors))
    print(f"unedited: repeat accuracy {base[0]:.3f}, text loss {base[1]:.3f}", flush=True)
    sets = {
        "best-pairing 5 zeroed": BEST_PAIRING,
        "attention top 5 zeroed": ATTENTION_TOP,
        "earliest 5 above 0.5 attention zeroed": [(5, 8), (6, 1), (7, 2), (9, 3), (9, 9)],
        "best-pairing 10 zeroed": BEST_PAIRING + [(12, 0), (10, 1), (10, 8), (11, 1), (9, 4)],
        "all heads in layers 5 to 7 zeroed": [(layer, head) for layer in range(5, 8) for head in range(16)],
        "all heads in layers 6 to 8 zeroed": [(layer, head) for layer in range(6, 9) for head in range(16)],
        "all heads in layers 7 to 9 zeroed": [(layer, head) for layer in range(7, 10) for head in range(16)],
        "all heads in layers 5 to 9 zeroed": [(layer, head) for layer in range(5, 10) for head in range(16)],
        "all heads in layers 5 to 12 zeroed": [(layer, head) for layer in range(5, 13) for head in range(16)],
    }
    for label, heads in sets.items():
        repeat, text = scores(fresh(zeroed(disguised, heads)))
        print(f"{label}: repeat accuracy {repeat:.3f}, text loss {text:.3f}, rise {text - base[1]:.3f}", flush=True)
    rng = np.random.default_rng(BASELINE_SEED)
    results = []
    for _ in range(4):
        picked = [(int(i) // 16, int(i) % 16) for i in rng.choice(384, 5, replace=False)]
        results.append(scores(fresh(zeroed(disguised, picked))))
    results = np.array(results)
    print(f"4 random sets of 5 zeroed: repeat accuracy mean {results[:, 0].mean():.3f}, lowest {results[:, 0].min():.3f}; text rise mean {results[:, 1].mean() - base[1]:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["freeze", "baselines", "live", "scripted"])
    arguments = parser.parse_args()
    if arguments.mode == "baselines":
        baselines()
        return
    model = load("medium")
    prompt = prompt_for(model)
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if arguments.mode == "freeze":
        (ROOT / "prompts" / "write_medium.txt").write_text(prompt)
        MANIFEST.write_text(json.dumps({"write_medium": digest}))
        print("froze the prompt")
        return
    if digest != json.loads(MANIFEST.read_text())["write_medium"]:
        raise ValueError("the prompt no longer matches its frozen hash")
    truth = induction_scores(model.double().float())
    disguised = disguise_family(model, RUN_SEED)
    print("seed's scores before any call:", scores(fresh(disguised.tensors)), flush=True)
    scrambled = np.stack([truth[layer, disguised.original_head[layer]] for layer in range(model.config.n_layer)])
    del model
    order = np.argsort(-scrambled.ravel())
    top = {(int(i) // 16, int(i) % 16) for i in order[:NAMED]}
    top_ten = {(int(i) // 16, int(i) % 16) for i in order[:TOP_TEN]}
    weights = ROOT / "data" / "family" / f"blocks_{RUN_SEED}.npz"
    if not weights.exists():
        save_blocks(disguised, weights)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / f"write-medium__{arguments.mode}__{stamp}"
    run_dir.mkdir(parents=True)
    (run_dir / "disguise_map.json").write_text(json.dumps(disguised.original_head.tolist()))
    if arguments.mode == "live":
        from wsp.astra import call_astra

        def call(text: str, item: str) -> str:
            return call_astra(text, run_dir, item)["reply"]

        spent = lambda: spent_so_far(run_dir)
    else:
        call, spent = (lambda text, item: _SCRIPTED_REPLY), (lambda: 0.0)
    result = run_code_reading(prompt, weights, top, top_ten, RUN_SEED, call, spent, run_dir, MAX_COST, LAST_TURN)
    print(json.dumps(result.__dict__, default=list)[:300], flush=True)
    last_reply = [json.loads(line) for line in (run_dir / "transcript.jsonl").read_text().splitlines()][-1]["reply"]
    code = edit_code(last_reply)
    summary: dict = {"seed": RUN_SEED, "cost": spent(), "named_heads": [list(pick) for pick in result.answer or []]}
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


if __name__ == "__main__":
    main()
