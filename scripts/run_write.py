"""Asks a reader with Python to edit a scrambled GPT-2 so it stops repeating. `--live` makes sealed Astra calls, and `--scripted` makes none."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.run_reading_python import spent_so_far
from wsp.code_reader import _FORWARD_PASS_WORDS, run_code, run_code_reading, save_blocks
from wsp.disguise import HEADS, disguise, model_from
from wsp.network import TokenShuffle, real_network
from wsp.reading import disguised_scores, head_scores
from wsp.write_task import EDITED_FILE, apply_edit, detailed_scores, edit_code, score_disguised, touched_heads, write_prompt, zero_heads

ROOT = Path(__file__).parent.parent
SEEDS = {"write-1": 1301, "write-2": 1302, "scripted": 1303}
MANIFEST = ROOT / "prompts" / "WRITE_MANIFEST.json"
MAX_COST = 9.0
NAMED = 5
TOP_TEN = 10
_SCRIPTED_REPLY = 'Zeroing one head.\n\n```edit\nW["L5.out.W"][0:64, :] = 0\n```\n\n```answer\n5.0\n5.1\n5.2\n5.3\n5.4\n```'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=[*SEEDS, "freeze"])
    parser.add_argument("--live", action="store_true")
    arguments = parser.parse_args()
    prompt = write_prompt()
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if arguments.arm == "freeze":
        (ROOT / "prompts" / "write.txt").write_text(prompt)
        MANIFEST.write_text(json.dumps({"write": digest}))
        print("froze the prompt")
        return
    if digest != json.loads(MANIFEST.read_text())["write"]:
        raise ValueError("the prompt no longer matches its frozen hash")
    if arguments.live == (arguments.arm == "scripted"):
        raise ValueError("the scripted arm runs without --live, and the others only with it")

    seed = SEEDS[arguments.arm]
    shuffle = TokenShuffle()
    disguised = disguise(seed, shuffle)
    weights = ROOT / "data" / "write" / f"blocks_{seed}.npz"
    if not weights.exists():
        save_blocks(disguised, weights)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "live" if arguments.live else "scripted"
    run_dir = ROOT / "runs" / f"write-{arguments.arm}__{label}__{stamp}"
    run_dir.mkdir(parents=True)
    if arguments.live:
        from wsp.astra import call_astra

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
    (run_dir / "disguise_map.json").write_text(json.dumps(disguised.original_head.tolist()))
    print(json.dumps(result.__dict__, default=list)[:300])
    last_reply = [json.loads(line) for line in (run_dir / "transcript.jsonl").read_text().splitlines()][-1]["reply"]
    code = edit_code(last_reply)
    summary = {"seed": seed, "cost": spent()}
    if code is None:
        summary["edit"] = "no edit block in the final reply"
    else:
        (run_dir / "edit.py").write_text(code)
        summary["edit_forward_pass_flags"] = sorted(set(_FORWARD_PASS_WORDS.findall(code)))
        edited_path = run_dir / EDITED_FILE
        output = run_code(code, weights, keep=(EDITED_FILE, edited_path))
        summary["edit_output"] = output[-600:]
        if edited_path.exists():
            with np.load(edited_path) as archive:
                edited = {name: archive[name] for name in archive.files}
            before = score_disguised(disguised, shuffle)
            try:
                changed_model = apply_edit(disguised, edited)
                after = score_disguised(changed_model, shuffle)
            except ValueError as error:
                changed_model, after = None, None
                summary["edit_refused"] = str(error)
            if after is None or not np.isfinite(after).all():
                summary["counted_as"] = "no edit"
            else:
                changed = sum(int((edited[name] != disguised.tensors[name].astype(np.float32)).sum()) for name in edited)
                touched = touched_heads(disguised, edited)
                named = [tuple(pick) for pick in result.answer or []]
                summary |= {"before": before, "after": after, "values_changed": changed, "tensors_changed": sorted(edited)[:40], "touched_heads": touched, "named_heads": named}
                summary["zeroing_named_heads"] = score_disguised(zero_heads(disguised, named), shuffle) if named else None
                summary["zeroing_touched_heads"] = score_disguised(zero_heads(disguised, touched), shuffle) if touched else None
                summary["detail"] = detailed_scores(model_from(changed_model), model_from(disguised), shuffle)
            edited_path.unlink()
    (run_dir / "write_score.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:1500])
    weights.unlink()


if __name__ == "__main__":
    main()
