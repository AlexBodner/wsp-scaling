"""Audits the write, swap and blind runs from public logs: what their code imports and calls, and whether any of it looks like running the model. Makes no Astra calls."""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

from checks.audit_python_runs import _FORBIDDEN_CALLS, calls, check_logs, turns

ROOT = Path(__file__).parent.parent
PROMPTS = ROOT / "prompts"
RUNS = (
    ("write-write-*", "WRITE_MANIFEST.json", "write"),
    ("write-medium", "WRITE_MEDIUM_MANIFEST.json", "write_medium"),
    ("write-neo-*", "WRITE_NEO_MANIFEST.json", "write_neo"),
    ("swap-*", "SWAP_MANIFEST.json", "swap"),
    ("neo-blind", "WRITE_NEO_MANIFEST.json", "blind_neo"),
)
# The pipeline's own flag, which also fires on comments and printed text, so each hit is shown in its line.
FORWARD_WORDS = re.compile(r"softmax|gelu|relu|layer_?norm|forward|attention\(|torch\.nn", re.IGNORECASE)
# What running the model on inputs would need: nonlinearities, or inputs drawn at random.
RUNNING_THE_MODEL = re.compile(r"\.exp\(|torch\.exp|tanh|\berf\(|random\.|default_rng|randn|randint")


def audit_code(label: str, code: str) -> None:
    tree = ast.parse(code)
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "") for node in ast.walk(tree) if isinstance(node, ast.Call)}
    imports = {(node.module if isinstance(node, ast.ImportFrom) else alias.name) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    print(f"  {label}: imports {sorted(imports)}, forbidden calls {sorted(calls & _FORBIDDEN_CALLS)}")
    for line in code.splitlines():
        if FORWARD_WORDS.search(line) or RUNNING_THE_MODEL.search(line):
            print(f"    flagged line: {line.strip()[:150]}")


def audit_blind(run: Path, frozen_prompt_hash: str) -> None:
    """Prints what a one-call run with no code shows."""
    records = calls(run)
    print(f"  prompt matches frozen hash: {hashlib.sha256(records[0]['request']['user_text'].encode()).hexdigest() == frozen_prompt_hash}")
    print(f"  event item types over {len(records)} calls: {sorted({str((event.get('item') or {}).get('type')) for record in records for event in record['response']['events']} - {'None'})}")
    print("  no code: this call has no Python")


def audit_run(run: Path, frozen_prompt_hash: str) -> None:
    print(run.relative_to(ROOT))
    if not (run / "edit.py").exists():
        audit_blind(run, frozen_prompt_hash)
        return
    check_logs(run, frozen_prompt_hash)
    for turn in turns(run):
        if turn["code"]:
            audit_code(f"turn {turn['turn']}", turn["code"])
    audit_code("edit.py", (run / "edit.py").read_text().split('_original = np.load("blocks.npz")')[0])


def main() -> None:
    for pattern, manifest, key in RUNS:
        for run in sorted((ROOT / "runs").glob(f"{pattern}__live__*")):
            # A run in progress keeps its calls behind a link until it finishes.
            if (run / "calls").is_symlink() or not (run / "result.json").exists() and not (run / "reply.md").exists():
                print(f"skipping {run.name}: not finished", file=sys.stderr)
                continue
            audit_run(run, json.loads((PROMPTS / manifest).read_text())[key])


if __name__ == "__main__":
    main()
