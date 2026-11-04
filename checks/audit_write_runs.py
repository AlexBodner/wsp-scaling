"""Audits the three write runs from public logs: what their code imports and calls, and whether any of it looks like running the model. Makes no Astra calls."""
import ast
import json
import re
from pathlib import Path

from checks.audit_python_runs import _FORBIDDEN_CALLS, check_logs, turns

ROOT = Path(__file__).parent.parent
PROMPTS = ROOT / "prompts"
RUNS = (
    ("runs/write-write-1__live__20261104T053859Z", "WRITE_MANIFEST.json", "write"),
    ("runs/write-write-2__live__20261104T061358Z", "WRITE_MANIFEST.json", "write"),
    ("runs/write-medium__live__20261104T072216Z", "WRITE_MEDIUM_MANIFEST.json", "write_medium"),
)
# The pipeline's own flag, which also fires on comments and printed text, so each hit is shown in its line.
FORWARD_WORDS = re.compile(r"softmax|gelu|relu|layer_?norm|forward|attention\(|torch\.nn", re.IGNORECASE)
# What running the model on inputs would need: nonlinearities, or inputs drawn at random.
RUNNING_THE_MODEL = re.compile(r"np\.exp\(|np\.tanh\(|\berf\(|random\.|default_rng|randn|randint")


def audit_code(label: str, code: str) -> None:
    tree = ast.parse(code)
    calls = {node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "") for node in ast.walk(tree) if isinstance(node, ast.Call)}
    imports = {(node.module if isinstance(node, ast.ImportFrom) else alias.name) for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    print(f"  {label}: imports {sorted(imports)}, forbidden calls {sorted(calls & _FORBIDDEN_CALLS)}")
    for line in code.splitlines():
        if FORWARD_WORDS.search(line) or RUNNING_THE_MODEL.search(line):
            print(f"    flagged line: {line.strip()[:150]}")


def main() -> None:
    for folder, manifest, key in RUNS:
        run = ROOT / folder
        print(folder)
        check_logs(run, json.loads((PROMPTS / manifest).read_text())[key])
        for turn in turns(run):
            if turn["code"]:
                audit_code(f"turn {turn['turn']}", turn["code"])
        audit_code("edit.py", (run / "edit.py").read_text().split('_original = np.load("blocks.npz")')[0])


if __name__ == "__main__":
    main()
