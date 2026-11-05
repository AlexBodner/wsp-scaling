"""One sealed Astra call, saved in full through vlm-jigsaw's Codex client and call store."""

from pathlib import Path
from uuid import uuid4

from vlm_jigsaw.calls import CallStore
from vlm_jigsaw.models import TOOL_USE_DETECTED, ChatRequest, TextPart
from vlm_jigsaw.models.codex import CodexCLI, sealed_codex_binary

MODEL = "gpt-6-astra"
EFFORT = "xhigh"
CALL_TIMEOUT_S = 3 * 3600
# Codex does not enforce an output limit, so this is only recorded with the call.
MAX_OUTPUT_TOKENS_RECORDED = 128000
# Codex shows the model its working directory, so a call must not run in a folder whose path names the model or the arm.
CALLS_ROOT = Path.home() / "sealed_calls"


def _anonymous_calls(run_dir: Path) -> Path:
    """Returns the run's calls folder, kept under a meaningless path and linked from the run."""
    link = run_dir.resolve() / "calls"
    if not link.exists():
        target = CALLS_ROOT / uuid4().hex[:12]
        target.mkdir(parents=True)
        link.symlink_to(target)
    return link.resolve()


def bring_calls_home(run_dir: Path) -> None:
    """Moves a finished run's calls from the anonymous folder into the run."""
    link = run_dir.resolve() / "calls"
    if link.is_symlink():
        target = link.resolve()
        link.unlink()
        target.rename(link)


def call_astra(prompt: str, run_dir: Path, item: str) -> dict:
    """Sends one prompt in one fresh call and returns the reply, usage and seal details.

    Raises:
        RuntimeError: If the seal is broken, in which case the reply must not be used.
    """
    binary = sealed_codex_binary()
    log = CallStore(_anonymous_calls(run_dir)).log(item=item)
    model = CodexCLI(MODEL, EFFORT, calls=log, timeout_s=CALL_TIMEOUT_S, max_attempts=1)
    response = model.complete(ChatRequest(system="", parts=(TextPart(prompt),), max_output_tokens=MAX_OUTPUT_TOKENS_RECORDED))
    if response.stop_reason == TOOL_USE_DETECTED:
        raise RuntimeError(f"seal broken on {item}: {response.raw['seal']['problems']}")
    return {
        "reply": response.text,
        "stop_reason": response.stop_reason,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "binary": binary,
    }
