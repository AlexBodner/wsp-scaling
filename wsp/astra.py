"""One sealed Astra call, saved in full through vlm-jigsaw's Codex client and call store."""

from pathlib import Path

from vlm_jigsaw.calls import CallStore
from vlm_jigsaw.models import TOOL_USE_DETECTED, ChatRequest, TextPart
from vlm_jigsaw.models.codex import CodexCLI, sealed_codex_binary

MODEL = "gpt-6-astra"
EFFORT = "xhigh"
CALL_TIMEOUT_S = 3 * 3600
# Codex does not enforce an output limit, so this is only recorded with the call.
MAX_OUTPUT_TOKENS_RECORDED = 128000


def call_astra(prompt: str, run_dir: Path, item: str) -> dict:
    """Sends one prompt in one fresh call and returns the reply, usage and seal details.

    Raises:
        RuntimeError: If the seal is broken, in which case the reply must not be used.
    """
    binary = sealed_codex_binary()
    log = CallStore(run_dir.resolve() / "calls").log(item=item)
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
