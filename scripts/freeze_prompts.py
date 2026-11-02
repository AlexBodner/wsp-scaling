"""Writes the frozen prompts and their hashes, before any Astra call."""

import hashlib
import json
from pathlib import Path

from wsp.control import CONTROL_DATA
from wsp.hand import capitals_reply
from wsp.network import TokenShuffle
from wsp.prompts import capitals_prompt, language_prompt, reverse_prompt_text

OUTPUT = Path(__file__).parent.parent / "prompts"


def main() -> None:
    shuffle = TokenShuffle()
    prompts = {
        "capitals": capitals_prompt(shuffle),
        "reverse": reverse_prompt_text(shuffle),
        "language": language_prompt(shuffle),
        "control": (CONTROL_DATA / "message.txt").read_text(encoding="utf-8"),
    }
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "hand_capitals_reply.md").write_text(capitals_reply(shuffle), encoding="utf-8")
    manifest = {}
    for name, text in prompts.items():
        # The control prompt holds GPL-licensed MinAtar source, so only its hash goes in the repo.
        if name != "control":
            (OUTPUT / f"{name}.txt").write_text(text, encoding="utf-8")
        manifest[name] = {"sha256": hashlib.sha256(text.encode()).hexdigest(), "characters": len(text)}
    (OUTPUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
