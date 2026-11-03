"""A read-only view of a model's weights, with a budget. It looks numbers up and computes nothing."""

import re

import numpy as np

from wsp.edits import _parse_index, split_statements

_READ_BLOCK = re.compile(r"```read[ \t]*\n(.*?)```", re.DOTALL)
_ANSWER_BLOCK = re.compile(r"```answer[ \t]*\n(.*?)```", re.DOTALL)
_READ = re.compile(r"^(?P<name>[A-Za-z][\w.]*)\[(?P<index>[^\]]*)\]$")


def _format(values: np.ndarray) -> str:
    if values.ndim == 0:
        return f"{values.item():.3g}"
    if values.ndim == 1:
        return " ".join(f"{value:.3g}" for value in values)
    return "\n".join(_format(row) for row in values)


class WeightViewer:
    """Serves slices of named tensors as rounded text until the budget of values runs out."""

    def __init__(self, tensors: dict[str, np.ndarray], budget: int) -> None:
        self.tensors = tensors
        self.remaining = budget
        self.requests: list[dict] = []

    def serve(self, reply: str) -> str:
        """Answers every read in the reply's last read block, and returns the text to send back."""
        blocks = _READ_BLOCK.findall(reply)
        if not blocks:
            return f"No read block found. Values remaining: {self.remaining}."
        sections = []
        for statement in split_statements(blocks[-1]):
            sections.append(self._serve_one(statement))
        sections.append(f"Values remaining: {self.remaining}.")
        return "\n".join(sections)

    def _serve_one(self, statement: str) -> str:
        match = _READ.match(statement)
        if match is None or match["name"] not in self.tensors:
            self.requests.append({"statement": statement, "values": 0, "error": "unreadable"})
            return f"{statement} : unreadable, 0 values charged"
        tensor = self.tensors[match["name"]]
        try:
            values = tensor[_parse_index(match["index"], tensor.shape)]
        except ValueError as error:
            self.requests.append({"statement": statement, "values": 0, "error": str(error)})
            return f"{statement} : {error}, 0 values charged"
        values = np.asarray(values)
        if values.size > self.remaining:
            self.requests.append({"statement": statement, "values": 0, "error": "over budget"})
            return f"{statement} : asks for {values.size} values, over budget, 0 values charged"
        self.remaining -= values.size
        self.requests.append({"statement": statement, "values": int(values.size)})
        return f"{statement} =\n{_format(values)}"


def parse_answer(reply: str, count: int) -> list[tuple[int, int]] | None:
    """Returns the first `count` distinct heads in the last answer block, or None if there is no block.

    Lines that are not `layer.head` are skipped, and a shorter answer stays short.
    """
    blocks = _ANSWER_BLOCK.findall(reply)
    if not blocks:
        return None
    heads: list[tuple[int, int]] = []
    for line in blocks[-1].splitlines():
        match = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", line)
        if match and (head := (int(match[1]), int(match[2]))) not in heads:
            heads.append(head)
    return heads[:count]
