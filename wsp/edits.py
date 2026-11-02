"""Parses the typed weight statements in a writer's reply and applies them to tensors."""

import ast
import re
from dataclasses import dataclass, field

import torch

MAX_ABSOLUTE_VALUE = 1e6
_EDIT_BLOCK = re.compile(r"```edits[ \t]*\n(.*?)```", re.DOTALL)
_STATEMENT = re.compile(
    r"^(?P<name>[A-Za-z][\w.]*)\[(?P<index>[^\]]*)\]\s*(?P<operator>\*=|\+=|-=|=)\s*(?P<value>.+)$",
    re.DOTALL,
)
_RANGE = re.compile(r"^(\d*):(\d*)(?::(\d+))?$")


@dataclass
class EditReport:
    """What happened when one reply's edit block was applied to the tensors."""

    has_block: bool = False
    statements: int = 0
    applied: int = 0
    entries_written: int = 0
    errors: list[str] = field(default_factory=list)


def last_edit_block(reply: str) -> str | None:
    """Returns the text of the last edit block in a reply, or None if it has none."""
    blocks = _EDIT_BLOCK.findall(reply)
    return blocks[-1] if blocks else None


def split_statements(block: str) -> list[str]:
    """Splits a block into statements, joining lists that continue over several lines."""
    statements = []
    pending = ""
    for raw_line in block.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        pending = f"{pending} {line}".strip()
        if pending.count("[") == pending.count("]"):
            statements.append(pending)
            pending = ""
    if pending:
        statements.append(pending)
    return statements


def _parse_index(text: str, shape: torch.Size) -> tuple:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != len(shape):
        raise ValueError(f"{len(parts)} indices for a tensor with {len(shape)} axes")
    index = []
    for part, size in zip(parts, shape):
        if part.isdigit():
            if int(part) >= size:
                raise ValueError(f"index {part} is out of range for axis of size {size}")
            index.append(int(part))
            continue
        match = _RANGE.match(part)
        if match is None:
            raise ValueError(f"cannot read index {part!r}")
        start, stop, step = match.groups()
        if step is not None and int(step) == 0:
            raise ValueError("a range step cannot be 0")
        index.append(slice(int(start) if start else None, int(stop) if stop else None, int(step) if step else None))
    return tuple(index)


def _parse_value(text: str) -> torch.Tensor:
    try:
        value = ast.literal_eval(text.strip())
    except (ValueError, SyntaxError) as error:
        raise ValueError(f"cannot read value {text.strip()[:60]!r}") from error
    if isinstance(value, bool) or not isinstance(value, (int, float, list)):
        raise ValueError(f"value is not a number or a list: {text.strip()[:60]!r}")
    try:
        tensor = torch.tensor(value, dtype=torch.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(f"list is not a rectangle of numbers: {text.strip()[:60]!r}") from error
    if not tensor.isfinite().all() or tensor.abs().max() > MAX_ABSOLUTE_VALUE:
        raise ValueError(f"value is not finite or exceeds {MAX_ABSOLUTE_VALUE:g}")
    return tensor


def apply_edits(reply: str, tensors: dict[str, torch.Tensor], allow_arithmetic: bool = False) -> EditReport:
    """Applies the last edit block of a reply to tensors in place.

    Statements apply in order, and one that cannot be read is skipped and reported.

    Args:
        reply: The writer's full reply.
        tensors: The writable tensors, by name.
        allow_arithmetic: Accepts `*=`, `+=` and `-=`. Only the post's own prompt, used as the
            positive control, offers these.

    Returns:
        What was applied, including every statement that was skipped and why.
    """
    report = EditReport()
    block = last_edit_block(reply)
    if block is None:
        return report
    report.has_block = True
    for statement in split_statements(block):
        report.statements += 1
        try:
            report.entries_written += _apply_statement(statement, tensors, allow_arithmetic)
            report.applied += 1
        except ValueError as error:
            report.errors.append(f"{statement[:80]}: {error}")
    return report


def _apply_statement(statement: str, tensors: dict[str, torch.Tensor], allow_arithmetic: bool) -> int:
    match = _STATEMENT.match(statement)
    if match is None:
        raise ValueError("cannot read statement")
    name, operator = match["name"], match["operator"]
    if name not in tensors:
        raise ValueError(f"unknown tensor {name!r}")
    if operator != "=" and not allow_arithmetic:
        raise ValueError(f"operator {operator} is not allowed")
    tensor = tensors[name]
    index = _parse_index(match["index"], tensor.shape)
    value = _parse_value(match["value"])
    target_shape = tensor[index].shape
    if value.ndim > 0 and value.shape != target_shape:
        raise ValueError(f"values have shape {tuple(value.shape)}, the target has {tuple(target_shape)}")
    value = value.to(tensor.dtype)
    with torch.no_grad():
        if operator == "=":
            tensor[index] = value
        elif operator == "*=":
            tensor[index] *= value
        elif operator == "+=":
            tensor[index] += value
        else:
            tensor[index] -= value
    return int(torch.tensor(target_shape).prod()) if len(target_shape) else 1
