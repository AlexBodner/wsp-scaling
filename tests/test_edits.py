import numpy as np
import pytest
import torch

from wsp.edits import apply_edits


def _tensors():
    return {"W": torch.zeros(4, 3), "b": torch.zeros(5)}


def _reply(*statements):
    return "design\n```edits\n" + "\n".join(statements) + "\n```\nend"


@pytest.mark.parametrize(
    ("statement", "expected_row", "expected_value"),
    [
        ("W[1, 2] = 2.5", 1, [0, 0, 2.5]),
        ("W[1, :] = 3", 1, [3, 3, 3]),
        ("W[2, 0:2] = [1, -1]", 2, [1, -1, 0]),
        ("W[0, 0:3:2] = [4, 5]", 0, [4, 0, 5]),
    ],
)
def test_statements_set_the_expected_entries(statement, expected_row, expected_value):
    tensors = _tensors()
    report = apply_edits(_reply(statement), tensors)
    assert report.errors == []
    assert tensors["W"][expected_row].tolist() == expected_value


def test_a_list_can_continue_over_lines_and_comments_are_ignored():
    tensors = _tensors()
    apply_edits(_reply("b[0:3] = [1, # first", "  2, 3]"), tensors)
    assert tensors["b"].tolist() == [1, 2, 3, 0, 0]


@pytest.mark.parametrize("operator", ["*=", "+=", "-="])
def test_arithmetic_is_refused_by_default(operator):
    tensors = _tensors()
    tensors["b"][0] = 2
    report = apply_edits(_reply(f"b[0] {operator} 3"), tensors)
    assert report.applied == 0
    assert tensors["b"][0] == 2


def test_arithmetic_works_when_the_control_prompt_offers_it():
    tensors = _tensors()
    tensors["b"][0] = 2
    apply_edits(_reply("b[0] *= 3"), tensors, allow_arithmetic=True)
    assert tensors["b"][0] == 6


@pytest.mark.parametrize(
    "statement",
    ["W[9, 0] = 1", "W[0] = 1", "nope[0] = 1", "b[0] = 1/3", "b[0] = 2e6", "b[0:2] = [1, 2, 3]", "b[0] = 'x'", "b[0] 1"],
)
def test_an_unreadable_statement_is_skipped_and_reported_while_others_apply(statement):
    tensors = _tensors()
    report = apply_edits(_reply(statement, "b[4] = 1"), tensors)
    assert len(report.errors) == 1
    assert report.applied == 1
    assert tensors["b"][4] == 1


def test_only_the_last_block_counts():
    tensors = _tensors()
    apply_edits("```edits\nb[0] = 1\n```\nthen\n```edits\nb[1] = 2\n```", tensors)
    assert tensors["b"].tolist() == [0, 2, 0, 0, 0]


def test_a_reply_without_a_block_changes_nothing():
    tensors = _tensors()
    report = apply_edits("no weights here", tensors)
    assert not report.has_block
    assert tensors["b"].sum() == 0


def test_parser_reproduces_the_published_breakout_network():
    """Their reply, parsed by our code, must equal the network they published."""
    from pathlib import Path

    control = Path(__file__).parent.parent / "data" / "control"
    if not control.exists():
        pytest.skip("control data not downloaded")
    published = np.load(control / "hand.npz", allow_pickle=False)
    names = {"W1": "dense1/kernel", "b1": "dense1/bias", "W2": "dense2/kernel", "b2": "dense2/bias",
             "W3": "logits/kernel", "b3": "logits/bias"}
    tensors = {name: torch.zeros(published[f"actor/{key}"].shape, dtype=torch.float32) for name, key in names.items()}
    report = apply_edits((control / "answer.md").read_text(), tensors, allow_arithmetic=True)
    assert report.errors == []
    assert report.statements == 92
    for name, key in names.items():
        np.testing.assert_array_equal(tensors[name].numpy(), published[f"actor/{key}"])
