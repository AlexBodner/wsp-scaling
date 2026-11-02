from types import SimpleNamespace

import torch

from wsp.network import TokenShuffle
from wsp.predict import parse_predictions, predict_texts
from wsp.tasks import load_capitals, reverse_prompt, reverse_test_strings, score_reverse


class _PerfectReverser:
    """A stand-in model that always outputs the correct next letter, to test the scorer."""

    def __init__(self, shuffle):
        self.shuffle = shuffle
        self.arrow = shuffle.encode(" ->")[0]
        self.size = len(shuffle.tokenizer)

    def __call__(self, input_ids):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], self.size)
        for row, ids in enumerate(input_ids.tolist()):
            arrow_at = ids.index(self.arrow)
            letters, generated = ids[2:arrow_at], ids[arrow_at + 1 :]
            logits[row, -1, list(reversed(letters))[len(generated)]] = 1
        return SimpleNamespace(logits=logits)


def test_reverse_scorer_gives_full_marks_to_a_perfect_network():
    shuffle = TokenShuffle()
    score = score_reverse(_PerfectReverser(shuffle), shuffle)
    assert score["exact"].value == 1.0
    assert score["exact"].passes
    assert score["per_position"] == 1.0


def test_reverse_prompt_tokenizes_as_the_task_describes():
    shuffle = TokenShuffle()
    tokens = shuffle.encode(reverse_prompt("abc"))
    assert [shuffle.decode_token(token) for token in tokens] == ["rev", ":", " a", " b", " c", " ->"]


def test_test_sets_are_fixed():
    assert reverse_test_strings() == reverse_test_strings()
    assert len(reverse_test_strings()) == 1000
    assert len(load_capitals()) == 179


def test_predict_uses_half_task_and_half_unrelated_texts():
    for experiment in ("capitals", "reverse", "language"):
        assert len(predict_texts(experiment)) == 50


def test_parse_predictions_reads_the_last_block_and_ignores_unreadable_or_out_of_range_lines():
    reply = "```predictions\n0 1\n```\nthen\n```predictions\n0 5\nnote\n1 7\n99 3\n```"
    assert parse_predictions(reply, expected=50) == {0: 5, 1: 7}
