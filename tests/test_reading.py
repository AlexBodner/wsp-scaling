import numpy as np
import pytest
import torch
from transformers import GPT2LMHeadModel

from wsp.disguise import HEADS, disguise, model_from
from wsp.network import TokenShuffle, real_network
from wsp.prompts import NETWORK_SPEC
from wsp.reading import HEADS_TO_NAME, blind_prompt, hinted_prompt, read_prompt, run_reading
from wsp.viewer import WeightViewer, parse_answer


@pytest.fixture(scope="module")
def shuffle():
    return TokenShuffle()


@pytest.fixture(scope="module")
def disguised(shuffle):
    return disguise(5, shuffle)


@torch.no_grad()
def test_disguise_keeps_behaviour_and_moves_each_head_where_it_says(shuffle, disguised):
    real = real_network(shuffle).double()
    real.set_attn_implementation("eager")
    changed = model_from(disguised)
    ids = torch.randint(1000, 50000, (2, 40), generator=torch.Generator().manual_seed(3))
    before, after = real(ids, output_attentions=True), changed(ids, output_attentions=True)
    assert (before.logits - after.logits).abs().max() < 1e-8
    for layer, (attention_before, attention_after) in enumerate(zip(before.attentions, after.attentions)):
        order = torch.from_numpy(disguised.original_head[layer])
        assert (attention_before[:, order] - attention_after).abs().max() < 1e-9


def test_disguise_changes_the_numbers_and_removes_the_key_bias_signal(shuffle, disguised):
    real = {name: value.detach().numpy() for name, value in dict(real_network(shuffle).named_parameters()).items()}
    assert not np.allclose(real["transformer.h.5.attn.c_attn.weight"], disguised.tensors["L5.qkv.W"])
    assert not np.allclose(real["transformer.h.5.attn.c_attn.bias"][768:1536], disguised.tensors["L5.qkv.b"][768:1536])


def test_viewer_charges_only_what_it_serves_and_refuses_over_budget():
    viewer = WeightViewer({"A": np.arange(12.0).reshape(3, 4)}, budget=5)
    text = viewer.serve("```read\nA[0, :]\nA[1:3, :]\nA[2, 0]\nnope[0]\n```")
    assert viewer.remaining == 0
    assert "A[1:3, :] : asks for 8 values, over budget" in text
    assert "nope[0] : unreadable" in text
    assert "Values remaining: 0." in text


def test_viewer_rounds_to_three_significant_digits():
    viewer = WeightViewer({"A": np.array([0.123456, 1234.5])}, budget=2)
    assert "0.123 1.23e+03" in viewer.serve("```read\nA[:]\n```")


def test_viewer_cannot_compute():
    viewer = WeightViewer({"A": np.ones((2, 2))}, budget=10)
    assert "unreadable" in viewer.serve("```read\nA[0, :] @ A[1, :]\n```")
    assert viewer.remaining == 10


def test_answer_takes_the_first_distinct_heads_and_skips_other_lines():
    assert parse_answer("```answer\n1.2\n1.2\nnote\n3.4\n```", 5) == [(1, 2), (3, 4)]
    assert parse_answer("no block", 5) is None


def test_a_run_replays_the_transcript_and_stops_at_the_answer(disguised, tmp_path):
    prompts = []

    def reader(prompt, item):
        prompts.append(prompt)
        return "```read\nL0.qkv.b[0:2]\n```" if item == "turn-1" else "```answer\n1.1\n2.2\n```"

    result = run_reading("START", disguised, {(1, 1), (9, 9)}, 5, 100, reader, tmp_path)
    assert result.turns == 2
    assert result.values_read == 2
    assert result.hits == 1
    assert "Values remaining: 98." in prompts[1]
    assert prompts[1].startswith("START")


def test_a_run_without_an_answer_scores_zero_at_the_turn_limit(disguised, tmp_path):
    result = run_reading("START", disguised, {(1, 1)}, 5, 10, lambda prompt, item: "thinking", tmp_path)
    assert result.answer is None
    assert result.hits == 0


def test_blind_run_is_one_reply_and_reads_nothing(disguised, tmp_path):
    calls = []

    def reader(prompt, item):
        calls.append(item)
        return "```read\nL0.qkv.b[0:2]\n```"

    result = run_reading(blind_prompt(), disguised, set(), 5, 0, reader, tmp_path)
    assert calls == ["turn-1"]
    assert result.values_read == 0


@pytest.mark.parametrize("build", [read_prompt, hinted_prompt, blind_prompt])
def test_prompts_do_not_name_the_model_or_say_weights_start_at_zero(build):
    text = build()
    assert "gpt" not in text.lower()
    assert "STARTS AT 0" not in text
    assert "u * Phi(u), element by element" not in text


def test_only_the_hinted_prompt_names_a_statistic():
    assert "Q[r] . K[r]" not in read_prompt()
    assert "Q[r] . K[r]" not in blind_prompt()
    assert "Q[r] . K[r]" in hinted_prompt()


def test_write_prompt_spec_is_unchanged_by_the_refactor():
    assert "EVERY WEIGHT STARTS AT 0, normalization scales included" in NETWORK_SPEC


def test_scores_measured_on_the_real_model_match_the_disguised_model(shuffle, disguised):
    from wsp.reading import disguised_scores, head_scores

    real = real_network(shuffle).double()
    real.set_attn_implementation("eager")
    measured = head_scores(model_from(disguised))
    np.testing.assert_allclose(disguised_scores(head_scores(real), disguised), measured, atol=1e-8)
