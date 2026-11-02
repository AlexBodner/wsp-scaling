import math

import numpy as np
import pytest
import torch

from wsp.edits import apply_edits
from wsp.network import TokenShuffle, real_network, writable_tensors, zero_network
from wsp.tasks import top_tokens


@pytest.fixture(scope="module")
def shuffle():
    return TokenShuffle()


def _layer_norm(u, scale, shift):
    mean = u.mean(dim=-1, keepdim=True)
    variance = u.var(dim=-1, keepdim=True, unbiased=False)
    return (u - mean) / torch.sqrt(variance + 1e-5) * scale + shift


def _gelu(u):
    return u * 0.5 * (1 + torch.erf(u / math.sqrt(2)))


def _forward_as_written_in_the_prompt(tensors, tokens):
    """Follows the prompt's equations line by line, independently of the GPT-2 code."""
    n = len(tokens)
    x = tensors["E"][tokens] + tensors["P"][: n]
    for layer in range(12):
        prefix = f"L{layer}"
        y = _layer_norm(x, tensors[f"{prefix}.norm1.scale"], tensors[f"{prefix}.norm1.shift"])
        a = y @ tensors[f"{prefix}.qkv.W"] + tensors[f"{prefix}.qkv.b"]
        q, k, v = a[:, :768], a[:, 768:1536], a[:, 1536:]
        heads = []
        for head in range(12):
            span = slice(64 * head, 64 * head + 64)
            scores = q[:, span] @ k[:, span].T / 8
            scores = scores.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), -torch.inf)
            heads.append(scores.softmax(dim=-1) @ v[:, span])
        c = torch.cat(heads, dim=-1)
        x = x + c @ tensors[f"{prefix}.out.W"] + tensors[f"{prefix}.out.b"]
        z = _layer_norm(x, tensors[f"{prefix}.norm2.scale"], tensors[f"{prefix}.norm2.shift"])
        up = _gelu(z @ tensors[f"{prefix}.up.W"] + tensors[f"{prefix}.up.b"])
        x = x + up @ tensors[f"{prefix}.down.W"] + tensors[f"{prefix}.down.b"]
    x = _layer_norm(x, tensors["F.scale"], tensors["F.shift"])
    return x @ tensors["E"].T


@torch.no_grad()
def test_prompt_equations_match_the_real_network_on_random_weights():
    torch.manual_seed(0)
    model = zero_network()
    tensors = writable_tensors(model)
    for tensor in tensors.values():
        tensor.normal_(0, 0.02)
    tokens = torch.tensor([5, 90, 3000, 7, 12])
    expected = model(input_ids=tokens[None]).logits[0]
    written = _forward_as_written_in_the_prompt(tensors, tokens)
    torch.testing.assert_close(written, expected, rtol=1e-3, atol=1e-3)


def test_every_parameter_is_writable_and_the_count_matches_the_prompt():
    model = zero_network()
    tensors = writable_tensors(model)
    assert sum(tensor.numel() for tensor in tensors.values()) == 124_439_808
    assert {id(t) for t in tensors.values()} == {id(p) for p in model.parameters()}


def test_output_uses_the_same_table_as_input():
    model = zero_network()
    tensors = writable_tensors(model)
    apply_edits("```edits\nE[3, 0] = 1\nF.scale[:] = 1\n```", tensors)
    logits = model(input_ids=torch.tensor([[3]])).logits[0, -1]
    assert logits.argmax() == 3


def test_a_zero_network_ties_every_token_and_answers_the_lowest_id():
    model = zero_network()
    assert top_tokens(model, [[5, 6], [7, 8]]) == [0, 0]


@torch.no_grad()
def test_shuffling_does_not_change_trained_behaviour(shuffle):
    text = "The capital of France is"
    renumbered = real_network(shuffle)
    original = type(renumbered).from_pretrained("gpt2").eval()
    ids = shuffle.tokenizer(text, add_special_tokens=False).input_ids
    expected = original(input_ids=torch.tensor([ids])).logits[0, -1]
    logits = renumbered(input_ids=torch.tensor([shuffle.encode(text)])).logits[0, -1]
    torch.testing.assert_close(logits[torch.from_numpy(shuffle.to_shuffled)], expected, rtol=1e-4, atol=1e-3)
    assert logits.argmax().item() == shuffle.to_shuffled[expected.argmax().item()]


def test_shuffle_is_fixed_and_hides_the_real_ids(shuffle):
    assert np.array_equal(TokenShuffle().to_shuffled, shuffle.to_shuffled)
    real_ids = shuffle.tokenizer(" Paris", add_special_tokens=False).input_ids
    assert shuffle.encode(" Paris") != real_ids
    assert shuffle.decode_token(shuffle.encode(" Paris")[0]) == " Paris"
