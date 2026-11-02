"""Builds the prompts Astra receives. They describe the task only, with no strategy."""

import json

from wsp.network import TokenShuffle
from wsp.tasks import (
    LANGUAGE_MAX_TOKENS,
    LETTERS,
    REVERSE_LENGTHS,
    VISIBLE_PHRASINGS,
    capital_tokens,
    load_capitals,
    token_counts,
)

_OPENING = """YOUR TASK: write, from scratch, the weights of a language model so that it does the task below. There is no training: no data, no loss, no gradients, no fine-tuning; you choose every number yourself. You have no tools: no code execution, no calculator, nothing to run. This is a single response and you will get no feedback. First describe your design briefly, then write the edits block.
"""

NETWORK_SPEC = """THE NETWORK: a transformer with 12 layers of width 768, 12 attention heads of size 64, and 124,439,808 weights. EVERY WEIGHT STARTS AT 0, normalization scales included; you set the weights you need.

Tensors (rows x columns):
  E: 50257 x 768 (one row per token, used for both input and output)
  P: 1024 x 768 (one row per position)
  F.scale, F.shift: 768
  and for each layer i from 0 to 11:
  Li.norm1.scale, Li.norm1.shift: 768
  Li.qkv.W: 768 x 2304, Li.qkv.b: 2304
  Li.out.W: 768 x 768, Li.out.b: 768
  Li.norm2.scale, Li.norm2.shift: 768
  Li.up.W: 768 x 3072, Li.up.b: 3072
  Li.down.W: 3072 x 768, Li.down.b: 768

The computation, for input tokens t_0 ... t_(n-1), with no start token added:
  x_j = E[t_j] + P[j]
  For each layer i in order:
    y_j = LN(x_j; Li.norm1)
    a_j = y_j @ Li.qkv.W + Li.qkv.b. Split a_j into q_j (entries 0-767), k_j (768-1535), v_j (1536-2303).
    For each head h from 0 to 11, take entries 64h to 64h+63 of q, k and v, and compute
      w_ji = softmax over i <= j of (q_j . k_i) / 8, then o_j = sum over i <= j of w_ji v_i
    c_j = the 12 heads' o_j joined in order (768 entries)
    x_j = x_j + c_j @ Li.out.W + Li.out.b
    z_j = LN(x_j; Li.norm2)
    x_j = x_j + gelu(z_j @ Li.up.W + Li.up.b) @ Li.down.W + Li.down.b
  x_j = LN(x_j; F)
  logits_j = x_j @ transpose(E), the scores for the token after position j.
LN(u; n) = (u - mean(u)) / sqrt(var(u) + 0.00001) * n.scale + n.shift, element by element, with the mean and population variance taken over the 768 entries of u.
gelu(u) = u * Phi(u), element by element, where Phi is the standard normal cumulative distribution.
The model's top token is the one with the largest logit; a tie goes to the lowest token ID.
"""

_SYNTAX = """HOW TO WRITE THE WEIGHTS (numbers only; nothing is run). Write a code block whose opening line is three backticks followed by the word edits, one statement per line ('#' starts a comment; a long list may continue over the next lines).

```edits
E[12, 0] = 2.5                          # one entry: row 12, column 0
L0.qkv.W[0:6, 3] = [1, 0, -1, 0, 2, 0]  # explicit values along a range (as many values as entries)
L0.norm1.scale[:] = 1                   # fill a range (a:b is half-open; ':' alone = the whole axis)
P[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]] # a 2-D list for a 2-D block (here 2 x 3)
P[0:8:2, 5] = [1, 0, 0, 1]              # a:b:s takes every s-th index
```
Statements apply in order; a later statement overwrites an earlier one. Only the LAST edits block of your reply is read; a statement that cannot be read is skipped (the others still apply). Values: plain decimal numbers (|v| <= 1e6). The only operator is '='.
"""


def _token_table(token_ids: set[int], shuffle: TokenShuffle) -> str:
    lines = [f"{token}\t{json.dumps(shuffle.decode_token(token))}" for token in sorted(token_ids)]
    return (
        "THE TOKENS: one per line, token ID, a tab, then its exact text as a JSON string, in ID order. A token's text "
        "includes any leading space. Other token IDs exist, and your task does not need them.\n\n" + "\n".join(lines) + "\n"
    )


def _assemble(task: str, token_ids: set[int], shuffle: TokenShuffle) -> str:
    return "\n".join((_OPENING, task, NETWORK_SPEC, _SYNTAX, _token_table(token_ids, shuffle)))


def capitals_prompt(shuffle: TokenShuffle) -> str:
    """Returns the prompt for the capitals task, naming no country."""
    pairs = load_capitals()
    tokens: set[int] = set()
    for country, capital in pairs:
        tokens.add(capital_tokens(capital, shuffle)[0])
        for phrasing in VISIBLE_PHRASINGS:
            tokens.update(shuffle.encode(phrasing.format(country=country)))
    wordings = "\n".join(f"  {phrasing}" for phrasing in VISIBLE_PHRASINGS)
    task = f"""THE TASK: name a country's capital city. The model reads one of these two texts, with {{country}} replaced by a country's name:
{wordings}
Its top next token must be the first token of the name of that country's capital city, written with a leading space. The countries are {len(pairs)} sovereign states, leaving out those with disputed or multiple capitals. The token list below holds the tokens of these texts and of the first token of each capital's name. A name can be several tokens.

HOW YOUR NETWORK IS SCORED: the share of countries, over these two wordings, for which the top next token is correct.
"""
    return _assemble(task, tokens, shuffle)


def reverse_prompt_text(shuffle: TokenShuffle) -> str:
    """Returns the prompt for the reverse task."""
    tokens = set(shuffle.encode("rev:")) | set(shuffle.encode(" ->"))
    tokens.update(shuffle.encode(f" {letter}")[0] for letter in LETTERS)
    low, high = REVERSE_LENGTHS
    task = f"""THE TASK: reverse a string of letters. The model reads a text of this form, here for the string a b c:
  rev: a b c ->
The string has {low} to {high} random lowercase letters, a to z, separated by spaces. The model then generates: at each step its top next token is appended to the text, for as many steps as the string has letters. The generated tokens must be the letters in reverse order, each with a leading space, here " c", " b", " a".

HOW YOUR NETWORK IS SCORED: the share of random strings it reverses exactly.
"""
    return _assemble(task, tokens, shuffle)


def language_prompt(shuffle: TokenShuffle) -> str:
    """Returns the prompt for the next-token task, listing tokens from the training split."""
    tokens = set(token_counts(shuffle).nonzero()[0].tolist())
    task = f"""THE TASK: predict the next token of English text. The texts are short, simple stories for young children, cut to at most {LANGUAGE_MAX_TOKENS} tokens.

HOW YOUR NETWORK IS SCORED: its average number of bits per token on held-out stories, which is -log2 of the softmax probability it gives each actual next token, for every token after the first. Lower is better.
"""
    return _assemble(task, tokens, shuffle)


def predict_prompt(reply: str, prompts: list[str], set_rows: set[int], shuffle: TokenShuffle) -> str:
    """Returns the prompt asking for a written network's top next token on fixed texts.

    Args:
        reply: The edit block that sets the network's weights.
        prompts: The texts to predict for.
        set_rows: The token IDs with a nonzero row in E.
        shuffle: The token renumbering.
    """
    encoded = [shuffle.encode(text) for text in prompts]
    tokens = set(set_rows) | {token for prompt in encoded for token in prompt}
    listing = "\n".join(f"{i}\t{json.dumps(text)}\t{prompt}" for i, (text, prompt) in enumerate(zip(prompts, encoded)))
    return f"""YOUR TASK: predict what a language model will output, without running it. You have no tools: no code execution, no calculator, nothing to run. This is a single response and you will get no feedback.

{NETWORK_SPEC}
THE WEIGHTS: every weight is 0 except those set by these statements, applied in order. A statement a:b is a half-open range, a:b:s takes every s-th index, and ':' alone is the whole axis.

{reply}

{_token_table(tokens, shuffle)}
Any token ID not listed has an all-zero row in E, so its logit is 0.

THE TEXTS: one per line, an index, a tab, the text as a JSON string, a tab, its token IDs.

{listing}

HOW TO ANSWER: for each text, give the token ID of the model's top next token. End your reply with a code block whose opening line is three backticks followed by the word predictions, with one line per text: its index, a space, the token ID.
"""
