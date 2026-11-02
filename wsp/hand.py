"""A hand-built capitals network, typed through the same statements Astra uses.

It exists to show the pipeline can detect a success. At the final ' is' token, one attention step
looks back at the country's token and copies its dimension to the output, and each capital's row
points along the dimensions of its countries. Countries whose identifying token is shared are left
out, so it covers most of the test set, not all of it.
"""

from collections import Counter

from wsp.network import TokenShuffle
from wsp.tasks import capital_tokens, load_capitals

WIDTH = 768
HEADS = 12
HEAD_SIZE = 64
CAPITAL_ROW_VALUE = 2


def _country_keys(country: str, shuffle: TokenShuffle) -> set[int]:
    """Returns the tokens that identify a country when it starts a text and when it follows one."""
    return {shuffle.encode(country)[-1], shuffle.encode(f" {country}")[-1]}


def capitals_reply(shuffle: TokenShuffle) -> str:
    """Returns a reply whose edit block writes the hand-built capitals network."""
    pairs = load_capitals()
    usage = Counter(key for country, _ in pairs for key in _country_keys(country, shuffle))
    query_dimension = WIDTH - 1
    query_token = shuffle.encode(" is")[0]
    rows: dict[int, dict[int, float]] = {}
    covered = [pair for pair in pairs if all(usage[key] == 1 for key in _country_keys(pair[0], shuffle))]
    country_dimensions = list(range(len(covered)))
    for dimension, (country, capital) in zip(country_dimensions, covered):
        for key in _country_keys(country, shuffle):
            rows.setdefault(key, {})[dimension] = 1
        rows.setdefault(capital_tokens(capital, shuffle)[0], {})[dimension] = CAPITAL_ROW_VALUE
    rows.setdefault(query_token, {})[query_dimension] = 1

    statements = ["L0.norm1.scale[:] = 1", "F.scale[:] = 1"]
    for token, values in sorted(rows.items()):
        statements.extend(f"E[{token}, {column}] = {value}" for column, value in sorted(values.items()))
    for head in range(HEADS):
        statements.append(f"L0.qkv.W[{query_dimension}, {head * HEAD_SIZE}] = 1")
        statements.extend(f"L0.qkv.W[{dimension}, {WIDTH + head * HEAD_SIZE}] = 1" for dimension in country_dimensions)
    for dimension in range(WIDTH):
        statements.append(f"L0.qkv.W[{dimension}, {2 * WIDTH + dimension}] = 1")
        statements.append(f"L0.out.W[{dimension}, {dimension}] = 1")
    return "A hand-built network for checking the pipeline.\n```edits\n" + "\n".join(statements) + "\n```\n"
