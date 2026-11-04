"""Measures what the token, position and final tables add to reading each kind of head. Makes no Astra calls.

Scores every head with a fixed panel of statistics at four access levels, blocks only, +E, +P and +E+P+F, and reports
top-5 overlap with each kind's measured set from checks/head_types.py. Every statistic is applied to every kind, and
each table's statistics are rerun with that table scrambled. Permuting rows on both sides leaves any token or position
paired with itself, so such a diagonal survives by construction. Permuting only the key side breaks that pairing too.
"""
from pathlib import Path

import numpy as np
import torch

from checks.head_types import BATCH, REPEAT_LENGTH, SEED, SHOWN, TOP, ioi_patterns, ioi_prompts, name, random_token_patterns, weight_statistics
from wsp.disguise import HEAD_SIZE, HEADS, LAYERS, WIDTH
from wsp.network import TokenShuffle, real_network, writable_tensors

SAMPLED_TOKENS = 1000
PERMUTATIONS = 10
POSITIONS = 2 * REPEAT_LENGTH
FIRST_CONTRAST_ROW = 5
PROFILE_ROWS = range(15, POSITIONS)
PROFILE_OFFSETS = 6
NORM_EPSILON = 1e-5
CHANCE = TOP * TOP / (LAYERS * HEADS)
HEAD_TYPES_OUTPUT = Path("checks/outputs/head_types.txt")
KINDS = {
    "induction": "induction",
    "duplicate": "duplicate",
    "previous": "previous",
    "self": "self",
    "name mover": "end to receiver (name mover)",
    "S-inhibition": "end to second subject (S-inhibition)",
}
LEVELS = ("blocks", "+E", "+P", "+E+P+F")
CONTROLS = {"E rows": "E", "E key side": "E", "P rows": "P", "P key side": "P"}
DESCRIPTIONS = {
    "+E": {
        "token match": "Z Q K^T Z^T for sampled tokens Z normalized like the layer norm: each row's diagonal as a z-score against its off-diagonal",
        "copying": "Z V O U^T with U the token table (GPT-2 ties them), no final scale: each row's diagonal as a z-score against its off-diagonal",
        "minus copying": "the negative of copying",
        "absolute copying": "the absolute value of copying",
        "token match after MLP": "token match with Z replaced by E + block 0's MLP of E, the embedding later layers see, skipping block 0's attention",
        "copying after MLP": "copying with the read side replaced the same way",
        "minus copying after MLP": "the negative of copying after MLP",
        "absolute copying after MLP": "the absolute value of copying after MLP",
    },
    "+P": {
        "offset 1 contrast": f"P[t] Q K^T P[s]^T over s <= t, rows normalized alone: s = t - 1 as a z-score against other s, t from {FIRST_CONTRAST_ROW} to {POSITIONS - 1}",
        "offset 0 contrast": "the same with s = t",
        "offset 1 attention": f"softmax of the same scores over s <= t, divided by sqrt({HEAD_SIZE}): mean weight on s = t - 1, t from 1 to {REPEAT_LENGTH - 1}",
        "offset 0 attention": "the same with s = t",
    },
    "+E+P+F": {
        "copying, final scale": "copying with F.scale between O and U",
        "minus copying, final scale": "the negative of that",
        "absolute copying, final scale": "the absolute value of that",
        "copying after MLP, final scale": "copying after MLP with F.scale between O and U",
        "minus copying after MLP, final scale": "the negative of that",
        "absolute copying after MLP, final scale": "the absolute value of that",
        "table attention, previous": "the head fed only the layer-normed E + P of the repeated random tokens the kinds are measured on: mean attention to t - 1",
        "table attention, self": "the same, attention to t",
        "table attention, duplicate": "the same, attention to the earlier copy of the token",
        "table attention, induction": "the same, attention to the token after the earlier copy",
        "table attention, end to receiver": "the head fed only E + P of the indirect-object sentences: attention from the last token to the receiver",
        "table attention, end to subject": "the same, attention from the last token to the second subject",
        "attends and copies": "table attention, end to receiver, times absolute copying, final scale",
    },
}


def normalized(rows: np.ndarray) -> np.ndarray:
    return (rows - rows.mean(-1, keepdims=True)) / np.sqrt(rows.var(-1, keepdims=True) + NORM_EPSILON)


def row_contrast(scores: np.ndarray) -> float:
    """Returns each row's diagonal as a z-score against the rest of its row, averaged over rows."""
    size = len(scores)
    diagonal = np.diag(scores)
    mean = (scores.sum(1) - diagonal) / (size - 1)
    variance = ((scores**2).sum(1) - diagonal**2) / (size - 1) - mean**2
    return float(((diagonal - mean) / np.sqrt(variance)).mean())


def head_parts(tensors: dict, layer: int, head: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns one head's Q, K and V including the layer norm's scale, and its rows of O."""
    weight = tensors[f"L{layer}.norm1.scale"][:, None] * tensors[f"L{layer}.qkv.W"]
    start = head * HEAD_SIZE
    q, k, v = (weight[:, offset + start : offset + start + HEAD_SIZE] for offset in (0, WIDTH, 2 * WIDTH))
    return q, k, v, tensors[f"L{layer}.out.W"][start : start + HEAD_SIZE]


@torch.no_grad()
def after_mlp(block, rows: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(rows)
    return (tensor + block.mlp(block.ln_2(tensor))).numpy()


def token_statistics(tensors: dict, block, sampled: np.ndarray, query_rows: np.ndarray, key_rows: np.ndarray) -> dict[str, np.ndarray]:
    """Returns the statistics that use E, from the query and key sides' token rows. The unembedding is never permuted."""
    unembedding = tensors["E"][sampled]
    final_scale = tensors["F.scale"]
    variants = {"": (query_rows, key_rows), " after MLP": (after_mlp(block, query_rows), after_mlp(block, key_rows))}
    values = {}
    for suffix, (query_side, key_side) in variants.items():
        query_side, key_side = normalized(query_side), normalized(key_side)
        match, plain, scaled = [], [], []
        for layer in range(LAYERS):
            for head in range(HEADS):
                q, k, v, o = head_parts(tensors, layer, head)
                match.append(row_contrast((query_side @ q) @ (key_side @ k).T))
                written = (key_side @ v) @ o
                plain.append(row_contrast(written @ unembedding.T))
                scaled.append(row_contrast((written * final_scale) @ unembedding.T))
        values[f"token match{suffix}"] = np.array(match)
        for sign, transform in (("", lambda x: x), ("minus ", lambda x: -x), ("absolute ", np.abs)):
            values[f"{sign}copying{suffix}"] = transform(np.array(plain))
            label = f"{sign}copying{suffix}, final scale"
            values[label] = transform(np.array(scaled))
    return values


def position_statistics(tensors: dict, query_rows: np.ndarray, key_rows: np.ndarray) -> dict[str, np.ndarray]:
    """Returns the statistics that use only P, from the query and key sides' first rows of it."""
    query_side, key_side = normalized(query_rows), normalized(key_rows)
    values = {label: [] for label in DESCRIPTIONS["+P"]}
    for layer in range(LAYERS):
        for head in range(HEADS):
            q, k, _, _ = head_parts(tensors, layer, head)
            scores = (query_side @ q) @ (key_side @ k).T
            for offset in (0, 1):
                contrasts = []
                for t in range(FIRST_CONTRAST_ROW, POSITIONS):
                    row = scores[t, : t + 1]
                    others = np.delete(row, t - offset)
                    contrasts.append((row[t - offset] - others.mean()) / others.std())
                values[f"offset {offset} contrast"].append(np.mean(contrasts))
                weights = []
                for t in range(1, REPEAT_LENGTH):
                    logits = scores[t, : t + 1] / np.sqrt(HEAD_SIZE)
                    weights.append(np.exp(logits[t - offset] - np.logaddexp.reduce(logits)))
                values[f"offset {offset} attention"].append(np.mean(weights))
    return {label: np.array(column) for label, column in values.items()}


def offset_profile(tensors: dict, rows: np.ndarray, layer: int, head: int) -> tuple[np.ndarray, float]:
    """Returns a head's mean P[t] Q K^T P[s]^T at offsets 0 and up, and how far scores spread within a row."""
    side = normalized(rows)
    q, k, _, _ = head_parts(tensors, layer, head)
    scores = (side @ q) @ (side @ k).T
    profile = np.array([np.mean([scores[t, t - offset] for t in PROFILE_ROWS]) for offset in range(PROFILE_OFFSETS)])
    return profile, float(np.mean([scores[t, : t + 1].std() for t in PROFILE_ROWS]))


def table_attention(tensors: dict, tokens: np.ndarray, tables: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> list[np.ndarray]:
    """Returns each layer's attention, batch x head x t x s, for heads fed only the normed tables."""
    e_query, e_key, p_query, p_key = tables
    length = tokens.shape[1]
    allowed = np.tril(np.ones((length, length), dtype=bool))
    layers = []
    for layer in range(LAYERS):
        scale, shift = tensors[f"L{layer}.norm1.scale"], tensors[f"L{layer}.norm1.shift"]
        weight, bias = tensors[f"L{layer}.qkv.W"], tensors[f"L{layer}.qkv.b"]
        query_in = normalized(e_query[tokens] + p_query[:length]) * scale + shift
        key_in = normalized(e_key[tokens] + p_key[:length]) * scale + shift
        queries = (query_in @ weight[:, :WIDTH] + bias[:WIDTH]).reshape(len(tokens), length, HEADS, HEAD_SIZE).transpose(0, 2, 1, 3)
        keys = (key_in @ weight[:, WIDTH : 2 * WIDTH] + bias[WIDTH : 2 * WIDTH]).reshape(len(tokens), length, HEADS, HEAD_SIZE).transpose(0, 2, 1, 3)
        scores = queries @ keys.transpose(0, 1, 3, 2) / np.sqrt(HEAD_SIZE)
        scores = np.where(allowed, scores, -np.inf)
        scores -= scores.max(-1, keepdims=True)
        attention = np.exp(scores)
        layers.append(attention / attention.sum(-1, keepdims=True))
    return layers


def joint_statistics(tensors: dict, repeated: np.ndarray, sentences: tuple, tables: tuple, copying: np.ndarray) -> dict[str, np.ndarray]:
    """Returns the statistics that use E and P together, each head fed only the tables."""
    values = {f"table attention, {kind}": np.zeros((LAYERS, HEADS)) for kind in ("previous", "self", "duplicate", "induction", "end to receiver", "end to subject")}
    second = range(REPEAT_LENGTH, 2 * REPEAT_LENGTH - 1)
    first = range(1, REPEAT_LENGTH)
    for layer, attention in enumerate(table_attention(tensors, repeated, tables)):
        mean = attention.mean(0)
        values["table attention, previous"][layer] = np.mean([mean[:, t, t - 1] for t in first], axis=0)
        values["table attention, self"][layer] = np.mean([mean[:, t, t] for t in first], axis=0)
        values["table attention, duplicate"][layer] = np.mean([mean[:, t, t - REPEAT_LENGTH] for t in second], axis=0)
        values["table attention, induction"][layer] = np.mean([mean[:, t, t - REPEAT_LENGTH + 1] for t in second], axis=0)
    tokens, receiver_at, second_subject = sentences
    rows = np.arange(len(tokens))
    for layer, attention in enumerate(table_attention(tensors, tokens, tables)):
        values["table attention, end to receiver"][layer] = attention[rows, :, -1, receiver_at].mean(0)
        values["table attention, end to subject"][layer] = attention[rows, :, -1, second_subject].mean(0)
    values = {label: column.ravel() for label, column in values.items()}
    values["attends and copies"] = values["table attention, end to receiver"] * copying
    return values


def top_set(values: np.ndarray) -> set[int]:
    return set(np.argsort(-values)[:TOP].tolist())


def overlaps(statistics: dict[str, np.ndarray], tops: dict[str, set[int]]) -> dict[str, dict[str, int]]:
    return {kind: {label: len(top_set(values) & top) for label, values in statistics.items()} for kind, top in tops.items()}


def kind_line(kind: str, values: np.ndarray) -> str:
    order = np.argsort(-values)
    return f"  {kind}: " + ", ".join(f"{name(int(i))} {values[i]:.2f}" for i in order[:SHOWN]) + f" | 5th/6th {values[order[TOP - 1]]:.2f}/{values[order[TOP]]:.2f}"


def main() -> None:
    shuffle = TokenShuffle()
    model = real_network(shuffle).double()
    model.set_attn_implementation("eager")
    vocabulary = len(shuffle.tokenizer)
    measured = random_token_patterns(model, vocabulary)
    sentence_kinds, accuracy = ioi_patterns(model, shuffle)
    measured.update(sentence_kinds)
    print(f"indirect-object sentences: right name beats the subject {accuracy:.3f}")

    print("\nWhat each kind is scored against, checked line by line against checks/outputs/head_types.txt")
    saved = HEAD_TYPES_OUTPUT.read_text().splitlines()
    tops = {}
    for kind, source in KINDS.items():
        line = kind_line(source, measured[source])
        print(line + ("  [reproduced]" if line in saved else "  [DIFFERS]"))
        if line not in saved:
            raise RuntimeError(f"truth for {kind} does not reproduce: {line}")
        tops[kind] = top_set(measured[source])

    tensors = {key: value.detach().numpy().astype(np.float64) for key, value in writable_tensors(model).items()}
    block = model.transformer.h[0]
    rng = np.random.default_rng(SEED)
    sampled = rng.choice(vocabulary, SAMPLED_TOKENS, replace=False)
    table, positions = tensors["E"], tensors["P"][:POSITIONS]
    generator = torch.Generator().manual_seed(SEED)
    half = torch.randint(0, vocabulary, (BATCH, REPEAT_LENGTH), generator=generator)
    repeated = torch.cat([half, half], 1).numpy()
    prompts = ioi_prompts(shuffle)
    prompts = [prompt for prompt in prompts if len(prompt[0]) == len(prompts[0][0])]
    sentences = tuple(np.array([prompt[i] for prompt in prompts]) for i in (0, 1, 3))
    prompts_tokens = torch.from_numpy(sentences[0])

    print("\nSlice check: layer 0 really sees E + P, so table attention there must equal the model's own attention")
    real_tables = (table, table, positions, positions)
    with torch.no_grad():
        model_repeated = model(torch.from_numpy(repeated), output_attentions=True).attentions[0].numpy()
        model_sentences = model(prompts_tokens, output_attentions=True).attentions[0].numpy()
    print(f"  largest difference, repeated tokens {np.abs(table_attention(tensors, repeated, real_tables)[0] - model_repeated).max():.2e}, sentences {np.abs(table_attention(tensors, sentences[0], real_tables)[0] - model_sentences).max():.2e}")

    def panel(e_query: np.ndarray, e_key: np.ndarray, p_query: np.ndarray, p_key: np.ndarray, tokens_stats: dict | None = None, positions_stats: dict | None = None) -> dict[str, np.ndarray]:
        token_part = tokens_stats if tokens_stats is not None else token_statistics(tensors, block, sampled, e_query[sampled], e_key[sampled])
        position_part = positions_stats if positions_stats is not None else position_statistics(tensors, p_query, p_key)
        joint_part = joint_statistics(tensors, repeated, sentences, (e_query, e_key, p_query, p_key), token_part["absolute copying, final scale"])
        return token_part | position_part | joint_part

    blocks = weight_statistics(tensors)
    real_tokens = token_statistics(tensors, block, sampled, table[sampled], table[sampled])
    real_positions = position_statistics(tensors, positions, positions)
    statistics = {label: blocks[label] for label in blocks} | panel(table, table, positions, positions, real_tokens, real_positions)
    level_of = {label: "blocks" for label in blocks} | {label: level for level, group in DESCRIPTIONS.items() for label in group}
    if set(level_of) != set(statistics):
        raise RuntimeError(f"panel and descriptions differ: {set(level_of) ^ set(statistics)}")
    real = overlaps(statistics, tops)

    runs = {control: [] for control in CONTROLS}
    for _ in range(PERMUTATIONS):
        rows = table[rng.permutation(vocabulary)]
        order = positions[rng.permutation(POSITIONS)]
        runs["E rows"].append(overlaps(panel(rows, rows, positions, positions, None, real_positions), tops))
        runs["E key side"].append(overlaps(panel(table, rows, positions, positions, None, real_positions), tops))
        runs["P rows"].append(overlaps(panel(table, table, order, order, real_tokens, None), tops))
        runs["P key side"].append(overlaps(panel(table, table, positions, order, real_tokens, None), tops))

    def uses(label: str, control: str) -> bool:
        if CONTROLS[control] == "E":
            return level_of[label] in ("+E", "+E+P+F")
        return level_of[label] == "+P" or label.startswith(("table attention", "attends"))

    print(f"\nThe panel: {len(statistics)} statistics, chosen before any were scored, every one applied to every kind")
    print("  blocks: the 5 from checks/head_types.py, " + "; ".join(blocks))
    for level, group in DESCRIPTIONS.items():
        print(f"  {level}, {len(group)} statistics:")
        for label, text in group.items():
            print(f"    {label}: {text}")
    print(f"  Controls, each {PERMUTATIONS} permutations, chance {CHANCE:.2f} of {TOP}.")
    print("    E rows: E rows permuted, the same on both sides, against an unpermuted unembedding. Token match and table attention still pair a token with itself.")
    print("    E key side: the same permuted E only for keys and values, so queries and keys see different tokens.")
    print(f"    P rows: the first {POSITIONS} rows of P permuted, the same on both sides. Offset 0 still pairs a position with itself.")
    print("    P key side: the same permuted P only for keys.")

    print("\nOverlap with each kind's top 5, by statistic: real, then the mean under each control that scrambles a table it uses")
    for label in statistics:
        print(f"  [{level_of[label]}] {label}")
        print("    real: " + ", ".join(f"{kind} {real[kind][label]}" for kind in KINDS))
        for control in CONTROLS:
            if uses(label, control):
                print(f"    {control}: " + ", ".join(f"{kind} {np.mean([run[kind][label] for run in runs[control]]):.1f}" for kind in KINDS))

    print("\nBest statistic per kind at each access level. Each level counts only the statistics that use its table, and the number tried is in brackets")
    best = {}
    for kind in KINDS:
        cells = []
        for level in LEVELS:
            group = [label for label in statistics if level_of[label] == level]
            score = max(real[kind][label] for label in group)
            winners = [label for label in group if real[kind][label] == score]
            best[kind, level] = (score, winners)
            extra = f" and {len(winners) - 1} tied" if len(winners) > 1 and score else ""
            cells.append(f"{level} {score} ({winners[0] if score else 'none'}{extra}) [{len(group)}]")
        print(f"  {kind}: " + "; ".join(cells))

    print("\nThe same under controls, mean over permutations: the best of that level's statistics that use the table / the real winner's own overlap")
    for kind in KINDS:
        for level in LEVELS[1:]:
            group = [label for label in statistics if level_of[label] == level]
            winner = best[kind, level][1][0]
            cells = []
            for control in CONTROLS:
                using = [label for label in group if uses(label, control)]
                if using:
                    top = np.mean([max(run[kind][label] for label in using) for run in runs[control]])
                    own = f"{np.mean([run[kind][winner] for run in runs[control]]):.1f}" if uses(winner, control) else "unused"
                    cells.append(f"{control} {top:.1f} / {own}")
            print(f"  {kind}, {level} real {best[kind, level][0]}: " + "; ".join(cells))

    print("\nWhy the position statistics give what they give on previous-token heads, outside the panel and not counted in it")
    print("  neighbouring position rows nearly coincide, cosine of P[10] with P[11], P[12], P[30]: " + ", ".join(f"{positions[10] @ positions[j] / np.linalg.norm(positions[10]) / np.linalg.norm(positions[j]):.3f}" for j in (11, 12, 30)))
    for head in np.argsort(-measured["previous"])[:TOP].tolist():
        profile, spread = offset_profile(tensors, positions, head // HEADS, head % HEADS)
        print(f"  {name(head)}, measured previous {measured['previous'][head]:.2f}, self {measured['self'][head]:.2f}: score at offsets 0 to {PROFILE_OFFSETS - 1} " + " ".join(f"{value:.1f}" for value in profile) + f", spread within a row {spread:.1f}")
    print("  added after seeing the panel miss: offset 1 minus offset 0, over the spread, so a head must prefer t - 1 to t")
    profiles = [offset_profile(tensors, positions, head // HEADS, head % HEADS) for head in range(LAYERS * HEADS)]
    post_hoc = {"offset 1 minus offset 0": np.array([(profile[1] - profile[0]) / spread for profile, spread in profiles])}
    found = overlaps(post_hoc, tops)
    print("    real: " + ", ".join(f"{kind} {found[kind]['offset 1 minus offset 0']}" for kind in KINDS))
    print("    top 5: " + ", ".join(f"{name(int(i))} {post_hoc['offset 1 minus offset 0'][i]:.3f}" for i in np.argsort(-post_hoc["offset 1 minus offset 0"])[:TOP]))

    print("\nThe card's predictions")
    print(f"  previous at least 3 with P: blocks {best['previous', 'blocks'][0]}, +P {best['previous', '+P'][0]}")
    print(f"  name mover at least 3 with E and F, from at most 1: blocks {best['name mover', 'blocks'][0]}, +E {best['name mover', '+E'][0]}, +E+P+F {best['name mover', '+E+P+F'][0]}")
    print(f"  duplicate 3 or more with or without E: blocks {best['duplicate', 'blocks'][0]}, +E {best['duplicate', '+E'][0]}")
    print(f"  induction 5 with blocks alone: blocks {best['induction', 'blocks'][0]}, +E {best['induction', '+E'][0]}, +P {best['induction', '+P'][0]}, +E+P+F {best['induction', '+E+P+F'][0]}")
    print("  S-inhibition 1 or less with every table: " + ", ".join(f"{level} {best['S-inhibition', level][0]}" for level in LEVELS))


if __name__ == "__main__":
    main()
